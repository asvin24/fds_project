"""Builds compact report figures (PDF) and LaTeX table bodies from results/*.csv and logs."""
import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mt
from sklearn.model_selection import KFold
from common import *

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.3,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.1, "lines.markersize": 2.6})
W = 7.0  # inches, text width 17.8 cm


def sty(a):
    style_axes(a); a.tick_params(labelsize=6.5, length=2.5, width=.5)


# ------------------------------------------------------------------ Fig 1: verification + Part 2
rows = []
for ln in open(f"{RESULTS}/p1_log.txt"):
    m = re.match(r"\s*(\d+)\s+([\d.e+]+)\s+\|\s+([\d.e+-]+)\s+([\d.e+-]+|nan)\s+([\d.e+-]+|nan)\s+\|", ln)
    if m: rows.append([float(v) for v in m.groups()])
P1 = np.array(rows)
sp = pd.read_csv(f"{RESULTS}/p2_scan_poly.csv"); sg = pd.read_csv(f"{RESULTS}/p2_scan_gauss.csv")
fig, ax = plt.subplots(1, 3, figsize=(W, 1.6))
for k, (lab, col, mk) in zip([2, 3, 4], [("augmented QR", C1, "o"), ("Cholesky on A", C2, "s"), ("explicit inverse", C3, "^")]):
    ax[0].semilogy(P1[:, 0], np.maximum(P1[:, k], 1e-17), marker=mk, color=col, label=lab)
ax[0].set_xlabel("basis size M (monomials)"); ax[0].set_ylabel(r"relative error of $m_N$"); ax[0].legend(frameon=False, loc="upper left")
ax[0].set_title("(a) accuracy against 60-digit reference", loc="left")
for d, lab, col in ((sp, "polynomial", C1), (sg, "Gaussian", C2)):
    g = d.groupby("M"); ev = g.log_ev.max(); ct = g.cv_tuned.min()
    ax[1].plot(ev.index, ev.values - ev.max(), "o-", color=col, label=lab)
    ax[1].plot(ev.idxmax(), 0, "o", ms=6, mfc="none", color=col, mew=1)
    ax[2].semilogy(ct.index, ct.values, "o-", color=col, label=lab)
    ax[2].plot(ct.idxmin(), ct.min(), "o", ms=6, mfc="none", color=col, mew=1)
ax[1].set_ylim(-60, 3); ax[1].set_xlabel("basis size M"); ax[1].set_ylabel(r"$\ln p(\mathbf{t})$ minus its maximum"); ax[1].legend(frameon=False, loc="lower right")
ax[1].set_title("(b) profiled log evidence", loc="left")
ax[2].set_xlabel("basis size M"); ax[2].set_ylabel("5-fold CV MSE"); ax[2].legend(frameon=False, loc="upper right")
ax[2].set_title(r"(c) validation error, tuned $\lambda$", loc="left")
for a in ax: sty(a)
plt.tight_layout(pad=0.4, w_pad=0.8); plt.savefig(f"{OUT}/fig1.pdf"); plt.close()

# ------------------------------------------------------------------ Fig 2: Part 3
S3 = pd.read_csv(f"{RESULTS}/p3_summary.csv")
sub = S3[S3.sigma0 == 0.20]
mets = [("mse_in", "interpolation signal MSE", True), ("nll_in", "interpolation NLL", False),
        ("mse_ex", "extrapolation signal MSE", True), ("nll_ex", "extrapolation NLL", False)]
fig, ax = plt.subplots(1, 4, figsize=(W, 1.55))
ls = {"smooth": "-", "sharp": "--", "hetero": ":"}; col = {"poly": C1, "gauss": C2}
for j, (k, lab, lg) in enumerate(mets):
    for dg in ls:
        for fam in col:
            d = sub[(sub.dgp == dg) & (sub.family == fam)]
            ax[j].plot(d.N, d[k], ls[dg], color=col[fam], marker="o")
    if lg: ax[j].set_yscale("log")
    ax[j].set_xscale("log"); ax[j].set_xticks([20, 60, 200]); ax[j].set_xticklabels([20, 60, 200])
    ax[j].xaxis.set_minor_formatter(mt.NullFormatter()); ax[j].set_xlabel("N"); ax[j].set_title(lab, loc="left"); sty(ax[j])
from matplotlib.lines import Line2D
h = [Line2D([], [], color=col["poly"], lw=1.4), Line2D([], [], color=col["gauss"], lw=1.4)] + [Line2D([], [], color="#555", ls=ls[d], lw=1.4) for d in ls]
fig.legend(h, ["polynomial", "Gaussian", "smooth", "sharp", "hetero"], loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(pad=0.4, w_pad=0.8, rect=(0, 0.1, 1, 1)); plt.savefig(f"{OUT}/fig2.pdf"); plt.close()

# ------------------------------------------------------------------ Fig 3: Part 4
x, t = part2_data(42); folds = list(KFold(5, shuffle=True, random_state=42).split(x))
xs = np.linspace(-1.5, 1.5, 301); sel = {}
for fam in ("poly", "gauss"):
    d = pd.DataFrame(part2_scan(fam, x, t, folds)); b = d.loc[d.log_ev.idxmax()]; sel[fam] = (int(b.M), b.s, b.alpha, b.beta)
panels = [("poly", "decay", "(a) polynomial, decaying scale"), ("gauss", "decay", "(b) Gaussian, decaying scale"), ("gauss", "grow", "(c) Gaussian, growing scale")]
fig, ax = plt.subplots(1, 3, figsize=(W, 1.6), sharex=True)
for a, (fam, kind, title) in zip(ax, panels):
    M, s, alpha, beta = sel[fam]; m = np.arange(M); c = np.exp(-0.25 * m) if kind == "decay" else np.exp(0.25 * m); C = np.diag(c)
    Phi = basis_for(fam, x, M, s); Ps = basis_for(fam, xs, M, s)
    base = fit_blr_qr(Phi, t, alpha, beta, Ps); naive = fit_blr_qr(Phi @ C, t, alpha, beta, Ps @ C)
    inv = fit_general_prior(Phi @ C, t, Ps @ C, np.sqrt(alpha) * C, beta)
    a.axvspan(-1.5, -1, color="#f3f3f3"); a.axvspan(1, 1.5, color="#f3f3f3")
    a.fill_between(xs, naive["mu"] - 2 * np.sqrt(naive["var"]), naive["mu"] + 2 * np.sqrt(naive["var"]), color=C2, alpha=.2, lw=0)
    a.plot(xs, naive["mu"], color=C2, label=r"isotropic prior on $\tilde{w}$")
    a.plot(xs, base["mu"], color=INK, lw=0.9, label="original model")
    a.plot(xs[::14], inv["mu"][::14], "o", ms=2.6, mfc="none", color=C3, mew=.8, label="transformed prior")
    a.set_ylim(-3, 3); a.set_xlabel("x"); a.set_title(title, loc="left"); sty(a)
ax[0].legend(frameon=False, loc="lower left", fontsize=5.6); ax[0].set_ylabel("predictive mean")
plt.tight_layout(pad=0.4, w_pad=0.8); plt.savefig(f"{OUT}/fig3.pdf"); plt.close()

# ------------------------------------------------------------------ Fig 4: Part 5
S5 = pd.read_csv(f"{RESULTS}/p5_summary.csv")
fig, ax = plt.subplots(1, 3, figsize=(W, 1.6))
e1 = S5[(S5.exp == "E1") & S5.alpha0.notna()]; fr = S5[(S5.exp == "E1") & S5.alpha0.isna()].iloc[0]
for key, c_, lab, mk in (("Mev", C1, "evidence", "o"), ("Mcv", C2, "5-fold CV", "s")):
    ax[0].plot(e1.alpha0, e1[f"{key}_mean"], marker=mk, color=c_, label=lab)
    ax[0].fill_between(e1.alpha0, e1[f"{key}_mean"] - e1[f"{key}_sd"], e1[f"{key}_mean"] + e1[f"{key}_sd"], color=c_, alpha=.13, lw=0)
ax[0].axhline(10, color=INK, lw=.6, ls="--"); ax[0].text(1.3e-4, 10.3, "true M = 10", fontsize=6)
ax[0].set_xscale("log"); ax[0].set_xlabel(r"fixed prior precision $\alpha_0$"); ax[0].set_ylabel("selected M, mean and SD")
ax[0].legend(frameon=False, loc="lower right"); ax[0].set_title("(a) prior sweep", loc="left")
e2 = S5[S5.exp == "E2"].reset_index(drop=True); names = ["B0 control", "B1 mean", "B2 noise", "B3 both"]
ax[1].errorbar(e2.dM_mean, range(4), xerr=[e2.dM_mean - e2.dM_ci_lo, e2.dM_ci_hi - e2.dM_mean], fmt="o", color=C3, capsize=2, lw=1)
ax[1].set_yticks(range(4)); ax[1].set_yticklabels(names); ax[1].invert_yaxis(); ax[1].axvline(0, color=MUTED, lw=.6)
ax[1].set_xlabel(r"$\Delta M = M_{cv} - M_{ev}$, mean and 95% CI"); ax[1].set_title("(b) misspecification ablation", loc="left")
for truth, c_, m_ in (("rough", C1, "o"), ("in-class", "#8aa4b8", "s")):
    e3 = S5[(S5.exp == "E3") & (S5.truth == truth)]
    ax[2].errorbar(e3.N * 0.96, e3.Mev_mean, e3.Mev_sd, fmt=m_ + "-", color=c_, capsize=1.5, lw=1, label=f"evidence, {truth}")
    ax[2].errorbar(e3.N * 1.04, e3.Mcv_mean, e3.Mcv_sd, fmt=m_ + "--", color=C2 if truth == "rough" else "#e0b090", capsize=1.5, lw=1, label=f"CV, {truth}")
ax[2].set_xscale("log"); ax[2].set_xticks([30, 60, 120, 200, 400]); ax[2].set_xticklabels([30, 60, 120, 200, 400]); ax[2].xaxis.set_minor_formatter(mt.NullFormatter())
ax[2].set_xlabel("N"); ax[2].set_ylabel("selected M, mean and SD"); ax[2].legend(frameon=False, loc="upper left", fontsize=5.6); ax[2].set_title("(c) growth with sample size", loc="left")
for a in ax: sty(a)
plt.tight_layout(pad=0.4, w_pad=0.8); plt.savefig(f"{OUT}/fig4.pdf"); plt.close()

# ------------------------------------------------------------------ tables
def g(v, n=3): return f"{v:.{n}g}"
def sci(v):
    if v == 0: return "0"
    e = int(np.floor(np.log10(abs(v)))); m_ = v / 10 ** e
    return f"${m_:.1f}\\times10^{{{e}}}$"

# Table 1 (Part 2)
rep = pd.read_csv(f"{RESULTS}/p2_replicates.csv")
main = {}
for fam, d in (("poly", sp), ("gauss", sg)):
    ev = d.loc[d.log_ev.idxmax()]; cf = d.loc[d.cv_fixed.idxmin()]; ct = d.loc[d.cv_tuned.idxmin()]
    main[fam] = {"evidence": ev, "cv_fixed": cf, "cv_tuned": ct}
from p2_main_helper import test_mse_main  # noqa
L = []
for fam, nm in (("poly", "Polynomial"), ("gauss", "Gaussian")):
    for crit, cn in (("evidence", "Evidence"), ("cv_fixed", "CV, fixed $\\alpha_0,\\beta_0$"), ("cv_tuned", "CV, tuned $\\lambda$")):
        r = main[fam][crit]; sub_ = rep[(rep.family == fam) & (rep.criterion == crit)]
        Ms = f"{int(r.M)}" + (f" ({r.s:g})" if fam == "gauss" else "")
        mt_ = test_mse_main(fam, crit, r)
        q1, q3 = np.percentile(sub_.M, [25, 75]); m1, m3 = np.percentile(sub_.test_mse, [25, 75])
        L.append(f"{nm if crit == 'evidence' else ''} & {cn} & {Ms} & {mt_:.4f} & {sub_.M.median():g} [{q1:g}, {q3:g}] & {sub_.test_mse.median():.4f} [{m1:.4f}, {m3:.4f}] \\\\")
    if fam == "poly": L.append("\\midrule")
open(f"{OUT}/tab1.tex", "w").write("\n".join(L))

# Table 2 (Part 3)
T = S3[(S3.sigma0 == 0.20) & (S3.N == 60)]
L = []
for dg in ("smooth", "sharp", "hetero"):
    for fam, nm in (("poly", "polynomial"), ("gauss", "Gaussian")):
        r = T[(T.dgp == dg) & (T.family == fam)].iloc[0]
        L.append(f"{dg if fam == 'poly' else ''} & {nm} & {int(r.M)} & {r.mse_in:.4f} & {r.nll_in:.2f} & {g(r.mse_ex)} & {r.nll_ex:.2f} \\\\")
    if dg != "hetero": L.append("\\midrule")
open(f"{OUT}/tab2.tex", "w").write("\n".join(L))

# Table 3 (Part 4)
P4 = pd.read_csv(f"{RESULTS}/p4_table.csv")
nm = {"uniform x3": "uniform, $c=3$", "decay exp(-0.25m)": r"decay, $c_m=e^{-0.25m}$", "grow exp(+0.25m)": r"growth, $c_m=e^{0.25m}$", "random lognormal(0,1)": "random, log-normal"}
L = []
for sc in nm:
    a = P4[(P4.family == "poly") & (P4.scaling == sc)].iloc[0]; b = P4[(P4.family == "gauss") & (P4.scaling == sc)].iloc[0]
    L.append(f"{nm[sc]} & ${a.ev_shift_naive:.1f}$ & ${a.ev_naive_reopt_shift:.1f}$ & {g(a.max_dmu_naive_ex)} & ${b.ev_shift_naive:.1f}$ & ${b.ev_naive_reopt_shift:.1f}$ & {g(b.max_dmu_naive_ex)} \\\\")
open(f"{OUT}/tab3.tex", "w").write("\n".join(L))

# Table 4 (Part 5)
L = []
def row(r, name):
    f_ = lambda v: f"${v:+.1f}$"
    return (f"{name} & {r.Mev_mean:.1f} & {r.Mcv_mean:.1f} & {f_(r.dM_mean)} [{f_(r.dM_ci_lo)}, {f_(r.dM_ci_hi)}] & {r.P_cv_gt_ev:.2f} & {sci(r.mse_ev_med)} & {sci(r.mse_cv_med)} \\\\")
E1 = S5[S5.exp == "E1"]
for a0, lab in ((1e-4, r"E1, $\alpha_0=10^{-4}$"), (100.0, r"E1, $\alpha_0=100$")):
    L.append(row(E1[E1.alpha0 == a0].iloc[0], lab))
L.append(row(E1[E1.alpha0.isna()].iloc[0], "E1, free hyper-parameters"))
L.append("\\midrule")
E2 = S5[S5.exp == "E2"].reset_index(drop=True)
for i, lab in enumerate(["E2, B0 in-class control", "E2, B1 rough mean", "E2, B2 heavy-tailed noise", "E2, B3 rough mean and heavy tails"]):
    L.append(row(E2.iloc[i], lab))
open(f"{OUT}/tab4.tex", "w").write("\n".join(L))
print("assets ok")
