"""Part 2: polynomial vs Gaussian basis; selection by evidence vs by 5-fold CV (fixed and tuned lambda)."""
import numpy as np, pandas as pd, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from common import *

R_REP = int(sys.argv[1]) if len(sys.argv) > 1 else 30
out = open(f"{RESULTS}/p2_log.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")

x_test = np.linspace(-1, 1, 500); y_test = f_part2(x_test)


def select(scan):
    d = pd.DataFrame(scan)
    ev = d.loc[d.log_ev.idxmax()]; cf = d.loc[d.cv_fixed.idxmin()]; ct = d.loc[d.cv_tuned.idxmin()]
    return d, ev, cf, ct


def test_mse(family, x, t, M, s, alpha, beta=1.0):
    Phi = basis_for(family, x, M, s); m = fit_blr_qr(Phi, t, alpha, beta)["m"]
    return np.mean((y_test - basis_for(family, x_test, M, s) @ m) ** 2)


def run(seed):
    x, t = part2_data(seed)
    folds = list(KFold(5, shuffle=True, random_state=42).split(x))
    res = {}
    for fam in ("poly", "gauss"):
        d, ev, cf, ct = select(part2_scan(fam, x, t, folds))
        res[fam] = dict(
            table=d,
            ev=dict(M=int(ev.M), s=ev.s, alpha=ev.alpha, beta=ev.beta, score=ev.log_ev,
                    test=test_mse(fam, x, t, int(ev.M), ev.s, ev.alpha, ev.beta)),
            cvf=dict(M=int(cf.M), s=cf.s, score=cf.cv_fixed,
                     test=test_mse(fam, x, t, int(cf.M), cf.s, ALPHA_FIXED, BETA_FIXED)),
            cvt=dict(M=int(ct.M), s=ct.s, lam=ct.lam, score=ct.cv_tuned,
                     test=test_mse(fam, x, t, int(ct.M), ct.s, ct.lam)))
    return x, t, res


# ------------------------------------------------ main dataset (seed 42)
x, t, main = run(42)
log("=== MAIN DATASET (seed 42, N=60, sigma=0.20, beta_true=25) ===")
for fam in ("poly", "gauss"):
    r = main[fam]
    log(f"[{fam}] Evidence : M={r['ev']['M']:>2} s={r['ev']['s']} alpha={r['ev']['alpha']:.2e} beta={r['ev']['beta']:.1f} "
        f"logEv={r['ev']['score']:.2f} | test MSE={r['ev']['test']:.5f}")
    log(f"[{fam}] CV fixed : M={r['cvf']['M']:>2} s={r['cvf']['s']} (alpha0={ALPHA_FIXED}, beta0={BETA_FIXED}) "
        f"CV-MSE={r['cvf']['score']:.4f} | test MSE={r['cvf']['test']:.5f}")
    log(f"[{fam}] CV tuned : M={r['cvt']['M']:>2} s={r['cvt']['s']} lambda={r['cvt']['lam']:.2e} "
        f"CV-MSE={r['cvt']['score']:.4f} | test MSE={r['cvt']['test']:.5f}")
    r["table"].to_csv(f"{RESULTS}/p2_scan_{fam}.csv", index=False)

# ---- profiles vs M (best over s / over lambda)
fig, ax = plt.subplots(2, 3, figsize=(11, 5.6))
for row, fam in enumerate(("poly", "gauss")):
    d = main[fam]["table"]
    g = d.groupby("M")
    ev = g.log_ev.max(); cf = g.cv_fixed.min(); ct = g.cv_tuned.min()
    ax[row, 0].plot(ev.index, ev.values, "o-", ms=3.5, lw=1.4, color=C1)
    ax[row, 0].axvline(main[fam]["ev"]["M"], color=MUTED, ls=":", lw=1)
    ax[row, 0].set_ylabel("profiled log evidence"); ax[row, 0].set_title(f"{fam}: evidence (max over $\\alpha,\\beta$" + (", s)" if fam == "gauss" else ")"), fontsize=9)
    ax[row, 1].semilogy(cf.index, cf.values, "o-", ms=3.5, lw=1.4, color=C2, label=r"fixed $\alpha_0,\beta_0$")
    ax[row, 1].semilogy(ct.index, ct.values, "o-", ms=3.5, lw=1.4, color=C3, label=r"tuned $\lambda$")
    ax[row, 1].set_ylabel("5-fold CV MSE"); ax[row, 1].legend(fontsize=8, frameon=False)
    ax[row, 1].set_title(f"{fam}: validation error", fontsize=9)
    # fitted curves on the data
    ax[row, 2].plot(x, t, ".", ms=4, color=MUTED, alpha=.7)
    ax[row, 2].plot(x_test, y_test, color=INK, lw=1, label="truth")
    for key, col, lab in (("ev", C1, "evidence"), ("cvt", C3, "CV tuned"), ("cvf", C2, "CV fixed")):
        r = main[fam][key]
        alpha, beta = (r["alpha"], r["beta"]) if key == "ev" else ((r["lam"], 1.0) if key == "cvt" else (ALPHA_FIXED, BETA_FIXED))
        m = fit_blr_qr(basis_for(fam, x, r["M"], r["s"]), t, alpha, beta)["m"]
        ax[row, 2].plot(x_test, basis_for(fam, x_test, r["M"], r["s"]) @ m, color=col, lw=1.3, label=f"{lab} (M={r['M']})")
    ax[row, 2].set_ylim(-2, 2); ax[row, 2].legend(fontsize=7, frameon=False, ncol=2); ax[row, 2].set_title(f"{fam}: selected fits", fontsize=9)
for a_ in ax.ravel(): style_axes(a_); a_.set_xlabel("M" if a_ in ax[:, :2].ravel() else "x", fontsize=8)
plt.tight_layout(); plt.savefig(f"{RESULTS}/p2_profiles.png", dpi=170); plt.close()

# ---- Gaussian: evidence landscape over (M, s) and chosen s*Delta-mu relation
d = main["gauss"]["table"]
piv = d.pivot(index="s", columns="M", values="log_ev")
fig, ax = plt.subplots(figsize=(8, 2.6))
im = ax.imshow(np.clip(piv.values, piv.values.max() - 60, None), aspect="auto", cmap="viridis",
               extent=[GAUSS_M[0] - .5, GAUSS_M[-1] + .5, len(GAUSS_S) - .5, -.5])
ax.set_yticks(range(len(GAUSS_S))); ax.set_yticklabels(GAUSS_S); ax.set_xlabel("M"); ax.set_ylabel("bandwidth s")
plt.colorbar(im, label="log evidence (clipped at max-60)"); ax.set_title("Gaussian basis: profiled evidence over (M, s)", fontsize=9)
plt.tight_layout(); plt.savefig(f"{RESULTS}/p2_gauss_landscape.png", dpi=170); plt.close()
best = d.loc[d.log_ev.idxmax()]
log(f"Gaussian evidence optimum: M={int(best.M)}, s={best.s}, spacing Delta_mu=2/(M-2)={2/(best.M-2):.3f}, s/Delta_mu={best.s/(2/(best.M-2)):.2f}")

# ---- beta recovery by evidence as function of M (poly): shows underfit -> beta underestimated
dp = main["poly"]["table"].groupby("M").apply(lambda g: g.loc[g.log_ev.idxmax()], include_groups=False)
log("\nPoly: evidence-estimated beta per M (true 25):  " + ", ".join(f"M={m}:{b:.1f}" for m, b in zip(dp.index, dp.beta)))

# ---- conditioning numbers
log("\nCondition numbers on the training inputs:")
for M in (8, 12, 15, 16):
    log(f"  poly M={M}: kappa={np.linalg.cond(poly_basis(x, M)):.2e}")
for M, s in ((15, 0.25), (25, 0.25), (25, 0.65)):
    log(f"  gauss M={M}, s={s}: kappa={np.linalg.cond(gaussian_basis(x, M, s)):.2e}")

# ---- extrapolation snapshot (shows RBF reverting to bias vs polynomial divergence)
xe = np.linspace(-1.5, 1.5, 7)
log("\nPredictive std at x in [-1.5..1.5] for the evidence-selected models (shows variance behaviour outside data):")
for fam in ("poly", "gauss"):
    r = main[fam]["ev"]
    q = fit_blr_qr(basis_for(fam, x, r["M"], r["s"]), t, r["alpha"], r["beta"], basis_for(fam, xe, r["M"], r["s"]))
    log(f"  {fam:>5}: x={np.round(xe,2)} mu={np.round(q['mu'],2)} sd={np.round(np.sqrt(q['var']),2)}; "
        f"beta^-1/2={r['beta']**-.5:.3f}, alpha^-1/2={r['alpha']**-.5:.1f}")

# ------------------------------------------------ replicates (sampling variation of the Part-2 comparison)
log(f"\n=== REPLICATES: {R_REP} datasets (seeds 0..{R_REP-1}) ===")
rows = []
for sd in range(R_REP):
    _, _, r = run(sd)
    for fam in ("poly", "gauss"):
        for key, lab in (("ev", "evidence"), ("cvf", "cv_fixed"), ("cvt", "cv_tuned")):
            rows.append(dict(seed=sd, family=fam, criterion=lab, M=r[fam][key]["M"], s=r[fam][key]["s"], test_mse=r[fam][key]["test"]))
rep = pd.DataFrame(rows); rep.to_csv(f"{RESULTS}/p2_replicates.csv", index=False)
summ = rep.groupby(["family", "criterion"]).agg(M_median=("M", "median"), M_q25=("M", lambda v: np.percentile(v, 25)),
                                                M_q75=("M", lambda v: np.percentile(v, 75)),
                                                mse_median=("test_mse", "median"), mse_q25=("test_mse", lambda v: np.percentile(v, 25)),
                                                mse_q75=("test_mse", lambda v: np.percentile(v, 75)))
log(summ.round(5).to_string()); summ.to_csv(f"{RESULTS}/p2_replicate_summary.csv")
log("\nShare of datasets where each criterion gives the lowest test MSE within a family:")
for fam in ("poly", "gauss"):
    p = rep[rep.family == fam].pivot(index="seed", columns="criterion", values="test_mse")
    log(f"  {fam}: " + ", ".join(f"{c}={np.mean(p.idxmin(axis=1) == c):.2f}" for c in p.columns))
pm = rep.pivot_table(index=["seed", "criterion"], columns="family", values="test_mse")
log("Gaussian test-MSE < poly test-MSE (per criterion, share of datasets): " +
    ", ".join(f"{c}={np.mean(pm.xs(c, level='criterion').gauss < pm.xs(c, level='criterion').poly):.2f}" for c in ("evidence", "cv_fixed", "cv_tuned")))
out.close()
