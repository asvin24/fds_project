"""Part 5: reproducible settings in which evidence and cross-validation select substantially different models.

Design (all models nested: first M orthonormal Legendre columns, degrees 0..M-1; columns have unit RMS on U[-1,1]):
  Evidence : maximise log p(t | M, alpha, beta)  over M (and over alpha,beta on a grid if hyper-parameters are 'free';
             alpha,beta held at given constants if 'fixed').
  CV       : 5-fold squared-error CV over M (and over lambda = alpha/beta if 'free'; lambda = alpha0/beta0 if 'fixed').
  => in every arm both criteria are given the same prior information.
Experiments (R repeated datasets, seeds 0..R-1; selections compared PAIRED on identical datasets):
  E1  prior sweep       : in-class sparse truth, Gaussian noise, fixed isotropic prior precision alpha0 swept over 7 decades.
  E2  misspecification  : factorial ablation (mean in/out of class) x (noise Gaussian / heavy-tailed), hyper-parameters free.
  E3  sample-size sweep : misspecified-mean setting vs matched in-class control as N grows.
usage: python p5_disagree.py [R=100]
"""
import numpy as np, pandas as pd, sys
from scipy.stats import binomtest
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker
from sklearn.model_selection import KFold
from common import *

R = int(sys.argv[1]) if len(sys.argv) > 1 else 100
X_TEST = np.linspace(-1, 1, 500)
A_GRID = np.logspace(-3, 3, 25); B_GRID = np.logspace(0, 3.5, 22); L_GRID = np.logspace(-6, 2, 33)


def f_legendre(w):
    w = np.asarray(w, float)
    return lambda x: legendre_basis(x, len(w)) @ w

def gauss_noise(sd):
    return lambda rng, n: rng.normal(0, sd, n)

def heavy_noise(sd, sd_out, frac=0.10):
    def g(rng, n):
        out = rng.uniform(size=n) < frac
        return np.where(out, rng.normal(0, sd_out, n), rng.normal(0, sd, n))
    return g

F_SPARSE = f_legendre([0, .6, 0, .5, 0, 0, 0, 0, 0, .15])            # true M = 10, small high-order coefficient
F_INCLASS = f_legendre([0, .6, 0, .5, 0, .3])                        # true M = 6
F_ROUGH = lambda z: np.abs(z) ** 0.5                                 # not a polynomial (sqrt cusp at 0)


def one_dataset(cfg, seed):
    rng = np.random.default_rng(1000 + seed)
    N, M_grid = cfg["N"], cfg["M_grid"]
    x = np.sort(rng.uniform(-1, 1, N)); t = cfg["f"](x) + cfg["noise"](rng, N)
    folds = list(KFold(5, shuffle=True, random_state=seed).split(x))
    if cfg["mode"] == "fixed":
        alphas, betas, lams = [cfg["alpha0"]], [cfg["beta0"]], [cfg["alpha0"] / cfg["beta0"]]
    else:
        alphas, betas, lams = A_GRID, B_GRID, L_GRID
    best_ev = (-np.inf,); best_cv = (np.inf,)
    for M in M_grid:
        Phi = legendre_basis(x, M)
        ev, a, b = best_evidence(Phi, t, alphas, betas)
        if ev > best_ev[0] + 1e-12: best_ev = (ev, M, a, b)          # ties -> smaller M
        cv = ridge_cv_curve(Phi, t, folds, lams); k = int(np.argmin(cv))
        if cv[k] < best_cv[0] - 1e-12: best_cv = (cv[k], M, lams[k])
    yt = cfg["f"](X_TEST)
    mse = lambda M, alpha, beta: np.mean((yt - legendre_basis(X_TEST, M) @ fit_blr_qr(legendre_basis(x, M), t, alpha, beta)["m"]) ** 2)
    return dict(seed=seed, M_ev=best_ev[1], M_cv=best_cv[1], alpha_hat=best_ev[2], noise_sd_hat=best_ev[3] ** -0.5,
                mse_ev=mse(best_ev[1], best_ev[2], best_ev[3]), mse_cv=mse(best_cv[1], best_cv[2], 1.0))


def paired_stats(df, rng):
    d = (df.M_cv - df.M_ev).values.astype(float); nz = d[d != 0]
    p = binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue if len(nz) else 1.0
    bs = np.array([rng.choice(d, len(d)).mean() for _ in range(4000)])
    dm = (df.mse_cv - df.mse_ev).values
    return dict(Mev_mean=df.M_ev.mean(), Mev_sd=df.M_ev.std(ddof=1), Mev_med=df.M_ev.median(),
                Mcv_mean=df.M_cv.mean(), Mcv_sd=df.M_cv.std(ddof=1), Mcv_med=df.M_cv.median(),
                dM_mean=d.mean(), dM_sd=d.std(ddof=1), dM_ci_lo=np.percentile(bs, 2.5), dM_ci_hi=np.percentile(bs, 97.5),
                P_cv_gt_ev=np.mean(d > 0), P_cv_lt_ev=np.mean(d < 0), P_equal=np.mean(d == 0), sign_test_p=p,
                mse_ev_med=df.mse_ev.median(), mse_cv_med=df.mse_cv.median(), P_cv_lower_mse=np.mean(dm < 0),
                mse_diff_mean=dm.mean(), mse_diff_se=dm.std(ddof=1) / np.sqrt(len(dm)),
                noise_sd_hat_med=df.noise_sd_hat.median())


def run(cfg, label, rng, **extra):
    df = pd.DataFrame([one_dataset(cfg, s) for s in range(R)]); df["setting"] = label
    for k, v in extra.items(): df[k] = v
    st = paired_stats(df, rng); st.update(setting=label, **extra)
    print(f"{label:<26} M_ev={st['Mev_mean']:5.2f}±{st['Mev_sd']:.2f}  M_cv={st['Mcv_mean']:5.2f}±{st['Mcv_sd']:.2f}  "
          f"dM={st['dM_mean']:+.2f} [{st['dM_ci_lo']:+.2f},{st['dM_ci_hi']:+.2f}]  P(cv>ev)={st['P_cv_gt_ev']:.2f} P(=)={st['P_equal']:.2f}  "
          f"sign p={st['sign_test_p']:.1e}  MSE ev/cv={st['mse_ev_med']:.5f}/{st['mse_cv_med']:.5f} (cv lower in {st['P_cv_lower_mse']:.2f})", flush=True)
    return df, st


if __name__ == "__main__":
    rng = np.random.default_rng(0); raw = []; summ = []
    M16, M30 = list(range(1, 17)), list(range(1, 31))

    print(f"===== E1: prior-precision sweep (truth in class: M_true=10, N=60, sigma=0.15 known; R={R}) =====")
    A0 = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    for a0 in A0:
        cfg = dict(f=F_SPARSE, noise=gauss_noise(.15), N=60, M_grid=M16, mode="fixed", alpha0=a0, beta0=1 / .15 ** 2)
        d, s = run(cfg, f"E1 fixed alpha0={a0:g}", rng, exp="E1", alpha0=a0); raw.append(d); summ.append(s)
    cfg = dict(f=F_SPARSE, noise=gauss_noise(.15), N=60, M_grid=M16, mode="free")
    d, s = run(cfg, "E1 free (type-II ML)", rng, exp="E1", alpha0=np.nan); raw.append(d); summ.append(s)

    print(f"\n===== E2: misspecification ablation (N=200, M in 1..30, alpha,beta / lambda free; R={R}) =====")
    sd, sdo = 0.05, 0.5
    E2 = {"B0 in-class + Gaussian (control)": (F_INCLASS, gauss_noise(sd)),
          "B1 rough mean only": (F_ROUGH, gauss_noise(sd)),
          "B2 heavy-tail noise only": (F_INCLASS, heavy_noise(sd, sdo)),
          "B3 rough mean + heavy tails": (F_ROUGH, heavy_noise(sd, sdo))}
    for nm, (f, nz) in E2.items():
        d, s = run(dict(f=f, noise=nz, N=200, M_grid=M30, mode="free"), nm, rng, exp="E2", N=200); raw.append(d); summ.append(s)

    print(f"\n===== E3: sample-size sweep (free hyper-parameters; R={R}) =====")
    for nm, f in (("rough", F_ROUGH), ("in-class", F_INCLASS)):
        for N in (30, 60, 120, 200, 400):
            d, s = run(dict(f=f, noise=gauss_noise(.05), N=N, M_grid=M30, mode="free"), f"E3 {nm} N={N}", rng, exp="E3", N=N, truth=nm)
            raw.append(d); summ.append(s)

    pd.concat(raw).to_csv(f"{RESULTS}/p5_raw.csv", index=False)
    S = pd.DataFrame(summ); S.to_csv(f"{RESULTS}/p5_summary.csv", index=False)

    # ------------------------------------------------ figures
    RW = pd.concat(raw)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    e1 = S[(S.exp == "E1") & S.alpha0.notna()]; free = S[(S.exp == "E1") & S.alpha0.isna()].iloc[0]
    for key, col, lab in (("Mev", C1, "evidence"), ("Mcv", C2, "5-fold CV")):
        ax[0].plot(e1.alpha0, e1[f"{key}_mean"], "o-", color=col, lw=1.5, ms=4, label=lab)
        ax[0].fill_between(e1.alpha0, e1[f"{key}_mean"] - e1[f"{key}_sd"], e1[f"{key}_mean"] + e1[f"{key}_sd"], color=col, alpha=.15, lw=0)
        ax[0].axhline(free[f"{key}_mean"], color=col, ls=":", lw=1)
    ax[0].axhline(10, color=INK, lw=.8, ls="--"); ax[0].text(1.2e-4, 10.25, "true M = 10", fontsize=7, color=INK)
    ax[0].set_xscale("log"); ax[0].set_xlabel(r"fixed prior precision $\alpha_0$ (shared by both criteria)"); ax[0].set_ylabel("selected M (mean ± SD over datasets)")
    ax[0].legend(fontsize=8, frameon=False, loc="lower right"); ax[0].set_title("E1: vague prior → evidence under-selects (dotted: free hyper-parameters)", fontsize=8.5)
    ax[1].plot(e1.alpha0, e1.dM_mean, "o-", color=C3, lw=1.5, ms=4)
    ax[1].fill_between(e1.alpha0, e1.dM_ci_lo, e1.dM_ci_hi, color=C3, alpha=.2, lw=0)
    ax[1].axhline(0, color=MUTED, lw=.8); ax[1].set_xscale("log"); ax[1].set_xlabel(r"$\alpha_0$"); ax[1].set_ylabel(r"paired $\Delta M=M_{cv}-M_{ev}$ (mean, 95% bootstrap CI)")
    ax[1].set_title("systematic gap vs prior", fontsize=8.5)
    for a in ax: style_axes(a)
    plt.tight_layout(); plt.savefig(f"{RESULTS}/p5_prior_sweep.png", dpi=170); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.7))
    labs = list(E2); cols = [MUTED, C2, C4, C1]
    for i, nm in enumerate(labs):
        d = RW[RW.setting == nm]; dd = (d.M_cv - d.M_ev)
        vals, cnt = np.unique(dd, return_counts=True)
        ax[0].plot(vals + (i - 1.5) * 0.18, cnt / len(dd), "o-", ms=3, lw=1.2, color=cols[i], label=nm.split(" ", 1)[0] + " " + nm.split(" ", 1)[1][:22])
    ax[0].axvline(0, color=MUTED, lw=.8); ax[0].set_xlabel(r"paired $\Delta M = M_{cv}-M_{ev}$ per dataset"); ax[0].set_ylabel("share of datasets")
    ax[0].legend(fontsize=6.5, frameon=False); ax[0].set_title("E2: ablation, N=200", fontsize=9)
    for k, (truth, col) in enumerate((("rough", C2), ("in-class", MUTED))):
        e3 = S[(S.exp == "E3") & (S.truth == truth)]
        ax[1].errorbar(e3.N * (0.97), e3.Mev_mean, e3.Mev_sd, fmt="o-", color=C1 if truth == "rough" else "#8cb8d6", lw=1.3, ms=4, capsize=2, label=f"evidence, {truth}")
        ax[1].errorbar(e3.N * (1.03), e3.Mcv_mean, e3.Mcv_sd, fmt="s--", color=col, lw=1.3, ms=4, capsize=2, label=f"CV, {truth}")
    ax[1].set_xscale("log"); ax[1].set_xlabel("N"); ax[1].set_ylabel("selected M (mean ± SD)"); ax[1].legend(fontsize=7, frameon=False)
    ax[1].set_title("E3: selected complexity vs N", fontsize=9)
    e3 = S[(S.exp == "E3") & (S.truth == "rough")]
    ax[2].plot(e3.N, e3.mse_ev_med, "o-", color=C1, lw=1.4, ms=4, label="evidence-selected"); ax[2].plot(e3.N, e3.mse_cv_med, "s--", color=C2, lw=1.4, ms=4, label="CV-selected")
    ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("N"); ax[2].set_ylabel("median test MSE vs truth (rough)"); ax[2].legend(fontsize=8, frameon=False)
    ax[2].set_title("who generalises better?", fontsize=9)
    for a in ax[1:]:
        a.set_xticks([30, 60, 120, 200, 400]); a.set_xticklabels([30, 60, 120, 200, 400]); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for a in ax: style_axes(a)
    plt.tight_layout(); plt.savefig(f"{RESULTS}/p5_misspec.png", dpi=170); plt.close()

    with open(f"{RESULTS}/p5_log.txt", "w") as fh:
        pd.set_option("display.width", 250)
        cols_ = ["setting", "Mev_mean", "Mev_sd", "Mcv_mean", "Mcv_sd", "dM_mean", "dM_ci_lo", "dM_ci_hi", "P_cv_gt_ev", "P_equal", "sign_test_p", "mse_ev_med", "mse_cv_med", "P_cv_lower_mse", "mse_diff_mean", "mse_diff_se"]
        fh.write(S[cols_].round(4).to_string(index=False))
    print("\nwrote results/p5_summary.csv, p5_raw.csv, p5_log.txt, figures")
