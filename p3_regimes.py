"""Part 3: three DGPs x sample size x noise level, evidence-selected poly / Gaussian models,
interpolation vs extrapolation, signal-MSE and predictive NLL, with R replicate datasets per cell."""
import numpy as np, pandas as pd, sys, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker
from common import *

R_REP = int(sys.argv[1]) if len(sys.argv) > 1 else 20
DGPS = ["smooth", "sharp", "hetero"]; NS = [20, 60, 200]; SIG = [0.05, 0.20, 0.50]
ALPHA_G = np.logspace(-4, 3, 29); BETA_G = np.logspace(-1, 3.5, 29)
POLY_C = [(M, None) for M in range(1, 17)]
GAUSS_C = [(M, s) for M in range(3, 31, 2) for s in (0.06, 0.12, 0.25, 0.45)]
X_IN = np.linspace(-1, 1, 400)
X_EX = np.concatenate([np.linspace(-1.5, -1.02, 100), np.linspace(1.02, 1.5, 100)])


def dgp(name, x, s0, rng=None):
    if name == "sharp":
        f = 0.4 * np.sin(np.pi * x) + 1.5 * np.exp(-((x - 0.25) ** 2) / (2 * 0.06 ** 2)); sg = np.full_like(x, s0)
    else:
        f = np.sin(np.pi * x) + 0.5 * np.cos(2 * np.pi * x)
        sg = np.full_like(x, s0) if name == "smooth" else s0 * (0.25 + 1.75 / (1 + np.exp(-5 * x)))
    t = None if rng is None else f + rng.normal(0, sg)
    return f, t, sg


def select(x, t, fam):
    best = (-np.inf,)
    for M, s in (POLY_C if fam == "poly" else GAUSS_C):
        ev, a, b = best_evidence(basis_for(fam, x, M, s), t, ALPHA_G, BETA_G)
        if ev > best[0]: best = (ev, M, s, a, b)
    return best


def metrics(fam, x, t, sel, xs, f, ts):
    ev, M, s, a, b = sel
    q = fit_blr_qr(basis_for(fam, x, M, s), t, a, b, basis_for(fam, xs, M, s))
    mse = np.mean((f - q["mu"]) ** 2)
    nll = 0.5 * np.mean(np.log(2 * np.pi * q["var"]) + (ts - q["mu"]) ** 2 / q["var"])
    cover = np.mean(np.abs(ts - q["mu"]) < 1.96 * np.sqrt(q["var"]))   # 95% predictive-interval coverage
    return mse, nll, cover, q


if __name__ == "__main__":
    rows = []; t0 = time.time(); rng = np.random.default_rng(123)
    examples = {}
    for dg in DGPS:
        for s0 in SIG:
            for N in NS:
                for rep in range(R_REP):
                    x = np.sort(rng.uniform(-1, 1, N)); _, t, _ = dgp(dg, x, s0, rng)
                    f_in, _, sg_in = dgp(dg, X_IN, s0); f_ex, _, sg_ex = dgp(dg, X_EX, s0)
                    t_in = f_in + rng.normal(0, sg_in); t_ex = f_ex + rng.normal(0, sg_ex)
                    for fam in ("poly", "gauss"):
                        sel = select(x, t, fam)
                        mi, ni, ci, _ = metrics(fam, x, t, sel, X_IN, f_in, t_in)
                        me, ne, ce, _ = metrics(fam, x, t, sel, X_EX, f_ex, t_ex)
                        rows.append(dict(dgp=dg, sigma0=s0, N=N, rep=rep, family=fam, M=sel[1], s=sel[2] or np.nan,
                                         alpha=sel[3], beta=sel[4], noise_sd_hat=sel[4] ** -0.5,
                                         mse_in=mi, nll_in=ni, cov_in=ci, mse_ex=me, nll_ex=ne, cov_ex=ce))
                    if rep == 0 and N == 60 and s0 == 0.20:
                        examples[dg] = (x, t, f_in, f_ex)
        print(dg, f"done {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{RESULTS}/p3_raw.csv", index=False)

    out = open(f"{RESULTS}/p3_log.txt", "w")
    def log(*a):
        s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")
    log(f"=== PART 3: {R_REP} replicate datasets per (dgp, sigma0, N, family); cells show MEDIAN [IQR] ===")
    agg = df.groupby(["dgp", "sigma0", "N", "family"]).agg(
        M=("M", "median"), s=("s", "median"), noise_sd_hat=("noise_sd_hat", "median"),
        mse_in=("mse_in", "median"), nll_in=("nll_in", "median"), cov_in=("cov_in", "mean"),
        mse_ex=("mse_ex", "median"), nll_ex=("nll_ex", "median"), cov_ex=("cov_ex", "mean")).reset_index()
    agg.to_csv(f"{RESULTS}/p3_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    log(agg.round(4).to_string(index=False))

    # ---- direct answers to the report's claims
    log("\n--- Effect of N (median over sigma0 and replicates) ---")
    for k in ("M", "mse_in", "nll_in", "mse_ex", "nll_ex"):
        log(f"{k}:\n" + df.groupby(["dgp", "family", "N"])[k].median().unstack("N").round(4).to_string())
    log("\n--- Effect of sigma0 (median over N and replicates) ---")
    for k in ("M", "mse_in", "nll_in", "mse_ex", "nll_ex"):
        log(f"{k}:\n" + df.groupby(["dgp", "family", "sigma0"])[k].median().unstack("sigma0").round(4).to_string())
    log("\n--- Evidence-estimated noise sd vs truth (smooth, sharp: sigma0; hetero: rms of sigma(x) over U[-1,1]) ---")
    xx = np.linspace(-1, 1, 10001)
    for s0 in SIG:
        rms = np.sqrt(np.mean(dgp("hetero", xx, s0)[2] ** 2))
        sub = df[(df.sigma0 == s0) & (df.N == 200)].groupby(["dgp", "family"]).noise_sd_hat.median()
        log(f"sigma0={s0}: hetero rms sigma(x)={rms:.3f}; N=200 estimated sd -> " + "; ".join(f"{a}/{b}:{v:.3f}" for (a, b), v in sub.items()))
    log("\n--- Poly vs Gauss: share of replicates where Gaussian wins (lower), by dgp ---")
    pv = df.pivot_table(index=["dgp", "sigma0", "N", "rep"], columns="family", values=["mse_in", "nll_in", "mse_ex", "nll_ex"])
    for k in ("mse_in", "nll_in", "mse_ex", "nll_ex"):
        w = (pv[(k, "gauss")] < pv[(k, "poly")]).groupby("dgp").mean()
        log(f"  {k}: " + ", ".join(f"{d}={v:.2f}" for d, v in w.items()))
    log("\n--- Hetero: 95% predictive-interval coverage on left (x<0) vs right (x>0) halves, N=200, sigma0=0.20 ---")
    rng2 = np.random.default_rng(7)
    for fam in ("poly", "gauss"):
        x = np.sort(rng2.uniform(-1, 1, 200)); _, t, _ = dgp("hetero", x, .2, rng2); sel = select(x, t, fam)
        f, _, sg = dgp("hetero", X_IN, .2); ts = f + rng2.normal(0, sg)
        q = fit_blr_qr(basis_for(fam, x, sel[1], sel[2]), t, sel[3], sel[4], basis_for(fam, X_IN, sel[1], sel[2]))
        inside = np.abs(ts - q["mu"]) < 1.96 * np.sqrt(q["var"])
        log(f"  {fam}: left coverage={inside[X_IN < 0].mean():.2f}, right coverage={inside[X_IN > 0].mean():.2f}, "
            f"mean predictive sd left={np.sqrt(q['var'][X_IN<0]).mean():.3f} (true {sg[X_IN<0].mean():.3f}), right={np.sqrt(q['var'][X_IN>0]).mean():.3f} (true {sg[X_IN>0].mean():.3f})")
    out.close()

    # ---- figure 1: metrics vs N
    cols = {"poly": C1, "gauss": C2}; ls = {0.05: ":", 0.20: "--", 0.50: "-"}
    mets = [("mse_in", "interp. signal MSE", True), ("nll_in", "interp. NLL", False),
            ("mse_ex", "extrap. signal MSE", True), ("nll_ex", "extrap. NLL", False)]
    fig, ax = plt.subplots(4, 3, figsize=(10.5, 10), sharex=True)
    med = df.groupby(["dgp", "family", "sigma0", "N"])[[m[0] for m in mets]].median().reset_index()
    for j, dg in enumerate(DGPS):
        for i, (k, lab, lg) in enumerate(mets):
            for fam in ("poly", "gauss"):
                for s0 in SIG:
                    d = med[(med.dgp == dg) & (med.family == fam) & (med.sigma0 == s0)]
                    ax[i, j].plot(d.N, d[k], ls[s0], color=cols[fam], lw=1.4, marker="o", ms=3.5,
                                  label=f"{fam}, $\\sigma_0$={s0}" if (i, j) == (0, 0) else None)
            if lg: ax[i, j].set_yscale("log")
            ax[i, j].set_xscale("log"); ax[i, j].set_xticks(NS); ax[i, j].set_xticklabels(NS); ax[i, j].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            style_axes(ax[i, j])
            if j == 0: ax[i, j].set_ylabel(lab, fontsize=8)
            if i == 0: ax[i, j].set_title(dg, fontsize=10)
            if i == 3: ax[i, j].set_xlabel("N")
    ax[0, 0].legend(fontsize=6.5, frameon=False, ncol=2)
    plt.tight_layout(); plt.savefig(f"{RESULTS}/p3_metrics.png", dpi=160); plt.close()

    # ---- figure 2: example predictive bands incl. extrapolation (N=60, sigma0=0.2)
    xg = np.linspace(-1.5, 1.5, 400)
    fig, ax = plt.subplots(2, 3, figsize=(11, 5.8), sharex=True)
    for j, dg in enumerate(DGPS):
        x, t, _, _ = examples[dg]
        fg, _, _ = dgp(dg, xg, 0.2)
        for i, fam in enumerate(("poly", "gauss")):
            sel = select(x, t, fam)
            q = fit_blr_qr(basis_for(fam, x, sel[1], sel[2]), t, sel[3], sel[4], basis_for(fam, xg, sel[1], sel[2]))
            sd = np.sqrt(q["var"])
            a = ax[i, j]; a.axvspan(-1.5, -1, color="#f3f3f3"); a.axvspan(1, 1.5, color="#f3f3f3")
            a.fill_between(xg, q["mu"] - 2 * sd, q["mu"] + 2 * sd, color=cols[fam], alpha=.2, lw=0)
            a.plot(xg, q["mu"], color=cols[fam], lw=1.4); a.plot(xg, fg, color=INK, lw=.9); a.plot(x, t, ".", ms=3.5, color=MUTED)
            a.set_ylim(-3.5, 3.5); style_axes(a)
            a.set_title(f"{dg} - {fam} (M={sel[1]}" + (f", s={sel[2]})" if sel[2] else ")"), fontsize=9)
    plt.tight_layout(); plt.savefig(f"{RESULTS}/p3_examples.png", dpi=160); plt.close()
