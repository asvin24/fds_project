"""Part 4: rescale the SELECTED bases (poly + Gaussian from Part 2) -- naive isotropic prior vs transformed prior."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from common import *

out = open(f"{RESULTS}/p4_log.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")

x, t = part2_data(42)
folds = list(KFold(5, shuffle=True, random_state=42).split(x))
xs = np.linspace(-1.5, 1.5, 301); inside = np.abs(xs) <= 1
f_s = f_part2(xs)
rng = np.random.default_rng(5)

sel = {}
for fam in ("poly", "gauss"):
    d = pd.DataFrame(part2_scan(fam, x, t, folds)); b = d.loc[d.log_ev.idxmax()]
    sel[fam] = (int(b.M), b.s, b.alpha, b.beta)
    log(f"selected {fam}: M={int(b.M)}, s={b.s}, alpha_hat={b.alpha:.3g}, beta_hat={b.beta:.3g}, logEv={b.log_ev:.3f}")

def scalings(M):
    m = np.arange(M)
    return {"uniform x3": np.full(M, 3.0), "decay exp(-0.25m)": np.exp(-0.25 * m),
            "grow exp(+0.25m)": np.exp(0.25 * m), "random lognormal(0,1)": np.exp(rng.normal(0, 1, M))}

rows = []; curves = {}
for fam in ("poly", "gauss"):
    M, s, alpha, beta = sel[fam]
    Phi = basis_for(fam, x, M, s); Ps = basis_for(fam, xs, M, s)
    base = fit_blr_qr(Phi, t, alpha, beta, Ps)
    ev_base_best = best_evidence(Phi, t, ALPHA_G2, BETA_G2)[0]
    rms_base = np.sqrt(np.mean((f_s[inside] - base["mu"][inside]) ** 2))
    for name, c in scalings(M).items():
        C = np.diag(c); Pc = Phi @ C; Psc = Ps @ C
        naive = fit_blr_qr(Pc, t, alpha, beta, Psc)                         # keep isotropic prior on w~
        inv = fit_general_prior(Pc, t, Psc, np.sqrt(alpha) * C, beta)       # Lambda0 = alpha C^2
        ev_re, a_re, b_re = best_evidence(Pc, t, ALPHA_G2, BETA_G2)         # naive, but (alpha,beta) re-optimised
        # non-diagonal check with the same C replaced by a general invertible T
        T = np.eye(M) + 0.09 * rng.normal(size=(M, M))
        PT = Phi @ T; PsT = Ps @ T
        invT = fit_general_prior(PT, t, PsT, np.sqrt(alpha) * T, beta)
        rows.append(dict(
            family=fam, scaling=name,
            ev_base=base["logev"], ev_naive=naive["logev"], ev_shift_naive=naive["logev"] - base["logev"],
            ev_naive_reopt_shift=ev_re - ev_base_best, alpha_reopt=a_re, beta_reopt=b_re,
            max_dmu_naive_in=np.max(np.abs(naive["mu"] - base["mu"])[inside]),
            max_dmu_naive_ex=np.max(np.abs(naive["mu"] - base["mu"])[~inside]),
            max_dsd_naive=np.max(np.abs(np.sqrt(naive["var"]) - np.sqrt(base["var"]))),
            rms_truth_base=rms_base, rms_truth_naive=np.sqrt(np.mean((f_s[inside] - naive["mu"][inside]) ** 2)),
            inv_dev=abs(inv["logev"] - base["logev"]), inv_dmu=np.max(np.abs(inv["mu"] - base["mu"])),
            inv_dvar=np.max(np.abs(inv["var"] - base["var"])), inv_dm=np.max(np.abs(C @ inv["m"] - base["m"])),
            invT_dev=abs(invT["logev"] - base["logev"]), invT_dmu=np.max(np.abs(invT["mu"] - base["mu"]))))
        curves[(fam, name)] = (base, naive, inv)

df = pd.DataFrame(rows); df.to_csv(f"{RESULTS}/p4_table.csv", index=False)
pd.set_option("display.width", 250)
log("\n=== Evidence & predictions under rescaling (alpha, beta held at their evidence optimum from the unscaled model) ===")
log(df[["family", "scaling", "ev_base", "ev_naive", "ev_shift_naive", "ev_naive_reopt_shift", "max_dmu_naive_in",
        "max_dmu_naive_ex", "max_dsd_naive", "rms_truth_base", "rms_truth_naive"]].round(4).to_string(index=False))
log("\n=== Transformed prior Lambda0 = alpha C^2 (and a non-diagonal T): deviations from the unscaled model ===")
log(df[["family", "scaling", "inv_dev", "inv_dmu", "inv_dvar", "inv_dm", "invT_dev", "invT_dmu"]].to_string(index=False,
    float_format=lambda v: f"{v:.2e}"))
log("\nUniform scaling c=3: re-optimising alpha recovers the baseline evidence exactly (alpha -> alpha/9); non-uniform scalings cannot be fixed this way (see ev_naive_reopt_shift).")
out.close()

# ---- figure
names = ["decay exp(-0.25m)", "grow exp(+0.25m)", "random lognormal(0,1)"]
fig, ax = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True)
for i, fam in enumerate(("poly", "gauss")):
    for j, nm in enumerate(names):
        base, naive, inv = curves[(fam, nm)]; a = ax[i, j]
        a.fill_between(xs, naive["mu"] - 2 * np.sqrt(naive["var"]), naive["mu"] + 2 * np.sqrt(naive["var"]), color=C2, alpha=.2, lw=0)
        a.plot(xs, naive["mu"], color=C2, lw=1.4, label="naive isotropic prior on $\\tilde w$")
        a.plot(xs, base["mu"], color=INK, lw=1.2, label="original model")
        a.plot(xs[::12], inv["mu"][::12], "o", ms=3.5, mfc="none", color=C3, label="transformed prior")
        a.axvspan(-1.5, -1, color="#f3f3f3"); a.axvspan(1, 1.5, color="#f3f3f3")
        a.set_ylim(-3, 3); style_axes(a); a.set_title(f"{fam}, {nm}", fontsize=8.5)
ax[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
plt.tight_layout(); plt.savefig(f"{RESULTS}/p4_predictions.png", dpi=160); plt.close()
