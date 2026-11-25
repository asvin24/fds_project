"""Part 1: numerical verification of posterior / predictive / evidence and stability of factorisations."""
import numpy as np, mpmath as mp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from common import *

out = open(f"{RESULTS}/p1_log.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")

# ---------- (a) correctness on a well-conditioned problem
rng = np.random.default_rng(0)
N, M, alpha, beta = 40, 6, 0.7, 20.0
x = rng.uniform(-1, 1, N); Phi = legendre_basis(x, M); t = np.sin(3 * x) + rng.normal(0, .2, N)
xs = np.linspace(-1, 1, 7); Ps = legendre_basis(xs, M)

A = alpha * np.eye(M) + beta * Phi.T @ Phi
S = np.linalg.inv(A); m_ref = beta * S @ Phi.T @ t            # textbook (explicit inverse), fine here
mu_ref = Ps @ m_ref; var_ref = 1 / beta + np.einsum("ij,jk,ik->i", Ps, S, Ps)
C = np.eye(N) / beta + Phi @ Phi.T / alpha                       # marginal t ~ N(0, C)
ev_ref = multivariate_normal(mean=np.zeros(N), cov=C).logpdf(t)  # independent derivation of the evidence

q = fit_blr_qr(Phi, t, alpha, beta, Ps)
m_c, ev_c = fit_blr_cholesky(Phi, t, alpha, beta)
ev_fast = evidence_grid(Phi, t, [alpha], [beta])[0, 0]
log("== (a) well-conditioned check (N=40, M=6) ==")
log(f"log evidence  brute-force marginal N(0, I/b + PhiPhi'/a): {ev_ref:.10f}")
log(f"              QR {q['logev']:.10f} | Cholesky {ev_c:.10f} | SVD-grid {ev_fast:.10f}")
log(f"max |m_QR - m_ref| = {np.max(np.abs(q['m'] - m_ref)):.2e}; max |m_chol - m_ref| = {np.max(np.abs(m_c - m_ref)):.2e}")
log(f"max |mu_QR - mu_ref| = {np.max(np.abs(q['mu'] - mu_ref)):.2e}; max |var_QR - var_ref| = {np.max(np.abs(q['var'] - var_ref)):.2e}")

# Monte-Carlo check of predictive distribution: sample w from posterior, add noise
W = rng.multivariate_normal(m_ref, S, size=400000)
tsamp = W @ Ps[3] + rng.normal(0, beta ** -0.5, 400000)
log(f"MC predictive at x*={xs[3]:.2f}: mean {tsamp.mean():.4f} (formula {q['mu'][3]:.4f}), var {tsamp.var():.5f} (formula {q['var'][3]:.5f})")

# ---------- (b) stability vs conditioning: monomials, mpmath (60 digits) reference
mp.mp.dps = 60
alpha_b, beta_b = 1e-8, 25.0
Nb = 60
xb = np.sort(np.random.default_rng(1).uniform(-1, 1, Nb)); tb = f_part2(xb) + np.random.default_rng(2).normal(0, .2, Nb)
tm = mp.matrix([mp.mpf(float(v)) for v in tb])
rows = []
for Mb in range(4, 21, 2):
    Pb = poly_basis(xb, Mb)
    kappa = np.linalg.cond(Pb)
    Pm = mp.matrix(Mb and [[mp.mpf(float(xb[i])) ** j for j in range(Mb)] for i in range(Nb)])
    Am = mp.mpf(alpha_b) * mp.eye(Mb) + mp.mpf(beta_b) * (Pm.T * Pm)
    mm = mp.lu_solve(Am, mp.mpf(beta_b) * (Pm.T * tm))
    logdet = mp.log(mp.det(Am))
    resm = tm - Pm * mm
    E = beta_b / 2 * (resm.T * resm)[0] + mp.mpf(alpha_b) / 2 * (mm.T * mm)[0]
    ev_m = Mb / 2 * mp.log(alpha_b) + Nb / 2 * mp.log(beta_b) - E - logdet / 2 - Nb / 2 * mp.log(2 * mp.pi)
    mref = np.array([float(v) for v in mm]); nrm = np.linalg.norm(mref)
    qq = fit_blr_qr(Pb, tb, alpha_b, beta_b)
    err_qr_m = np.linalg.norm(qq["m"] - mref) / nrm; err_qr_ev = abs(qq["logev"] - float(ev_m))
    try:
        mc, evc = fit_blr_cholesky(Pb, tb, alpha_b, beta_b)
        err_ch_m = np.linalg.norm(mc - mref) / nrm; err_ch_ev = abs(evc - float(ev_m))
    except Exception:
        err_ch_m = err_ch_ev = np.nan
    try:
        Ai = np.linalg.inv(alpha_b * np.eye(Mb) + beta_b * Pb.T @ Pb); mi = beta_b * Ai @ Pb.T @ tb
        err_inv_m = np.linalg.norm(mi - mref) / nrm
    except Exception:
        err_inv_m = np.nan
    rows.append((Mb, kappa, err_qr_m, err_ch_m, err_inv_m, err_qr_ev, err_ch_ev))
log("\n== (b) accuracy vs 60-digit reference, monomials, alpha=1e-8 (nearly unregularised), N=60 ==")
log(f"{'M':>3} {'cond(Phi)':>10} | rel.err m_N: {'QR':>9} {'Cholesky':>9} {'inv(A)':>9} | abs.err log-ev: {'QR':>9} {'Cholesky':>9}")
for r in rows:
    log(f"{r[0]:>3} {r[1]:>10.2e} | {'':13}{r[2]:>9.1e} {r[3]:>9.1e} {r[4]:>9.1e} | {'':16}{r[5]:>9.1e} {r[6]:>9.1e}")

R = np.array(rows, float)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for k, (lab, col) in zip([2, 3, 4], [("augmented QR", C1), ("Cholesky on A", C2), ("explicit inv(A)", C3)]):
    ax[0].semilogy(R[:, 0], np.maximum(R[:, k], 1e-18), "o-", ms=4, lw=1.4, color=col, label=lab)
ax[0].set_xlabel("basis size M (monomials)"); ax[0].set_ylabel("relative error of $m_N$"); ax[0].legend(fontsize=8, frameon=False)
ax[0].set_title("Posterior-mean error vs 60-digit reference", fontsize=9)
ax[1].semilogy(R[:, 0], R[:, 1], "s-", ms=4, lw=1.4, color=MUTED, label=r"$\kappa(\Phi)$")
ax[1].semilogy(R[:, 0], R[:, 1] ** 2, "s--", ms=4, lw=1.4, color=C4, label=r"$\kappa(\Phi)^2=\kappa(\Phi^\top\Phi)$")
ax[1].set_xlabel("basis size M"); ax[1].legend(fontsize=8, frameon=False); ax[1].set_title("Conditioning", fontsize=9)
for a_ in ax: style_axes(a_)
plt.tight_layout(); plt.savefig(f"{RESULTS}/p1_stability.png", dpi=170); plt.close()

# ---------- (c) SVD-grid evidence == QR evidence on the Part-2 grid
x2, t2 = part2_data()
maxd = 0
for fam, M, s in [("poly", 8, None), ("poly", 14, None), ("gauss", 15, 0.25), ("gauss", 25, 0.65)]:
    Phi2 = basis_for(fam, x2, M, s)
    al = np.logspace(-5, 3, 9); be = np.logspace(-1, 3, 9)
    G = evidence_grid(Phi2, t2, al, be)
    Q = np.array([[fit_blr_qr(Phi2, t2, a_, b_)["logev"] for b_ in be] for a_ in al])
    maxd = max(maxd, np.max(np.abs(G - Q)))
log(f"\n== (c) SVD-grid evidence vs QR evidence on Part-2 bases: max |diff| = {maxd:.2e} (so the grid searches below are exact) ==")
out.close()
