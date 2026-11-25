"""Shared core for Project 1: stable BLR, fast evidence grid, bases, CV, plotting style."""
import numpy as np
from scipy.linalg import qr, solve_triangular, cho_factor, cho_solve
from numpy.polynomial import legendre as _leg
import os

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

# colour-blind-safe categorical order (Okabe-Ito subset), recessive neutrals
C1, C2, C3, C4 = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e4e4e4"


def style_axes(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


# ---------------------------------------------------------------- bases
def poly_basis(x, M):
    return np.vander(np.asarray(x, float), N=M, increasing=True)


def gaussian_basis(x, M, s):
    """bias + (M-1) Gaussian RBFs; centres linspace(-1,1,M-1) (centre 0 if M==2)."""
    x = np.asarray(x, float)
    bias = np.ones((len(x), 1))
    if M == 1:
        return bias
    centers = np.array([0.0]) if M == 2 else np.linspace(-1.0, 1.0, M - 1)
    rbf = np.exp(-((x[:, None] - centers[None, :]) ** 2) / (2.0 * s ** 2))
    return np.hstack([bias, rbf])


def legendre_basis(x, M):
    """Orthonormal Legendre columns sqrt(2k+1) P_k(x): unit RMS under U[-1,1]."""
    P = _leg.legvander(np.asarray(x, float), M - 1)
    return P * np.sqrt(2.0 * np.arange(M) + 1.0)


# ---------------------------------------------------------------- reference QR engine
def fit_general_prior(Phi, t, Phi_star, Lam0_sqrt, beta):
    """BLR with prior w ~ N(0, Lam0^{-1}), Lam0 = Lam0_sqrt.T @ Lam0_sqrt, via augmented QR.
    Works for any (not only diagonal) square root."""
    N, M = Phi.shape
    Phi_aug = np.vstack([np.sqrt(beta) * Phi, Lam0_sqrt])
    t_aug = np.concatenate([np.sqrt(beta) * t, np.zeros(M)])
    Q, R = qr(Phi_aug, mode="economic")
    m = solve_triangular(R, Q.T @ t_aug, lower=False)
    logdet_A = 2.0 * np.sum(np.log(np.abs(np.diag(R))))
    sign, logdet_L = np.linalg.slogdet(Lam0_sqrt)
    logdet_Lam0 = 2.0 * logdet_L
    res = t - Phi @ m
    E = 0.5 * beta * res @ res + 0.5 * np.sum((Lam0_sqrt @ m) ** 2)
    logev = 0.5 * logdet_Lam0 + 0.5 * N * np.log(beta) - E - 0.5 * logdet_A - 0.5 * N * np.log(2 * np.pi)
    mu = var = None
    if Phi_star is not None:
        mu = Phi_star @ m
        v = solve_triangular(R, Phi_star.T, trans="T", lower=False)
        var = 1.0 / beta + np.sum(v ** 2, axis=0)
    return dict(m=m, logev=logev, mu=mu, var=var, R=R)


def fit_blr_qr(Phi, t, alpha, beta, Phi_star=None):
    M = Phi.shape[1]
    return fit_general_prior(Phi, t, Phi_star, np.sqrt(alpha) * np.eye(M), beta)


def fit_blr_cholesky(Phi, t, alpha, beta):
    N, M = Phi.shape
    A = alpha * np.eye(M) + beta * Phi.T @ Phi
    L, low = cho_factor(A, lower=True)
    m = cho_solve((L, low), beta * Phi.T @ t)
    logdetA = 2 * np.sum(np.log(np.diag(L)))
    res = t - Phi @ m
    E = 0.5 * beta * res @ res + 0.5 * alpha * m @ m
    logev = 0.5 * M * np.log(alpha) + 0.5 * N * np.log(beta) - E - 0.5 * logdetA - 0.5 * N * np.log(2 * np.pi)
    return m, logev


# ---------------------------------------------------------------- fast evidence over (alpha, beta) grid
def evidence_grid(Phi, t, alphas, betas):
    """Log evidence on the full (alpha x beta) grid from ONE thin SVD of Phi.
    Identical (up to rounding) to fit_blr_qr evaluated pointwise -- verified in p1_verify.py."""
    N, M = Phi.shape
    U, s, Vt = np.linalg.svd(Phi, full_matrices=False)
    r = len(s)
    c = U.T @ t
    rss_perp = np.sum((t - U @ c) ** 2)
    a = np.asarray(alphas, float)[:, None, None]
    b = np.asarray(betas, float)[None, :, None]
    sv2 = (s ** 2)[None, None, :]
    d = a + b * sv2
    w = b * s[None, None, :] * c[None, None, :] / d
    res_in = a * c[None, None, :] / d
    rss = rss_perp + np.sum(res_in ** 2, axis=-1)
    wn2 = np.sum(w ** 2, axis=-1)
    a2, b2 = a[..., 0], b[..., 0]
    logdetA = np.sum(np.log(d), axis=-1) + (M - r) * np.log(a2)
    E = 0.5 * b2 * rss + 0.5 * a2 * wn2
    return 0.5 * M * np.log(a2) + 0.5 * N * np.log(b2) - E - 0.5 * logdetA - 0.5 * N * np.log(2 * np.pi)


def best_evidence(Phi, t, alphas, betas):
    ev = evidence_grid(Phi, t, alphas, betas)
    i, j = np.unravel_index(np.argmax(ev), ev.shape)
    return ev[i, j], float(np.asarray(alphas)[i]), float(np.asarray(betas)[j])


# ---------------------------------------------------------------- ridge / CV (posterior mean depends only on lambda = alpha/beta)
def ridge_cv_curve(Phi, t, folds, lams):
    """K-fold CV mean squared error for each lambda; one SVD per fold, vectorised over lambda."""
    lams = np.atleast_1d(np.asarray(lams, float))
    sse = np.zeros(len(lams))
    for tr, va in folds:
        U, s, Vt = np.linalg.svd(Phi[tr], full_matrices=False)
        c = U.T @ t[tr]
        coef = (s[None, :] * c[None, :]) / (s[None, :] ** 2 + lams[:, None])  # (L, r)
        W = coef @ Vt                                                          # (L, M)
        pred = Phi[va] @ W.T                                                   # (nva, L)
        sse += np.sum((t[va][:, None] - pred) ** 2, axis=0)
    return sse / len(t)


def ridge_fit(Phi, t, lam):
    return fit_blr_qr(Phi, t, alpha=lam, beta=1.0)["m"]


# ---------------------------------------------------------------- Part 2 benchmark
def f_part2(x):
    return np.sin(np.pi * x) + 0.4 * np.cos(3 * np.pi * x)


def part2_data(seed=42, N=60, sigma=0.20):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1, 1, N))
    t = f_part2(x) + rng.normal(0, sigma, N)
    return x, t


POLY_M = list(range(1, 17))
GAUSS_M = list(range(2, 26))
GAUSS_S = [0.15, 0.25, 0.40, 0.65]
ALPHA_G2 = np.logspace(-5, 3, 41)
BETA_G2 = np.logspace(-1, 3, 41)
LAMBDA_G2 = np.logspace(-8, 2, 51)
ALPHA_FIXED, BETA_FIXED = 1e-2, 25.0


def basis_for(family, x, M, s):
    return poly_basis(x, M) if family == "poly" else gaussian_basis(x, M, s)


def part2_scan(family, x, t, folds):
    """Every candidate (M, s): profiled evidence, CV(fixed lam0), CV(tuned lam). Returns list of dicts."""
    cands = [(M, None) for M in POLY_M] if family == "poly" else [(M, s) for M in GAUSS_M for s in GAUSS_S]
    lam0 = ALPHA_FIXED / BETA_FIXED
    out = []
    for M, s in cands:
        Phi = basis_for(family, x, M, s)
        ev, a, b = best_evidence(Phi, t, ALPHA_G2, BETA_G2)
        cv_fixed = ridge_cv_curve(Phi, t, folds, [lam0])[0]
        cvt = ridge_cv_curve(Phi, t, folds, LAMBDA_G2)
        k = int(np.argmin(cvt))
        out.append(dict(M=M, s=s, log_ev=ev, alpha=a, beta=b, cv_fixed=cv_fixed,
                        cv_tuned=cvt[k], lam=LAMBDA_G2[k]))
    return out
