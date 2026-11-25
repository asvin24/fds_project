"""Test MSE of the Part-2 selections on the main dataset (seed 42), recomputed for the table."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from common import *

x, t = part2_data(42)
x_test = np.linspace(-1, 1, 500); y_test = f_part2(x_test)


def test_mse_main(fam, crit, r):
    s = None if fam == "poly" else r.s
    M = int(r.M)
    if crit == "evidence": alpha, beta = r.alpha, r.beta
    elif crit == "cv_fixed": alpha, beta = ALPHA_FIXED, BETA_FIXED
    else: alpha, beta = r.lam, 1.0
    m = fit_blr_qr(basis_for(fam, x, M, s), t, alpha, beta)["m"]
    return np.mean((y_test - basis_for(fam, x_test, M, s) @ m) ** 2)
