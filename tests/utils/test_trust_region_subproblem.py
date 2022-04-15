import pytest
import numpy as np
from scipy.stats import special_ortho_group

from ludfo.utils.trust_region import solve_tr_subproblem

def test_tr_zeros():
    dim = 3
    g = np.zeros(dim)
    Q = np.zeros((dim, dim))
    tol = 1e-8
    success, x, value = solve_tr_subproblem(g, Q, tol)
    print(x)
    assert success
    assert np.linalg.norm(x) < tol
    assert abs(g @ x + 0.5 * x.T @ Q @ x - value) < tol


def test_tr_zero_linear_distinct_positive_eigenvalues():
    dim = 4
    rot = special_ortho_group.rvs(dim)
    g = np.zeros(dim)
    Q = rot.T @ np.diag(list(range(1, dim + 1))) @ rot
    tol = 1e-8
    success, x, value = solve_tr_subproblem(g, Q, tol)
    print(x)
    assert success
    assert np.linalg.norm(x) < tol
    assert abs(g @ x + 0.5 * x.T @ Q @ x - value) < tol

