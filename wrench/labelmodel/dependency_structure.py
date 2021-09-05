"""
https://github.com/HazyResearch/metal/blob/cb_deps/tutorials/Learned_Deps.ipynb
"""

import cvxpy as cp
import numpy as np
import scipy as sp


def get_deps_from_inverse_sig(J, thresh=0.2):
    deps = []
    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            if abs(J[i, j]) > thresh:
                deps.append((i, j))
    return deps


def learn_structure(L, thresh=1.5):
    N = float(np.shape(L)[0])
    M = L.shape[1]
    sigma_O = (np.dot(L.T, L)) / (N - 1) - \
              np.outer(np.mean(L, axis=0), np.mean(L, axis=0))

    # bad code
    O = 1 / 2 * (sigma_O + sigma_O.T)
    O_root = np.real(sp.linalg.sqrtm(O))

    # low-rank matrix
    L_cvx = cp.Variable([M, M], PSD=True)

    # sparse matrix
    S = cp.Variable([M, M], PSD=True)

    # S-L matrix
    R = cp.Variable([M, M], PSD=True)

    # reg params
    lam = 1 / np.sqrt(M)
    gamma = 1e-8

    objective = cp.Minimize(
        0.5 * (cp.norm(R @ O_root, 'fro') ** 2) - cp.trace(R) + lam * (gamma * cp.pnorm(S, 1) + cp.norm(L_cvx, "nuc")))
    constraints = [R == S - L_cvx, L_cvx >> 0]

    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=False)
    opt_error = prob.value

    # extract dependencies
    J_hat = S.value

    deps_hat = get_deps_from_inverse_sig(J_hat, thresh=thresh)
    deps = [(i, j) for i, j in deps_hat if i < j]
    return deps
