"""
Linear and non-linear solver for source inversion
"""
from __future__ import print_function, division, absolute_import
import numpy as np
from .measure import get_f_df


def linear_solver(old_par, A, b, npar, zero_trace=True):
    """
    if invert for moment tensor with zero-trace constraints
    or no constraint
    """
    if zero_trace:
        na = npar + 1
    else:
        na = npar

    AA = np.zeros([na, na])
    bb = np.zeros(na)
    AA[0:npar, 0:npar] = A
    bb[0:npar] = b

    if zero_trace:
        bb[na - 1] = - np.sum(old_par[0:3])
        AA[0:6, na - 1] = np.array([1, 1, 1, 0, 0, 0])
        AA[na - 1, 0:6] = np.array([1, 1, 1, 0, 0, 0])
        AA[na - 1, na - 1] = 0.0

    try:
        dm = np.linalg.solve(AA, bb)
    except Exception as err:
        raise ValueError("Can not solve the linear equation due to:%s"
                         % err)
    new_par = old_par[0:npar] + dm[0:npar]
    print("old par")
    print(old_par)
    print("dm")
    print(dm)
    return new_par


def nonlinear_solver(old_par, A, b, npar):
    """
    if invert for moment tensor with double couple constraints
    setup starting solution, solve directly for moment instead
    of dm, exact implementation of (A16)
    logger.info('Non-linear Inversion')

    :return:
    """
    na = npar + 2
    mstart = np.copy(old_par)
    m1 = np.copy(mstart)
    lam = np.zeros(2)
    AA = np.zeros([na, na])
    bb = np.zeros(na)

    error = np.zeros([NMAX_NL_ITER, na])
    for iter_idx in range(NMAX_NL_ITER):
        get_f_df(npar, A, b, m1, lam, mstart, AA, bb)
        bb = - bb
        xout = np.linalg.solve(AA, bb)
        m1 = m1 + xout[0:npar]
        lam = lam + xout[npar:na]
        error[iter_idx, :] = np.dot(AA, xout) - bb
    # dm = m1 - mstart
    return m1
