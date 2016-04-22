#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Measurement util functions
"""
import numpy as np
from .util import construct_taper, check_trace_consistent, get_window_idx
import constant


def _xcorr_win_(arr1, arr2):
    """
    cross-correlation between arr1 and arr2 to get the max_cc_value
    and nshift
    """
    if len(arr1) != len(arr2):
        raise ValueError("Length of arr1(%d) and arr2(%d) must be the same"
                         % (len(arr1), len(arr2)))
    cc = np.correlate(arr1, arr2, mode="full")
    nshift = cc.argmax() - len(arr1) + 1
    # Normalized cross correlation.
    max_cc_value = \
        cc.max() / np.sqrt((arr1 ** 2).sum() * (arr2 ** 2).sum())
    return max_cc_value, nshift


def _dlnA_win_(arr1, arr2):
    """
    Power ration of arr1 over arr2, unit in dB.
    """
    if len(arr1) != len(arr2):
        raise ValueError("Length of arr1(%d) and arr2(%d) not the same"
                         % (len(arr1), len(arr2)))
    return 10 * np.log10(np.sum(arr1 ** 2) / np.sum(arr2 ** 2))


def _cc_amp_(arr1, arr2):
    """
    Cross-correlation amplitude ratio
    """
    return 10 * np.log10(np.sum(arr1 * arr2) / np.sum(arr2 ** 2))


def _energy_(arr, taper=None):
    """
    Energy of an array
    """
    if taper is None:
        return np.sum(arr ** 2)
    else:
        return np.sum((taper * arr) ** 2)


def _diff_energy_(arr1, arr2, taper=None):
    """
    energy of two array difference
    """
    if taper is None:
        return np.sum((arr1 - arr2) ** 2)
    else:
        return np.sum((taper * (arr1 - arr2)) ** 2)


def correct_window_index(arr1, arr2, istart, iend):
    """
    Correct the window index based on cross-correlation shift
    """
    npts = min(len(arr1), len(arr2))
    max_cc, nshift = _xcorr_win_(arr1[istart:iend], arr2[istart:iend])
    istart_d = max(1, istart + nshift)
    iend_d = min(npts, iend + nshift)
    istart_s = max(1, istart_d - nshift)
    iend_s = min(npts, iend_d - nshift)
    if (iend_d - istart_d) != (iend_s - istart_s):
        raise ValueError("After correction, window length not the same: "
                         "[%d, %d] and [%d, %d]" % (istart_d, iend_d,
                                                    istart_s, iend_s))
    return istart_d, iend_d, istart_s, iend_s, max_cc, nshift


def measure_window(obsd_array, synt_array, istart, iend,
                   station_correction=True):
    """
    Make measurements on windows. If station_correction, correct window
    idx first based on cross-correlation time shift measurement.

    :param obsd: obsd data array
    :param synt: synt data array
    :param istart: start index of original window
    :param iend: end index of original window
    :param station_correction: station correction flag
    :return:
    """
    # correct window index if required
    if station_correction:
        istart_d, iend_d, istart_s, iend_s, max_cc, nshift = \
            correct_window_index(obsd_array, synt_array, istart, iend)
    else:
        max_cc, nshift = _xcorr_win_(obsd_array[istart:iend],
                                     synt_array[istart:iend])
        istart_d = istart
        iend_d = iend
        istart_s = istart
        iend_s = iend

    # make measurements
    _obs = obsd_array[istart_d: iend_d]
    _syn = synt_array[istart_s: iend_s]
    dlnA = _dlnA_win_(_obs, _syn)
    cc_amp_ratio = _cc_amp_(_obs, _syn)
    return nshift, max_cc, dlnA, cc_amp_ratio


def get_f_df(npar, A, b, m, lam, mstart, fij, f0):
    """
    Iterative solver for Non-linear case(double-couple constraint)

    :param A: basic Hessian matrix
    :param b: basic misfit vector
    :param m: current source array
    :param lam: constraints coefficient for zero-trace and
    double-couple constraints
    :param mstart: starting source solution
    :param fij: reconstructed Hessian Matrix AA
    :param f0: reconstructed misfit vector bb
    :return:
    """

    NM = constant.NM

    # U_j
    dc1_dm = np.array([1, 1, 1, 0, 0, 0])

    # V_j
    dc2_dm = np.zeros(6)
    dc2_dm[0] = m[1] * m[2] - m[5] ** 2
    dc2_dm[1] = m[0] * m[2] - m[4] ** 2
    dc2_dm[2] = m[0] * m[1] - m[3] ** 2
    dc2_dm[3] = 2 * m[4] * m[5] - 2 * m[2] * m[3]
    dc2_dm[4] = 2 * m[3] * m[5] - 2 * m[1] * m[4]
    dc2_dm[5] = 2 * m[3] * m[4] - 2 * m[0] * m[5]

    # f(x^i) = H_jk (m_k^i -m_k^0) - b_j + lam_1 * U_j + lam_2 * V_j (A11)
    f0.fill(0.)
    f0[0:npar] = np.dot(A[0:npar, 0:npar], m[0:npar] -
                        mstart[0:npar]) - b[0:npar]
    # print "f0 step1:", f0
    f0[0:constant.NM] += \
        lam[0] * dc1_dm[0:constant.NM] + lam[1] * dc2_dm[0:constant.NM]
    # f_(n+1) and f_(n+2)
    f0[npar] = m[0] + m[1] + m[2]
    moment_tensor = np.array([[m[0], m[3], m[4]],
                              [m[3], m[1], m[5]], [m[4], m[5], m[2]]])
    f0[npar + 1] = np.linalg.det(moment_tensor)
    f0[npar + 1] = m[0] * (m[1] * m[2] - m[5] ** 2) \
        - m[3] * (m[3] * m[2] - m[5] * m[4]) \
        + m[4] * (m[3] * m[5] - m[4] * m[1])

    # Y_jk
    dc2_dmi_dmj = np.zeros([6, 6])
    dc2_dmi_dmj[0, :] = np.array([0.0, m[2], m[1], 0.0, 0.0, -2.0 * m[5]])
    dc2_dmi_dmj[1, :] = np.array([m[2], 0.0, m[0], 0.0, -2.0 * m[4], 0.0])
    dc2_dmi_dmj[2, :] = np.array([m[1], m[0], 0.0, -2.0 * m[3], 0.0, 0.0])
    dc2_dmi_dmj[3, :] = np.array([0.0, 0.0, -2.0 * m[3], -2.0 * m[2],
                                  2 * m[5], 2 * m[4]])
    dc2_dmi_dmj[4, :] = np.array([0.0, -2.0 * m[4], 0.0, 2.0 * m[5],
                                  -2.0 * m[1], 2 * m[3]])
    dc2_dmi_dmj[5, :] = np.array([-2.0 * m[5], 0.0, 0.0, 2.0 * m[4],
                                  2.0 * m[3], -2.0 * m[0]])

    # ! f_jk = H_jk + lam_2 * Y_jk
    fij.fill(0)
    fij[0:npar, 0:npar] = A[0:npar, 0:npar]
    fij[0:NM, 0:NM] = fij[0:NM, 0:NM] + lam[1] * dc2_dmi_dmj[0:NM, 0:NM]
    fij[0:NM, npar] = dc1_dm
    fij[0:NM, npar + 1] = dc2_dm
    fij[npar, 0:NM] = dc1_dm
    fij[npar + 1, 0:NM] = dc2_dm


def calculate_dsyn(datalist, win_idx, parlist, dcmt_par, taper=None):
    """
    Calculate dsyn matrix based on perturbed seismograms. Only synt
    and perturbed synthetic are used here:
    dsyn = synt_perturbation / perturbation.

    :param datalist:
    :return:
    """
    istart = win_idx[0]
    iend = win_idx[1]
    win_len = iend - istart

    dsyn = np.zeros((len(parlist), win_len))
    for itype, type_name in enumerate(parlist):
        if itype < constant.NM:
            # moment tensor, linear term
            dsyn[itype, :] = \
                datalist[type_name].data[istart:iend] / dcmt_par[itype]
        elif itype < constant.NML:
            # location, finite difference to get dsyn
            dsyn[itype, :] = \
                (datalist[type_name].data[istart:iend] -
                 datalist['synt'].data[istart:iend]) / dcmt_par[itype]
        elif itype == constant.NML:
            # time shift
            dt = datalist['synt'].stats.delta
            dsyn[itype, 0:(win_len - 1)] = \
                np.diff(datalist['synt'].data) / (dt * dcmt_par[itype])
            dsyn[itype, -1] = dsyn[itype, -2]
        elif itype == constant.NML + 1:
            raise NotImplementedError("Not implemented with npar == %d"
                                      % itype)
        else:
            raise ValueError("npar(%d) error" % itype)
        # elif itype == constant.NML + 1:  # half duration
        #    dsyn[itype, 0:npts - 1] = -0.5 * cmt_par[itype] * (
        #        dsyn[constant.NML, 1:npts] -
        #        dsyn[constant.NML, 0:npts - 1]) / dt
        #    dsyn[itype, npts - 1] = dsyn[itype, npts - 2]
    if taper is not None:
        dsyn = taper * dsyn
    return dsyn


def compute_A_b(datalist, win_time, parlist, dcmt_par):
    """
    Calculate the matrix A and vector b based on one pair of
    observed data and synthetic data on a given window.

    :param window: data and window information
    :type window: :class:`pycmt3d.Window`
    :param win_idx: window index(a specific window)
    :type win_idx: integer
    :param dsyn: derivative synthetic data matrix
    :type dsyn: numpy.array
    :return:
    """
    obsd = datalist['obsd']
    synt = datalist['synt']
    check_trace_consistent(obsd, synt)
    dt = obsd.stats.delta
    win_idx = get_window_idx(win_time, obsd.stats.delta)

    win_len = win_idx[1] - win_idx[0]
    taper = construct_taper(win_len, taper_type=constant.taper_type)

    dsyn = calculate_dsyn(datalist, win_idx, parlist, dcmt_par,
                          taper=taper)

    # station correction
    istart_d, iend_d, istart_s, iend_s, _, _ = \
        correct_window_index(obsd.data, synt.data, win_idx[0], win_idx[1])


    npar = len(parlist)
    A1 = np.dot(dsyn, dsyn.transpose()) * dt
    b1 = np.sum(
        taper * (obsd.data[istart_d:iend_d] -
                 synt.data[istart_s:iend_s]) *
        dsyn, axis=1)

    return A1, b1


def calculate_variance_on_trace(obsd, synt, win_time):
    """
    Calculate the variance reduction on a pair of obsd and
    synt and windows

    :param obsd: observed data trace
    :type obsd: :class:`obspy.core.trace.Trace`
    :param synt: synthetic data trace
    :type synt: :class:`obspy.core.trace.Trace`
    :param win_time: [win_start, win_end]
    :type win_time: :class:`list` or :class:`numpy.array`
    :return:  waveform misfit reduction and observed data
    energy [v1, d1]
    :rtype: [float, float]
    """
    dt = synt.stats.delta
    win_time = np.array(win_time)
    num_wins = win_time.shape[0]

    v1_array = np.zeros(num_wins)
    d1_array = np.zeros(num_wins)
    tshift_array = np.zeros(num_wins)
    cc_array = np.zeros(num_wins)
    dlnA_array = np.zeros(num_wins)
    cc_amp_array = np.zeros(num_wins)

    for _win_idx in range(win_time.shape[0]):
        istart, iend = get_window_idx(win_time[_win_idx], obsd.stats.delta)

        istart_d, iend_d, istart_s, iend_s, _, _ = \
            correct_window_index(obsd.data, synt.data, istart, iend)

        nshift, cc, dlnA, cc_amp_value = \
            measure_window(obsd, synt, istart, iend)

        taper = construct_taper(iend - istart, taper_type=constant.taper_type)

        v1_array[_win_idx] = dt *_diff_energy_(obsd.data[istart_d:iend_d],
                                               synt.data[istart_s:iend_s],
                                               taper=taper)
        d1_array[_win_idx] = dt *_energy_(obsd.data[istart_d:iend_d],
                                          taper=taper)

        tshift_array[_win_idx] = nshift * dt
        cc_array[_win_idx] = cc
        dlnA_array[_win_idx] = dlnA
        cc_amp_array[_win_idx] = cc_amp_value

    return v1_array, d1_array, tshift_array, cc_array, dlnA_array, \
        cc_amp_array
