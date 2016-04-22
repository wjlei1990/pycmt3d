#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
General util functions
"""
import numpy as np
import math
from scipy import signal


def get_window_idx(win_time, dt):
    """
    Get window index from window time and dt
    """
    istart = int(win_time[0] / dt)
    iend = int(win_time[1] / dt)
    if istart < 0:
        raise ValueError("Start index(%d) smaller than 0")
    if istart >= iend:
        raise ValueError("Start index(%d) larger or equal than end index(%d)"
                         % (istart, iend))
    return np.array([istart, iend])


def check_trace_consistent(tr1, tr2, mode="part"):
    """
    Check if two traces are consistent with each other.
    If mode is 'part', only starttime and dt is compared
    If mode is 'full', npts is also compared
    """
    _options = ["part", "full"]
    if mode not in _options:
        raise ValueError("mode(%s) must be within %s" % (mode, _options))

    if not np.isclose(tr1.stats.delta, tr2.stats.delta):
        raise ValueError("DT of two traces are not the same: %f, %f"
                         % (tr1.stats.delta, tr2.stats.delta))

    if not np.isclose(tr1.stats.starttime - tr2.stats.starttime, 0):
        raise ValueError("Starttime of two traces not the same: %s, %s"
                         % (tr1.stats.starttime, tr2.stats.starttime))

    if mode == "full":
        if tr1.stats.npts != tr2.stats.npts:
            raise ValueError("NPTS not the same: %d, %d" % (tr1.stats.npts,
                                                            tr2.stats.npts))
    else:
        return


def sum_matrix(data, coef=None):
    """
    Sum a list of matrix with same dimension(sum over first index)
    :return: \sum coef[i] * data[i]
    """
    if coef is None:
        coef = np.ones(len(data))
    elif len(coef) != len(data):
        raise ValueError("dimension of coef and data not the same")

    sum_value = coef[0] * data[0]
    for _idx in range(1, len(coef)):
        sum_value += coef[_idx] * data[_idx]
    return sum_value


def random_select(nsamples, nselected=1, replace=True):
    """
    Draw nselected number of samples from nsamples,
    index [0, nsamples-1]
    :param nsamples: the total number of samples
    :type nsamples: int
    :param nselected: the number of ssamples drawed
    :type nselected: int
    :return: select position array. If selected twice, then on the same
        index, value would be 2.
    """
    subset_array = np.zeros(nsamples)
    location_array = np.random.choice(nsamples, nselected,
                                      replace=replace)
    for _idx in location_array:
        subset_array[_idx] += 1
    return subset_array


def _float_to_str(value):
    """
    Convert float value to a specific precision string

    :param value:
    :return: string of the value
    """
    return "%.5f" % value


def _float_array_to_str(array):
    """
    Convert float array to string

    :return:
    """
    string = "[  "
    for ele in array:
        string += "%.3e," % ele
    string += "]"
    return string


def tukey_window(window_length, alpha=0.2):
    """
    The Tukey window, also known as the tapered cosine window,
    can be regarded as a cosine lobe of width \alpha * N / 2
    that is convolved with a rectangle window of width (1 - \alpha / 2).
    At \alpha = 1 it becomes rectangular, and
    at \alpha = 0 it becomes a Hann window.
    We use the same reference as MATLAB to provide the same results
    in case users compare a MATLAB output to this function output
    ---------
    Reference
    ---------
    http://www.mathworks.com/access/helpdesk/help/toolbox/signal/tukeywin.html
    """
    # Special cases
    if alpha <= 0:
        return np.ones(window_length)  # rectangular window
    elif alpha >= 1:
        return np.hanning(window_length)

    # Normal case
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)

    # first condition 0 <= x < alpha/2
    first_condition = x < alpha/2
    w[first_condition] = \
        0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2)))

    # second condition already taken care of

    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x >= (1 - alpha/2)
    w[third_condition] = \
        0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2)))

    return w


def construct_taper(npts, taper_type="tukey", alpha=0.2):
    taper_type = taper_type.lower()
    _options = ['hann', 'boxcar', 'tukey']
    if taper_type not in _options:
        raise ValueError("taper type option: %s" % taper_type)
    if taper_type == "hann":
        taper = signal.hann(npts)
    elif taper_type == "boxcar":
        taper = signal.boxcar(npts)
    elif taper_type == "tukey":
        taper = signal.tukey(npts, alpha=alpha)
    else:
        raise ValueError("Taper type not supported: %s" % taper_type)
    return taper
