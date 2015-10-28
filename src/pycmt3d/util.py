#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import math


def sum_matrix(coef, data):

    if len(coef) != len(data):
        raise ValueError("dimension of coef and data not the same")
    sum_value = coef[0] * data[0]
    for _idx in range(1, len(coef)):
        sum_value += coef[_idx] * data[_idx]
    return sum_value


def gen_random_array(npts, sample_number=0):

    if npts <= 0:
        return
    if sample_number <= 1:
        sample_number = 1
    subset_array = np.zeros(npts)
    location_array = np.random.choice(npts, sample_number)
    for _idx in location_array:
        subset_array[_idx] += 1
    return subset_array


def hanning_window(npts):
    """
    Hanning taper constructor

    :param npts: number of points
    :return:
    """
    taper = np.zeros(npts)
    for i in range(npts):
        taper[i] = 0.5 * (1 - math.cos(2 * np.pi * (float(i) / (npts - 1))))
    return taper


def boxcar_window(npts):
    """
    Boxcar taper constructor
    """
    taper = np.ones(npts)
    return taper


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


def construct_taper(npts, taper_type="tukey"):
    taper_type = taper_type.lower()
    if taper_type not in ['hanning', 'boxcar', 'tukey']:
        raise ValueError("taper type option: 1) boxcar; 2) hanning; 3) tukey")
    if taper_type == "hanning":
        taper = hanning_window(npts)
    elif taper_type == "boxcar":
        taper = boxcar_window(npts)
    elif taper_type == "tukey":
        taper = tukey_window(npts, alpha=0.1)
    else:
        raise ValueError("Taper type not supported: %s" % taper_type)
    return taper
