#!/usr/bin/env python
# -*- coding: utf-8 -*-
from source import CMTSource
import const
from __init__ import logger
import util
from util import construct_taper
from util import _float_to_str, _float_array_to_str
from plot_util import PlotUtil, _plot_new_seismogram
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import math
from obspy.core.util.geodetics import gps2DistAzimuth
import matplotlib.gridspec as gridspec


def _plot_stats_histogram_one_(pos, cat, vtype, data_b, data_a, num_bin):
    plt.subplot(pos)
    plt.xlabel(vtype, fontsize=15)
    plt.ylabel(cat, fontsize=15)
    if vtype == "cc":
        ax_min = min(min(data_b), min(data_a))
        ax_max = max(max(data_b), max(data_a))
    elif vtype == "Kai":
        ax_min = 0.0
        ax_max = max(max(data_b), max(data_a))
    else:
        ax_min = min(min(data_b), min(data_a))
        ax_max = max(max(data_b), max(data_a))
        abs_max = max(abs(ax_min), abs(ax_max))
        ax_min = -abs_max
        ax_max = abs_max
    binwidth = (ax_max - ax_min) / num_bin
    plt.hist(
        data_b, bins=np.arange(ax_min, ax_max+binwidth/2., binwidth),
        facecolor='blue', alpha=0.3)
    plt.hist(
        data_a, bins=np.arange(ax_min, ax_max+binwidth/2., binwidth),
        facecolor='green', alpha=0.5)

def _plot_stats_histogram_per_cat_(G, irow, cat, data_before,
                                   data_after):
    num_bins = [15, 15, 15, 15, 15]
    vtype_list = ['time shift', 'cc', 'Power_Ratio(dB)',
                  'CC Amplitude Ratio(dB)', 'Kai']
    # plot order
    var_index = [0, 1, 2, 3, 4]
    for _idx, var_idx in enumerate(var_index):
        vtype = vtype_list[var_idx]
        _plot_stats_histogram_one_(
            G[irow, _idx], cat, vtype, data_before[:, var_idx],
            data_after[:, var_idx],
            num_bins[var_idx])

def _plot_stats_histogram(stats_before, stats_after, figname,
                          figure_format="png"):
    """
    Plot inversion histogram

    :param outputdir:
    :return:
    """
    nrows = len(stats_before.keys())
    ncols = stats_before[stats_before.keys()[0]].shape[1]

    plt.figure(figsize=(4*ncols, 4*nrows))
    G = gridspec.GridSpec(nrows, ncols)
    irow = 0
    for cat in stats_before.keys():
        _plot_stats_histogram_per_cat_(
            G, irow, cat, stats_before[cat], stats_after[cat])
        irow += 1
    plt.tight_layout()
    plt.savefig(figname)
