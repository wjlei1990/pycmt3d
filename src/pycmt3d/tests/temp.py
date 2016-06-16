#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pycmt3d test suite.

Run with pytest.

:copyright:
    Wenjie Lei (lei@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import print_function, division
import inspect
import os
from pycmt3d import CMTSource
from pycmt3d import DataContainer
from pycmt3d import Config, WeightConfig
from pycmt3d.constant import PARLIST
from pycmt3d import Cmt3D


# Most generic way to get the data folder path.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe()))), "data")
OBSD_DIR = os.path.join(DATA_DIR, "data_T006_T030")
SYNT_DIR = os.path.join(DATA_DIR, "syn_T006_T030")
CMTFILE = os.path.join(DATA_DIR, "CMTSOLUTION")

cmtsource = CMTSource.from_CMTSOLUTION_file(CMTFILE)


def construct_dcon_two():
    """
    Data Container with two stations
    """
    dcon = DataContainer(parlist=PARLIST[:9])
    os.chdir(DATA_DIR)
    # window_file = os.path.join(DATA_DIR,
    #                           "flexwin_T006_T030.output.two_stations.origin")
    window_file = os.path.join(DATA_DIR,
                               "flexwin_T006_T030.output")
    dcon.add_measurements_from_sac(window_file, tag="T006_T030")
    return dcon


def weighting_two():
    dcon_two = construct_dcon_two()

    # weight_config = DefaultWeightConfig(
    #    normalize_by_energy=False, normalize_by_category=False,
    #    comp_weight={"Z": 1.0, "R": 1.0, "T": 1.0},
    #    love_dist_weight=1.0, pnl_dist_weight=1.0,
    #    rayleigh_dist_weight=1.0, azi_exp_idx=0.5)

    weight_config = WeightConfig(
        normalize_by_energy=True, normalize_by_category=True)

    config = Config(6, dlocation=0.5, ddepth=0.5, dmoment=1.0e22,
                    zero_trace=True, weight_data=True,
                    envelope_coef=0.1,
                    station_correction=True,
                    weight_config=weight_config)

    srcinv = Cmt3D(cmtsource, dcon_two, config)
    srcinv.source_inversion()
    srcinv.plot_new_synt_seismograms("test")
    srcinv.plot_stats_histogram(outputdir="test")
    srcinv.plot_summary(outputdir="test", mode="regional")


if __name__ == "__main__":
    weighting_two()
