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


def print_cmtsource_summary(cmt):
    """
    Print CMTSolution source summary

    :return:
    """
    logger.info("=" * 10 + "  Event Summary  " + "=" * 10)
    logger.info("Event name: %s" % cmt.eventname)
    logger.info("   Latitude and longitude: %.2f, %.2f" % (
        cmt.latitude, cmt.longitude))
    logger.info("   Depth: %.1f km" % (cmt.depth_in_m / 1000.0))
    logger.info("   Region tag: %s" % cmt.region_tag)
    logger.info("   Trace: %.3e" % (
        (cmt.m_rr + cmt.m_tt + cmt.m_pp) / cmt.M0))
    logger.info("   Moment Magnitude: %.2f" % cmt.moment_magnitude)


def _write_log_file_(filename, nshift_list, cc_list, dlnA_list):
    with open(filename, 'w') as f:
        for i in range(len(nshift_list)):
            nshift = nshift_list[i]
            cc = cc_list[i]
            dlnA = dlnA_list[i]
            f.write("%5d %10.3f %10.3f\n" % (nshift, cc, dlnA))


def print_inversion_summary(self):
    """
    Print out the inversion summary

    :return:
    """
    logger.info("*" * 20)
    logger.info("Invert cmt parameters(%d par)" % self.config.npar)

    logger.info("Old CMT par: [%s]" % (
        ', '.join(map(str, self.cmt_par))))
    logger.info("dm: [%s]" % (
        ', '.join(map(str, self.new_cmt_par - self.cmt_par))))
    logger.info("New CMT par: [%s]" % (
        ', '.join(map(str, self.new_cmt_par))))

    logger.info("Trace: %e" % (np.sum(self.new_cmt_par[0:3])))
    logger.info("Energy change(scalar moment): %5.2f%%" % (
        (self.new_cmtsource.M0 - self.cmtsource.M0) /
        self.cmtsource.M0 * 100.0))

    self.inversion_result_table()


def inversion_result_table(self):
    """
    Print out the inversion table

    :return:
    """
    title = "*" * 20 + " Inversion Result Table(%d npar) " % \
        self.config.npar + "*" * 20
    logger.info(title)

    if not self.config.bootstrap:
        logger.info("PAR         Old_CMT        New_CMT")
        logger.info("Mrr:  %15.6e  %15.6e" % (
            self.cmtsource.m_rr, self.new_cmtsource.m_rr))
        logger.info("Mtt:  %15.6e  %15.6e" % (
            self.cmtsource.m_tt, self.new_cmtsource.m_tt))
        logger.info("Mpp:  %15.6e  %15.6e" % (
            self.cmtsource.m_pp, self.new_cmtsource.m_pp))
        logger.info("Mrt:  %15.6e  %15.6e" % (
            self.cmtsource.m_rt, self.new_cmtsource.m_rt))
        logger.info("Mrp:  %15.6e  %15.6e" % (
            self.cmtsource.m_rp, self.new_cmtsource.m_rp))
        logger.info("Mtp:  %15.6e  %15.6e" % (
            self.cmtsource.m_tp, self.new_cmtsource.m_tp))
        logger.info(
            "dep:  %15.3f  %15.3f" % (
                self.cmtsource.depth_in_m / 1000.0,
                self.new_cmtsource.depth_in_m / 1000.0))
        logger.info("lon:  %15.3f  %15.3f" % (
            self.cmtsource.longitude, self.new_cmtsource.longitude))
        logger.info("lat:  %15.3f  %15.3f" % (
            self.cmtsource.latitude, self.new_cmtsource.latitude))
        logger.info("ctm:  %15.3f  %15.3f" % (
            self.cmtsource.time_shift, self.new_cmtsource.time_shift))
        logger.info("hdr:  %15.3f  %15.3f" % (
            self.cmtsource.half_duration,
            self.new_cmtsource.half_duration))
    else:
        logger.info("PAR         Old_CMT          New_CMT     "
                    "Bootstrap_Mean     Bootstrap_STD     STD/Mean")
        logger.info(
        "Mrr:  %15.6e  %15.6e  %15.6e  %15.6e   %10.2f%%" % (
            self.cmtsource.m_rr, self.new_cmtsource.m_rr,
            self.par_mean[0], self.par_std[0],
            self.std_over_mean[0] * 100))
        logger.info(
            "Mtt:  %15.6e  %15.6e  %15.6e  %15.6e   %10.2f%%" % (
                self.cmtsource.m_tt, self.new_cmtsource.m_tt,
                self.par_mean[1], self.par_std[1],
                self.std_over_mean[1] * 100))
        logger.info(
            "Mpp:  %15.6e  %15.6e  %15.6e  %15.6e   %10.2f%%" % (
                self.cmtsource.m_pp, self.new_cmtsource.m_pp,
                self.par_mean[2], self.par_std[2],
                self.std_over_mean[2] * 100))
        logger.info(
            "Mrt:  %15.6e  %15.6e  %15.6e  %15.6e   %10.2f%%" % (
                self.cmtsource.m_rt, self.new_cmtsource.m_rt,
                self.par_mean[3], self.par_std[3],
                self.std_over_mean[3] * 100))
        logger.info(
            "Mrp:  %15.6e  %15.6e  %15.6e  %15.6e   %10.2f%%" % (
                self.cmtsource.m_rp, self.new_cmtsource.m_rp,
                self.par_mean[4], self.par_std[4],
                self.std_over_mean[4] * 100))
        logger.info(
            "Mtp:  %15.6e  %15.6e  %15.6e  %15.6e   %10.2f%%" % (
                self.cmtsource.m_tp, self.new_cmtsource.m_tp,
                self.par_mean[5], self.par_std[5],
                self.std_over_mean[5] * 100))
        logger.info("dep:  %15.3f  %15.3f  %15.3f  %15.3f   %10.2f%%" % (
            self.cmtsource.depth_in_m / 1000.0,
            self.new_cmtsource.depth_in_m / 1000.0,
            self.par_mean[6], self.par_std[6],
            self.std_over_mean[6] * 100))
        logger.info("lon:  %15.3f  %15.3f  %15.3f  %15.3f   %10.2f%%" % (
            self.cmtsource.longitude, self.new_cmtsource.longitude,
            self.par_mean[7], self.par_std[7],
            self.std_over_mean[7] * 100))
        logger.info("lat:  %15.3f  %15.3f  %15.3f  %15.3f   %10.2f%%" % (
            self.cmtsource.latitude, self.new_cmtsource.latitude,
            self.par_mean[8], self.par_std[8],
            self.std_over_mean[8] * 100))
        logger.info("ctm:  %15.3f  %15.3f  %15.3f  %15.3f   %10.2f%%" % (
            self.cmtsource.time_shift, self.new_cmtsource.time_shift,
            self.par_mean[9], self.par_std[9],
            self.std_over_mean[9] * 100))
        logger.info("hdr:  %15.3f  %15.3f  %15.3f  %15.3f   %10.2f%%" % (
            self.cmtsource.half_duration, self.new_cmtsource.half_duration,
            self.par_mean[10], self.par_std[10],
        self.std_over_mean[10] * 100))
