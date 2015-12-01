#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration object for pycmt3d
"""

try:
    import numpy as np
except:
    msg = ("No module named numpy. "
           "Please install numpy first, it is needed before using pycmt3d.")
    raise ImportError(msg)
from __init__ import logger
from user_defined_weighting_function import default_weight_function
import const


class Config(object):
    """
    Configuration for source inversion

    :param npar: number of parameters to be inverted
    :param dlocation: location perturbation when calculated perturbed
    synthetic data
    :param ddepth: depth perturbation
    :param dmoment: moment perturbation
    :param weight_data: bool value of weighting data
    :param weight_function: weighting function
    :param normalize_window: add window energy into the weighting term
    :param norm_mode: two modes: 1) "data_and_synt" 2) "data_only"
    :param station_correction: bool value of whether applies station correction
    :param zero_trace: bool value of whether applies zero-trace constraint
    :param double_couple: bool value of whether applied double-couple
    constraint
    :param lamda_damping: damping coefficient
    :param bootstrap: bool value of whether applied bootstrap method
    :param bootstrap_repeat: bootstrap iterations
    """

    def __init__(self, npar, dlocation=0.0, ddepth=0.0, dmoment=0.0,
                 weight_data=True, weight_function=None,
                 weight_azi_mode="num_files",
                 normalize_window=True, norm_mode="data_only",
                 normalize_category=False,
                 station_correction=True, zero_trace=True,
                 double_couple=False, lamda_damping=0.0,
                 bootstrap=True, bootstrap_repeat=300):

        self.npar = npar
        if self.npar not in [6, 7, 9, 10, 11]:
            print ('Error: the current npar (number of parameters) is: %d '
                   % self.npar)
            print ('The npar (number of parameters) must be 6, 7, 9, 10 or 11')
            print ('When npar is 6: moment tensor')
            print ('When npar is 7: moment tensor + depth')
            print ('When npar is 9: moment tensor + depth + location'
                   '(e.g. latitude and longitude)')
            print ('When npar is 10: moment tensor + depth + location + time')
            print ('When npar is 11: moment tensor + depth + location + time '
                   '+ half duration')
            raise ValueError("Re-enter npar")
        self.dlocation = dlocation
        self.ddepth = ddepth
        self.dmoment = dmoment
        self.weight_data = weight_data
        if weight_function is not None:
            self.weight_function = weight_function
        else:
            self.weight_function = default_weight_function
        self.weight_azi_mode = weight_azi_mode.lower()
        self.normalize_window = normalize_window
        self.normalize_category = normalize_category
        if norm_mode.lower() not in ['data_and_synt', 'data_only',
                                     'data_average_only', 'data_abs_only']:
            raise ValueError("Weight mode incorrect: 1) data_and_synt; 2) "
                             "data_only;3) data_avergage_only; "
                             "4) data_abs_only")
        self.norm_mode = norm_mode.lower()
        self.station_correction = station_correction
        self.zero_trace = zero_trace
        self.double_couple = double_couple
        self.lamda_damping = lamda_damping

        self.par_name = const.PAR_LIST
        self.scale_par = np.array(
            [const.SCALE_MOMENT, const.SCALE_MOMENT, const.SCALE_MOMENT,
             const.SCALE_MOMENT, const.SCALE_MOMENT, const.SCALE_MOMENT,
             const.SCALE_DEPTH, const.SCALE_LONGITUDE, const.SCALE_LATTITUDE,
             const.SCALE_CTIME, const.SCALE_HDUR])
        self.dcmt_par = np.array(
            [self.dmoment, self.dmoment, self.dmoment, self.dmoment,
             self.dmoment, self.dmoment, self.ddepth, self.dlocation,
             self.dlocation, 1.0, 1.0])/self.scale_par
        self.bootstrap = bootstrap
        self.bootstrap_repeat = bootstrap_repeat

        self.print_summary()

    def print_summary(self):
        """
        Print function of configuration

        :return:
        """
        npar = self.npar
        logger.info("="*10 + "  Config Summary  " + "="*10)
        logger.info("Number of Inversion Par: %d" % npar)
        logger.info("   Par: [%s]" % (', '.join(self.par_name[0:npar])))
        logger.info("   Delta: [%s]" % (
            ', '.join(map(str, self.dcmt_par[0:npar]*self.scale_par[0:npar]))))

        logger.info("Weighting scheme")
        if self.weight_data:
            if self.weight_function == default_weight_function:
                logger.info("   Weighting data ===> "
                            "Using Default weighting function")
            else:
                logger.info("   Weighting data ===> "
                            "Using user-defined weighting function")
        else:
            logger.info("   No weighting applied")
        logger.info("Inversion Scheme")
        if self.double_couple:
            logger.info("   invert for double-couple source ===> "
                        "Non-linear Inversion")
        elif self.zero_trace:
            logger.info("   invert for zero-trace source ===> "
                        "Linear Inversion")
        else:
            logger.info("   No constraints applied ===> Linear Inversion ")
        logger.info("   inversion dampling lambda: %f" % self.lamda_damping)
