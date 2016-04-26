#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration object for pycmt3d
"""
from __future__ import print_function, division
try:
    import numpy as np
except:
    msg = ("No module named numpy. "
           "Please install numpy first, it is needed before using pycmt3d.")
    raise ImportError(msg)
from .util import _float_array_to_str
from .constant import SCALE_LONGITUDE, SCALE_LATITUDE, SCALE_DEPTH
from .constant import SCALE_MOMENT, PARLIST


class WeightConfig(object):
    """
    Base class of weight config. Shouldn't be used for most cases.
    Since we introduce complex weighting strategies here, so I think
    it might be worth to seperate WeightConfig from the Config.
    """
    def __init__(self, mode="default", normalize_by_energy=False,
                 normalize_by_category=False):
        self.mode = mode.lower()
        self.normalize_by_energy = normalize_by_energy
        self.normalize_by_category = normalize_by_category

    def __repr__(self):
        string = "Weight Strategy:\n"
        string += "mode: %s\n" % self.mode
        string += "normalize_by_energy: %s\n" % self.normalize_by_energy
        string += "normalize_by_category: %s\n" % self.normalize_by_category
        return string

    def __str__(self):
        return self.__repr__()


class DefaultWeightConfig(WeightConfig):
    """
    Weight config in original CMT3D packages
    """
    def __init__(self, normalize_by_energy=False, normalize_by_category=False,
                 comp_weight={"Z":2.0, "R": 1.0, "T": 2.0},
                 love_dist_weight=0.78, pnl_dist_weight=1.15,
                 rayleigh_dist_weight=0.55,
                 azi_exp_idx=0.5):
        WeightConfig.__init__(self, mode="default",
                              normalize_by_energy=normalize_by_energy,
                              normalize_by_category=normalize_by_category)
        self.comp_weight = comp_weight
        self.love_dist_weight = love_dist_weight
        self.pnl_dist_weight = pnl_dist_weight
        self.rayleigh_dist_weight = rayleigh_dist_weight
        self.azi_exp_idx = azi_exp_idx

    def __repr__(self):
        string = "Weight Strategy:\n"
        string += "mode: %s\n" % self.mode
        string += "normalize_by_energy: %s\n" % self.normalize_by_energy
        string += "normalize_by_category: %s\n" % self.normalize_by_category
        string += "component weight: %s\n" % self.comp_weight
        string += "pnl, rayleigh and love distance weights: %s\n" % (
            self.pnl_dist_weight, self.rayleigh_dist_weight,
            self.love_dist_weight)
        string += "azimuth exponential index: %s\n" % self.azi_exp_idx
        return string


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
    :param damping: damping coefficient
    :param bootstrap: bool value of whether applied bootstrap method
    :param bootstrap_repeat: bootstrap iterations
    """

    def __init__(self, npar, dlocation=0.0, ddepth=0.0, dmoment=0.0,
                 zero_trace=True, double_couple=False, damping=0.0,
                 station_correction=True,
                 weight_data=True, weight_config=None,
                 bootstrap=True, bootstrap_repeat=300):

        _options = [6, 7, 9, 10, 11]
        if npar not in _options:
            print('Error: the current npar (number of parameters) is: %d'
                  % self.npar)
            print('The npar (number of parameters) must be within %s'
                  % _options)
            print('When npar is 6: moment tensor')
            print('When npar is 7: moment tensor + depth')
            print('When npar is 9: moment tensor + depth + location'
                  '(e.g. latitude and longitude)')
            print('When npar is 10(not implemented yet): '
                  'moment tensor + depth + location + time')
            print('When npar is 11(not implemented yet): '
                  'moment tensor + depth + location + time + half duration')
            raise ValueError("Re-enter npar")
        if npar in [10, 11]:
            raise NotImplementedError("Not implemented with npar=%d" % npar)

        self.npar = npar
        self.parlist = PARLIST[:npar]

        self.dlocation = dlocation
        self.ddepth = ddepth
        self.dmoment = dmoment

        self.weight_data = weight_data
        self.weight_config = weight_config

        self.station_correction = station_correction
        self.zero_trace = zero_trace
        self.double_couple = double_couple
        self.damping = damping

        # scaling term
        self.scale_par = np.array(
            [SCALE_MOMENT, SCALE_MOMENT, SCALE_MOMENT,
             SCALE_MOMENT, SCALE_MOMENT, SCALE_MOMENT,
             SCALE_DEPTH, SCALE_LONGITUDE, SCALE_LATITUDE,
             1.0, 1.0])
        # original cmt perturbation
        self._dcmt_par = np.array(
            [self.dmoment, self.dmoment, self.dmoment, self.dmoment,
             self.dmoment, self.dmoment, self.ddepth, self.dlocation,
             self.dlocation, 1.0, 1.0])
        # scaled cmt perturbation
        self.dcmt_par = self._dcmt_par / self.scale_par

        self.bootstrap = bootstrap
        self.bootstrap_repeat = bootstrap_repeat

    def __repr__(self):
        npar = self.npar
        string = "="*10 + "  Config Summary  " + "="*10 + "\n"

        string += "Number of inversion params: %d\n" % npar
        string += "Deriv params: [%s]\n" % ",".join(self.parlist)
        string += \
            "CMT perturbation: %s\n" % _float_array_to_str(self._dcmt_par)
        string += \
            "CMT scaling term: %s\n" % _float_array_to_str(self.scale_par)

        string += "-" * 5 + "\nInversion Schema\n"
        string += "Zero trace: %s  Doulbe couple: %s\n" % (
            self.zero_trace, self.double_couple)
        string += "Damping:%s\n" % self.damping
        string += "Bootstrap:%s\n" % self.bootstrap
        if self.bootstrap:
            string += "Bootstrap repeat times: %d\n" % self.bootstrap_repeat

        string += "-" * 5 + "\nWeight Schema\n"
        string += "%s" % str(self.weight_config)
        return string
