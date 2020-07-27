#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Config classes for weighting and inversion

:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU Lesser General Public License, version 3 (LGPLv3)
    (http://www.gnu.org/licenses/lgpl-3.0.en.html)
"""
from __future__ import print_function, division, absolute_import
import numpy as np
from .util import _float_array_to_str
from .constant import DEFAULT_SCALE_VECTOR, NM, NML, PARLIST


class Config(object):
    """
    Configuration for source inversion

    """
    def __init__(self, npar,
                 dlatitude_in_deg=None,
                 dlongitude_in_deg=None,
                 ddepth_in_m=None,
                 dmoment_tensor=None,
                 scale_vector=None,
                 zero_trace=True, double_couple=False,
                 envelope_coef=0.0,  max_nl_iter=60,
                 damping=0.0, station_correction=True,
                 weight_data=True, weight_config=None,
                 bootstrap=True, bootstrap_repeat=300,
                 taper_type="tukey"):
        """
        :param npar: number of parameters to be inverted
        :param dlatitude_in_deg: latitude perturbation when calculated
            perturbed synthetic data, unit is degree
        :param dlongtidue_in_deg: latitude perturbation when calculated
            perturbed synthetic data, unit is degree
        :param ddepth_in_m: depth perturbation, unit is meter.
        :param dmoment_tensor: perturbation of each component of
            the moment tensor, unit is dyne * cm. Here we assume for each
            component, the perturbation is the same.
        :param scale_vector: the scaling vector for d***. If none, then
            it will use the default
        :param zero_trace: bool value of whether applies zero-trace constraint
        :param double_couple: bool value of whether applied double-couple
            constraint
        :param envelope_coef: the coefficient of envelope misfit function,
            should be within [0, 1]
        :param max_nl_iter: max number of non-linear iterations
        :param damping: damping coefficient
        :param station_correction: bool value of whether applies station
            correction
        :param weight_data: bool value of weighting data
        :param weight_config: the weighting configuration
        :param bootstrap: bool value of whether applied bootstrap method
        :param bootstrap_repeat: bootstrap iterations
        :param taper_type: the taper type used for taper the seismograms
            in the windows
        """

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

        self.dlatitude = dlatitude_in_deg
        self.dlongitude = dlongitude_in_deg
        self.ddepth_in_m = ddepth_in_m
        self.dmoment = dmoment_tensor
        self._check_perturbation_sanity()

        self.weight_data = weight_data
        self.weight_config = weight_config

        self.station_correction = station_correction
        self.zero_trace = zero_trace
        self.double_couple = double_couple

        if envelope_coef < 0.0 or envelope_coef > 1.0:
            raise ValueError("Envelope coefficient must be within [0, 1]")
        self.envelope_coef = envelope_coef

        if max_nl_iter <= 0:
            raise ValueError("max_nl_iter(%d) must be larger than 0"
                             % max_nl_iter)
        self.max_nl_iter = max_nl_iter
        self.damping = damping

        # scaling term
        if scale_vector is None:
            self.scale_vector = DEFAULT_SCALE_VECTOR[:npar]
        elif len(scale_vector) != npar:
            raise ValueError("Length of scale_vector(%d) must be %d"
                             % (len(scale_vector), npar))
        else:
            self.scale_vector = scale_vector

        # original cmt perturbation
        self.dcmt_par = np.array(
            [self.dmoment, self.dmoment, self.dmoment, self.dmoment,
             self.dmoment, self.dmoment, self.ddepth_in_m, self.dlongitude,
             self.dlatitude, 1.0, 1.0])[:npar]
        # scaled cmt perturbation
        self.dcmt_par_scaled = self.dcmt_par / self.scale_vector

        self.bootstrap = bootstrap
        self.bootstrap_repeat = bootstrap_repeat

        self.taper_type = taper_type

    def _check_perturbation_sanity(self):
        """
        Check cmt perturbation is set according to npar
        """
        if self.npar >= NM:
            if self.dmoment is None:
                raise ValueError("npar(%d) requires dmoment_tensor(%s) "
                                 "to be set" % (self.npar, self.dmoment))
        if self.npar >= (NM + 1):
            if self.ddepth_in_m is None:
                raise ValueError("npar(%d) requires ddepth_in_m(%s) "
                                 "to be set" % (self.npar, self.ddepth_in_m))

        if self.npar >= NML:
            if self.dlatitude is None or self.dlongitude is None:
                raise ValueError("npar(%d) requires dlongitude_in_deg and"
                                 "dlongitude in deg to be set" % (self.npar))

    def __repr__(self):
        npar = self.npar
        string = "="*10 + "  Config Summary  " + "="*10 + "\n"

        string += "Number of inversion params: %d\n" % npar
        string += "Deriv params: [%s]\n" % ",".join(self.parlist)
        string += \
            "CMT perturbation: %s\n" % _float_array_to_str(self.dcmt_par)
        string += \
            "CMT scaling term: %s\n" % _float_array_to_str(self.scale_vector)

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
