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


class WeightConfigBase(object):
    """
    Base class of weight config. Shouldn't be used.
    Since we introduce complex weighting strategies here, so I think
    it might be worth to seperate WeightConfig from the Config.
    """
    def __init__(self, mode, normalize_by_energy=False,
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


class WeightConfig(WeightConfigBase):
    def __init__(self, normalize_by_energy=False,
                 normalize_by_category=False,
                 azi_bins=12, azi_exp_idx=0.5):
        WeightConfigBase.__init__(
            self, "classic", normalize_by_energy=normalize_by_energy,
            normalize_by_category=normalize_by_category)
        self.azi_bins = azi_bins
        self.azi_exp_idx = azi_exp_idx

    def __repr__(self):
        string = "Weight Strategy:\n"
        string += "mode: %s\n" % self.mode
        string += "normalize_by_energy: %s\n" % self.normalize_by_energy
        string += "normalize_by_category: %s\n" % self.normalize_by_category
        string += "Azimuth bins and exp index: %d, %f" % (self.azi_bins,
                                                          self.azi_exp_idx)
        return string


class DefaultWeightConfig(WeightConfigBase):
    """
    Weight config in original CMT3D packages
    """
    def __init__(self,
                 normalize_by_energy=False,
                 normalize_by_category=False,
                 comp_weight=None,
                 love_dist_weight=0.78,
                 pnl_dist_weight=1.15,
                 rayleigh_dist_weight=0.55,
                 azi_exp_idx=0.5, azi_bins=12,
                 ref_dist=1.0):

        WeightConfigBase.__init__(self, "default",
                                  normalize_by_energy=normalize_by_energy,
                                  normalize_by_category=normalize_by_category)
        if comp_weight is None:
            self.comp_weight = {"Z": 2.0, "R": 1.0, "T": 2.0}
        else:
            self.comp_weight = comp_weight

        self.love_dist_weight = love_dist_weight
        self.pnl_dist_weight = pnl_dist_weight
        self.rayleigh_dist_weight = rayleigh_dist_weight
        self.azi_exp_idx = azi_exp_idx
        self.azi_bins = azi_bins
        self.ref_dist = ref_dist

    def __repr__(self):
        string = "Weight Strategy:\n"
        string += "mode: %s\n" % self.mode
        string += "normalize_by_energy: %s\n" % self.normalize_by_energy
        string += "normalize_by_category: %s\n" % self.normalize_by_category
        string += "component weight: %s\n" % self.comp_weight
        string += "pnl, rayleigh and love distance weights: %f, %f, %f\n" % (
            self.pnl_dist_weight, self.rayleigh_dist_weight,
            self.love_dist_weight)
        string += "number of azimuth bins: %d\n" % self.azi_bins
        string += "azimuth exponential index: %f\n" % self.azi_exp_idx
        return string
