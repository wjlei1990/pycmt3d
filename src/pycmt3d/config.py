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

import const
import math
from __init__ import logger

def default_weight_function(kcmpnm, dist_in_km, azi_count, nwins,
                    comp_r_weight=2.0, comp_t_weight=2.0, comp_z_weight=1.5,
                    az_exp_weight=0.5, pnl_dist_weight=0.75, rayleigh_dist_weight=0.55,
                    love_dist_weight=0.55):

    data_weight = np.zeros(nwins)
    # component weight
    comp_direct = kcmpnm[2]
    if (comp_direct == 'Z'):
        cmp_weight = comp_z_weight
    elif (comp_direct == 'R'):
        cmp_weight = comp_r_weight
    elif (comp_direct == 'T'):
        cmp_weight = comp_t_weight
    else:
        raise ValueError('The direction of component of seismic data has to be Z, R or T')

    # distance weights
    # for global seismograms, this obviously has to be changed
    for win_idx in range(nwins):
        if (comp_direct == 'T'):
            dist_exp_weight = love_dist_weight
        else:
            if (nwins>1 and win_idx==0):
                dist_exp_weight = pnl_dist_weight
            else:
                dist_exp_weight = rayleigh_dist_weight

        # assemble data weights
        data_weight[win_idx] = cmp_weight * ((dist_in_km/const.REF_DIST) ** dist_exp_weight) / (azi_count ** az_exp_weight)

    return data_weight


class Config(object):

    def __init__(self, npar, dlocation=0.0, ddepth=0.0, dmoment=0.0, ddelta=0.0,
                 weight_data=True, weight_function=None,
                 station_correction=True, zero_trace=True,
                 double_couple=False, lamda_damping=0.0,
                 bootstrap=False, bootstrap_repeat=100):
        self.npar = npar
        if (not self.npar in [6, 7, 9, 10, 11]):
            print ('Error: the current npar (number of parameters) is ', self.npar)
            print ('The npar (number of parameters) must be 6, 7, 9, 10 or 11')
            print ('When npar is 6: moment tensor')
            print ('When npar is 7: moment tensor + depth')
            print ('When npar is 9: moment tensor + depth + location(e.g. latitude and longitude)')
            print ('When npar is 10: moment tensor + depth + location + time')
            print ('When npar is 11: moment tensor + depth + location + time + half duration')
            return None
        self.dlocation = dlocation
        self.ddepth = ddepth
        self.dmoment = dmoment
        self.ddelta = ddelta
        self.weight_data = weight_data
        if weight_function is not None:
            self.weight_function = weight_function
        else:
            logger.info("None weighting function specified..assign to default")
            self.weight_function = default_weight_function
        self.station_correction = station_correction
        self.zero_trace = zero_trace
        self.double_couple = double_couple
        self.lamda_damping = lamda_damping

        self.par_name = ("Mrr", "Mtt", "Mpp", "Mrt", "Mrp", "Mtp",
                         "dep", "lon", "lat", "ctm", "hdr")
        self.scale_par = np.array([const.SCALE_MOMENT, const.SCALE_MOMENT, const.SCALE_MOMENT,
                                   const.SCALE_MOMENT, const.SCALE_MOMENT, const.SCALE_MOMENT,
                                   const.SCALE_DEPTH, const.SCALE_DELTA, const.SCALE_DELTA,
                                   const.SCALE_CTIME, const.SCALE_HDUR])
        self.dcmt_par = np.array([self.dmoment, self.dmoment, self.dmoment, self.dmoment,
                                  self.dmoment, self.dmoment, self.ddepth, self.ddelta,
                                  self.ddelta, 1.0, 1.0])/self.scale_par
        self.bootstrap = bootstrap
        self.bootstrap_repeat = bootstrap_repeat

    # The function weight_function is to calculate the weight for different component and azimuths
    # The default value of input weights are based on my previous research, the user should modify it according to your circumstances

