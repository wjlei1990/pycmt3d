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

class Config(object):

    def __init__(self, npar, dlocation=const.SCALE_LOCATION, ddepth=const.SCALE_DEPTH,
                 dmoment=const.SCALE_MOMENT, ddelta=const.SCALE_DELTA,
                 weight_data=True, weight_function=None,
                 station_correction=True, zero_trace=True,
                 double_couple=True, lamda_damping=0.0):
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
        self.dmomoent = dmoment
        self.ddelta = ddelta
        self.weight_data = weight_data
        self.weight_function = weight_function
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

    # The function weight_function is to calculate the weight for different component and azimuths
    # The default value of input weights are based on my previous research, the user should modify it according to your circumstances
    def weight_function(kcmpnm, azimuth, dist_in_km, window_index=0,nwins=1,
                        comp_z_weight=2.0, comp_t_weight=2.0, comp_z_weight=1.0,
                        az_exp_weight=0.5, pnl_dist_weight=1.15, rayleigh_dist_weight=0.55
                        love_dist_weight=0.78):
        daz = 360.0 / const.NREGIONS
        naz = np.zeros(const.NREGIONS+1)  # start with water level of 0

        # component weight
        comp_direct = kcmpnm[2]
        if (comp_direct == 'Z'):
            cmp_weight = comp_z_weight
        elif (comp_direct == 'R'):
            cmp_weight = comp_r_weight
        elif (comp_direct == 'T'):
            cmp_weight = comp_t_weight
        else:
            raise ValueError('The direction of component of seismic data has to be 'Z', 'R', or 'T'')

        # distance weights
        # for global seismograms, this obviously has to be changed
        if (comp_direct == 'T'):
            dist_exp_weight = love_dist_weight
        else:
            if (nwins>1 and window_index==0):
                dist_exp_weight = pnl_dist_weight
            else:
                dist_exp_weigth = rayleigh_dist_weight


        # azimuth counts
        k = floor(azimuth / daz) 
        if ( k<0 or k>const.NREGIONS):
            raise ValueError ('Error bining azimuth')
        naz[k] += 1

        # assemble data weights
        data_weight = cmp_weight * (dist_in_km/const.REF_DIST) ** dist_exp_weight / naz ** az_exp_weight

        return data_weight
