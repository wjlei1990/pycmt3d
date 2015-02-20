#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration object for pycmt3d
"""
import numpy as np
import const

class Config(object):

    def __init__(self, npar, dlocation=const.SCALE_LOCATION, ddepth=const.SCALE_DEPTH,
                 dmoment=const.SCALE_MOMENT, ddelta=const.SCALE_DELTA,
                 weight_data=True, weight_function=None,
                 station_correction=True, zero_trace=True,
                 double_couple=True, lamda_damping=0.0):
        self.npar = npar
        if (not self.npar in [6, 7, 9, 10, 11]):
            print '''Error: the current npar (number of parameters) is ''', self.npar,'''
            The npar (number of parameters) must be 6, 7, 9, 10 or 11"
            When npar is 6: moment tensor
            When npar is 7: moment tensor + depth
            When npar is 9: moment tensor + depth + location(e.g. latitude and longitude)
            When npar is 10: moment tensor + depth + location + time
            When npar is 11: moment tensor + depth + location + time + half duration
            '''
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
