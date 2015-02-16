#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration object for pycmt3d
"""
import numpy as np

class Config(object):

    def __init__(self, npar, dlocation, ddepth, dmoment, ddelta,
                 weight_data=True, weight_function=None,
                 station_correction=True, zero_trace=True,
                 double_couple=True, lamda_damping=0.0):
        self.npar = npar
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
        self.scale_par = np.array([SCALE_MOMENT, SCALE_MOMENT, SCALE_MOMENT, SCALE_MOMENT
                          SCALE_MOMENT, SCALE_MOMENT, SCALE_DEPTH, SCALE_DDELTA,
                          SCALE_DDELTA, SCALE_CTIME, SCALE_HDR])
        self.dcmt_par = np.array([dmoment, dmoment, dmoment, dmoment, dmoment,
                            dmoment, ddepth, ddelta, ddelta, 1.0, 1.0])/self.scale_par