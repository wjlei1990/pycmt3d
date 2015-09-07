#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
All the constants used in pycmt3d
"""

import numpy as np

PAR_LIST = ("Mrr", "Mtt", "Mpp", "Mrt", "Mrp", "Mtp",
            "dep", "lon", "lat", "ctm", "hdr")

# Mathmatical constants
PI = np.pi

# Scale of cmt parameters
# (latitude, longtitude, depth and moment centroid time and half duration)
SCALE_LONGITUDE = 0.01   # degree
SCALE_LATTITUDE = 0.01   # degree
SCALE_DEPTH = 1.0        # km
SCALE_MOMENT = 1.0e+23   # dyns*cm
SCALE_CTIME = 1.0        # seconds
SCALE_HDUR = 1.0         # seconds

# Maximum number of parameters
NPARMAX = 11

# Number of pars for moment only
NM = 6

# number of pars for moment+location only
NML = 9

# Small numbers
EPS2 = 1.0e-2
EPS5 = 1.0e-5

# Number of regions for azimuthal weighting
NREGIONS = 10

# Reference distance for Pnl, Rayleigh and Love wave weighting
REF_DIST = 100.0

# Earth's radius for depth scaling
R_EARTH = 6371  # km

# Max step in non-linear solvoer
NMAX_NL_ITER = 60

# subset ratio of bootstrap
BOOTSTRAP_SUBSET_RATIO = 0.4

# taper used in the inversion
taper_type = "tukey"
