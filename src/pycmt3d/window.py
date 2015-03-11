#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    import numpy as np
except:
    msg = ("No module named numpy. "
           "Please install numpy first, it is needed before using pycmt3d.")
    raise ImportError(msg)

class Window(object):

    def __init__(self, station, network, location, component, num_wins=0,
                 win_time=None, weight=None, obsd_fn=None, synt_fn=None,
                 datalist=None):

        self.station = station
        self.network = network
        self.location = location
        self.component = component
        self.num_wins = num_wins
        self.win_time = win_time
        self.weight = weight
        self.obsd_fn = obsd_fn
        self.synt_fn = synt_fn
        self.datalist = datalist

        # for weighting
        self.azimuth = None
        self.dist_in_km = None
