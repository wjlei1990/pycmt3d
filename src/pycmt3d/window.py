#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class Window(object):

    def __init__(self, station, network, location, component,
                 win_time, weight = 0.0, obsd_fn=None, synt_fn=None):

        self.station = station
        self.network = network
        self.location = location
        self.component = component
        self.win_time = win_time
        self.weight = 0.0
        self.obsd_fn = None
        self.synt_fn = None

