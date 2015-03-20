#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    import numpy as np
except:
    msg = ("No module named numpy. "
           "Please install numpy first, it is needed before using pycmt3d.")
    raise ImportError(msg)

from __init__ import logger
from obspy import read
import os

class Window(object):
    """
    Obsd, synt and window information from one component of one station
    """
    def __init__(self, station=None, network=None, location=None, component=None, num_wins=0,
                 win_time=None, weight=None, obsd_fn=None, synt_fn=None,
                 datalist=None):
        """
        """
        self.station = station
        self.network = network
        self.location = location
        self.component = component
        self.num_wins = num_wins    # number of windows
        self.win_time = win_time    # window time
        self.weight = weight
        self.obsd_fn = obsd_fn
        self.synt_fn = synt_fn
        self.datalist = datalist

        # for weighting
        self.azimuth = None
        self.dist_in_km = None


class DataContainer(object):
    """
    Class that contains all necessary data and window information
    """
    def __init__(self, flexwin_file, par_list):
        """

        :param flexwin_file: old way of flexwin output file for cmt3d
        :param par_list: derivative parameter name list
        """
        self.flexwin_file = flexwin_file
        self.par_list = par_list
        self.window = []
        self.load_winfile()
        self.print_summary()

    def load_winfile(self):
        """
        old way of loading flexwin outputfile
        """
        with open(self.flexwin_file, "r") as f:
            num_file = int(f.readline().strip())
            if num_file == 0:
                return
            for idx in range(num_file):
                # keep the old format of cmt3d input
                obsd_fn = f.readline().strip()
                synt_fn = f.readline().strip()
                num_wins = int(f.readline().strip())
                win_time = np.zeros((num_wins,2))
                for iwin in range(num_wins):
                    [left, right] = f.readline().strip().split()
                    win_time[iwin, 0] = float(left)
                    win_time[iwin, 1] = float(right)
                win_obj = Window(num_wins=num_wins, win_time=win_time,
                                  obsd_fn=obsd_fn, synt_fn=synt_fn)
                # load all data, observed and synthetics into the object
                self.load_data(win_obj)
                self.window.append(win_obj)

        # count the total number of files and windows
        self.nfiles = len(self.window)
        nwins = 0
        for window in self.window:
            nwins += window.win_time.shape[0]
        self.nwins = nwins

    def load_data(self, win_obj):
        """
        Old way of loading obsd and synt data...

        :param win_obj:
        :return:
        """
        win_obj.datalist = {}
        obsd_fn = win_obj.obsd_fn
        synt_fn = win_obj.synt_fn
        # obsd
        obsd = read(obsd_fn)[0]
        win_obj.datalist['obsd'] = obsd
        win_obj.station = obsd.stats.station
        win_obj.network = obsd.stats.network
        win_obj.component = obsd.stats.channel
        win_obj.location = obsd.stats.location
        # synt
        win_obj.datalist['synt'] = read(synt_fn)[0]
        # other synt data will be referred as key value: Mrr, Mtt, Mpp, Mrt, Mrp, Mtp, dep, lat, lon, ctm, hdr
        for deriv_par in self.par_list:
            synt_dev_fn = synt_fn + "." + deriv_par
            win_obj.datalist[deriv_par] = read(synt_dev_fn)[0]

    def print_summary(self):
        nfiles_R = 0
        nfiles_T = 0
        nfiles_Z = 0
        nwins_R = 0
        nwins_T = 0
        nwins_Z = 0
        for window in self.window:
            if window.component[2:3] == "R":
                nfiles_R += 1
                nwins_R += window.num_wins
            elif window.component[2:3] == "T":
                nfiles_T += 1
                nwins_T += window.num_wins
            elif window.component[2:3] == "Z":
                nfiles_Z += 1
                nwins_Z += window.num_wins
            else:
                raise ValueError("Unrecognized compoent in windows: %s.%s.%s"
                                 %(window.station, window.network, window.component))

        logger.info("="*10 + "  Data Summary  " + "="*10)
        logger.info("Number of Deriv synt: %d" % len(self.par_list))
        logger.info("   Par: [%s]" % (', '.join(self.par_list)))
        logger.info("Number of data pairs: %d" %self.nfiles)
        logger.info("   [Z, R, T] = [%d, %d, %d]" %(nfiles_Z, nfiles_R, nfiles_T ))
        logger.info("Number of windows: %d"%self.nwins)
        logger.info("   [Z, R, T] = [%d, %d, %d]" %(nwins_Z, nwins_R, nwins_T))