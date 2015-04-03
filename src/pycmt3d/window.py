#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from __init__ import logger
from obspy import read
import time
import os
from obspy.core.util.geodetics import gps2DistAzimuth


class Window(object):
    """
    Obsd, synt, deriv synt trace and window information from one component of one station.
    Also, window weighting, station azimuth, distance is also included.
    """

    def __init__(self, station=None, network=None, location=None, component=None, num_wins=0,
                 win_time=None, weight=None, obsd_fn=None, synt_fn=None,
                 datalist=None):
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

        # location
        self.latitude = None
        self.longitude = None
        # event location
        self.event_latitude = None
        self.event_longitude = None

        # for weighting
        self.azimuth = None
        self.dist_in_km = None
        self.energy = np.zeros(num_wins)

    def win_energy(self, mode='data_and_synt'):
        """
        Calculate energy inside the window

        :param mode: if mode == 'data_and_synt', then energy is calculated by sqrt(obsd_energy * synt_energy); \
        if mode == 'data_only', then energy is calculated by obsd_energy only
        """
        obsd = self.datalist['obsd']
        synt = self.datalist['synt']
        dt = obsd.stats.delta
        if mode.lower() not in ['data_and_synt', 'data_only']:
            raise ValueError("Weight mode incorrect: 1) data_and_synt; 2) data_only")
        for _idx in range(self.num_wins):
            istart_d = int(self.win_time[_idx, 0]/dt)
            iend_d = int(self.win_time[_idx, 1]/dt)
            if mode.lower() == "data_and_synt":
                self.energy[_idx] = np.sqrt(np.sum(obsd.data[istart_d:iend_d]**2*dt) *
                                            np.sum(synt.data[istart_d:iend_d]**2*dt))
            elif mode.lower() == "data_only":
                self.energy[_idx] = np.sum(obsd.data[istart_d:iend_d]**2*dt)

    def get_location_info(self, cmtsource):
        """
        calculating azimuth and distance, and then store it

        :param cmtsource: cmt source
        :return:
        """
        self.event_latitude = cmtsource.latitude
        self.event_longitude = cmtsource.longitude
        # calculate location
        self.latitude = self.datalist['synt'].stats.sac['stla']
        self.longitude = self.datalist['synt'].stats.sac['stlo']
        dist_in_m, az, baz = gps2DistAzimuth(self.event_latitude, self.event_longitude, self.latitude, self.longitude)
        self.dist_in_km = dist_in_m / 1000.0
        self.azimuth = az


class DataContainer(object):
    """
    Class that contains methods that load data and window information
    """
    def __init__(self, flexwin_file, par_list, load_from_asdf=False, asdf_file_dict=None):
        """
        :param flexwin_file: old way of flexwin output file for cmt3d
        :param par_list: derivative parameter name list
        :param load_from_asdf: bool whether load from asdf file
        :param asdf_file_dict: asdf file dictionary.
        """
        self.flexwin_file = flexwin_file
        self.par_list = par_list
        self.load_from_asdf = load_from_asdf
        self.asdf_file_dict = asdf_file_dict

        self.window = []
        self.npar = len(par_list)
        self.nfiles = 0
        self.nwins = 0

        time_stamp1 = time.time()
        if self.load_from_asdf:
            self.asdf_ds = None
            self.check_and_load_asdf_file()
            self.load_winfile()
        else:
            self.load_winfile()

        time_stamp2 = time.time()
        self.elapsed_time = time_stamp2 - time_stamp1

        self.print_summary()

    def check_and_load_asdf_file(self):
        from pyasdf import ASDFDataSet
        if not isinstance(self.asdf_file_dict, dict):
            raise ValueError("asdf_file_dict should be dictionary. Key from par_list and "
                             "value is the asdf file name")
        if len(self.asdf_file_dict) != (self.npar+2):
            raise ValueError("par_list is not consistent with asdf_file_dict")
        for key in self.par_list:
            if key not in self.asdf_file_dict.keys():
                raise ValueError("key in par_list is not in asdf_file_dict")
        if 'obsd' not in self.asdf_file_dict.keys():
            raise ValueError("No obsd asdf file found in asdf_file_dict")
        if 'synt' not in self.asdf_file_dict.keys():
            raise ValueError("No synt asdf file found in asdf_file_dict")
        dataset = {}
        dataset['obsd'] = ASDFDataSet(self.asdf_file_dict['obsd'])
        dataset['synt'] = ASDFDataSet(self.asdf_file_dict['synt'])
        for deriv_par in self.par_list:
            dataset[deriv_par] = ASDFDataSet(self.asdf_file_dict[deriv_par])
        self.asdf_ds = dataset

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
                win_time = np.zeros((num_wins, 2))
                for iwin in range(num_wins):
                    [left, right] = f.readline().strip().split()
                    win_time[iwin, 0] = float(left)
                    win_time[iwin, 1] = float(right)
                win_obj = Window(num_wins=num_wins, win_time=win_time,
                                 obsd_fn=obsd_fn, synt_fn=synt_fn)
                # load all data, observed and synthetics into the object
                if self.load_from_asdf:
                    self.load_data_from_asdf(win_obj)
                else:
                    self.load_data_from_sac(win_obj)
                self.window.append(win_obj)

        # count the total number of files and windows
        self.nfiles = len(self.window)
        nwins = 0
        for window in self.window:
            nwins += window.win_time.shape[0]
        self.nwins = nwins

    def load_data_from_sac(self, win_obj):
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

    def load_data_from_asdf(self, win_obj):
        """
        load data from asdf file

        :return:
        """
        win_obj.datalist['obsd'] = self.get_obsd_trace_from_asdf(win_obj.obsd_fn, self.asdf_ds['obsd'])
        win_obj.datalist['synt'] = self.get_synt_trace_from_asdf(win_obj.synt_fn, self.asdf_ds['synt'])
        for deriv_par in self.par_list:
            win_obj.datalist[deriv_par] = self.get_synt_trace_from_asdf(win_obj.synt_fn, self.asdf_file_dict[deriv_par])

    def get_obsd_trace_from_asdf(self, obsd_fn, dataset):
        obsd_fn = os.path.basename(obsd_fn)
        sta, network, loc, comp, type = obsd_fn.split(".")
        


    def get_synt_trace_from_asdf(self, synt_fn, dataset):
        return 0

    def print_summary(self):
        """
        Print summary of data container

        :return:
        """
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
                                 % (window.station, window.network, window.component))

        logger.info("="*10 + "  Data Summary  " + "="*10)
        logger.info("Number of Deriv synt: %d" % len(self.par_list))
        logger.info("   Par: [%s]" % (', '.join(self.par_list)))
        logger.info("Number of data pairs: %d" % self.nfiles)
        logger.info("   [Z, R, T] = [%d, %d, %d]" % (nfiles_Z, nfiles_R, nfiles_T))
        logger.info("Number of windows: %d" % self.nwins)
        logger.info("   [Z, R, T] = [%d, %d, %d]" % (nwins_Z, nwins_R, nwins_T))
        logger.info("Loading takes %6.2f seconds" % self.elapsed_time)