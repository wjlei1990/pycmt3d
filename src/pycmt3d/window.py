#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from __init__ import logger
from obspy import read
import time
import os
from obspy.core.util.geodetics import gps2DistAzimuth
from obspy import read_inventory


class Window(object):
    """
    Obsd, synt, deriv synt trace and window information from one component of one station.
    Also, window weighting, station azimuth, distance is also included.
    """

    def __init__(self, station=None, network=None, location=None, component=None, num_wins=0,
                 win_time=None, weight=None, obsd_fn=None, synt_fn=None,
                 datalist=None, tag=None, source=None):
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

        # Provenance information
        self.tag = tag
        self.source = source

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
        #self.latitude = self.datalist['synt'].stats.sac['stla']
        #self.longitude = self.datalist['synt'].stats.sac['stlo']
        dist_in_m, az, baz = gps2DistAzimuth(self.event_latitude, self.event_longitude, self.latitude, self.longitude)
        self.dist_in_km = dist_in_m / 1000.0
        self.azimuth = az


class DataContainer(object):
    """
    Class that contains methods that load data and window information
    """
    def __init__(self, par_list):
        """
        :param flexwin_file: old way of flexwin output file for cmt3d
        :param par_list: derivative parameter name list
        :param load_from_asdf: bool whether load from asdf file
        :param asdf_file_dict: asdf file dictionary.
        """
        self.par_list = par_list

        self.window = []
        self.npar = len(par_list)
        self.nfiles = 0
        self.nwins = 0

        self.elapse_time = 0.0

    def add_measurements_from_sac(self, flexwinfile, tag=None):
        """
        Add measurments(window and data) from the given flexwinfile and the data format should be sac

        :param flexwinfile:
        :return:
        """
        t1 = time.time()
        win_list = self.load_winfile(flexwinfile)
        for win_obj in win_list:
            self.load_data_from_sac(win_obj, tag=tag)

        self.window += win_list
        # count the total number of files and windows
        nfiles = len(win_list)
        self.nfiles += nfiles
        nwins = 0
        for window in win_list:
            nwins += window.win_time.shape[0]
        self.nwins += nwins
        t2 = time.time()
        logger.info("="*10 + " Measurements Loading " + "="*10)
        logger.info("Data loaded in sac format: %s" % flexwinfile)
        logger.info("Elapsed time: %5.2f s" %(t2-t1))
        logger.info("Number of files and window added: [%d, %d]" %(nfiles, nwins))

    def add_measurements_from_asdf(self, flexwinfile, asdf_file_dict, obsd_tag=None, synt_tag=None):
        """
        Add measurments(window and data) from the given flexwinfile and the data format should be asdf.
        Usually, you can leave the obsd_tag=None and synt_tag=None unless if you have multiple tags in
        asdf file.

        :param flexwinfile:
        :param asdf_file_dict:
        :param tag:
        :return:
        """
        t1 = time.time()
        # load window information
        win_list = self.load_winfile(flexwinfile)
        # load in the asdf data
        asdf_dataset = self.check_and_load_asdf_file(asdf_file_dict)
        # load data for each window
        for win_obj in win_list:
            self.load_data_from_asdf(win_obj, asdf_dataset, obsd_tag=obsd_tag, synt_tag=synt_tag)

        self.window += win_list
        # count the total number of files and windows
        nfiles = len(win_list)
        self.nfiles += nfiles
        nwins = 0
        for window in win_list:
            nwins += window.win_time.shape[0]
        self.nwins += nwins
        t2 = time.time()
        logger.info("="*10 + " Measurements Loading " + "="*10)
        logger.info("Data loaded in asdf format: %s" % flexwinfile)
        logger.info("Elapsed time: %5.2f s" %(t2-t1))
        logger.info("Number of files and window added: [%d, %d]" %(nfiles, nwins))

    def check_and_load_asdf_file(self, asdf_file_dict):
        from pyasdf import ASDFDataSet
        if not isinstance(asdf_file_dict, dict):
            raise ValueError("asdf_file_dict should be dictionary. Key from par_list and "
                             "value is the asdf file name")
        if len(asdf_file_dict) != (self.npar+2):
            raise ValueError("par_list is not consistent with asdf_file_dict")
        for key in self.par_list:
            if key not in asdf_file_dict.keys():
                raise ValueError("key in par_list is not in asdf_file_dict")
        if 'obsd' not in asdf_file_dict.keys():
            raise ValueError("No obsd asdf file found in asdf_file_dict")
        if 'synt' not in asdf_file_dict.keys():
            raise ValueError("No synt asdf file found in asdf_file_dict")
        dataset = {}
        dataset['obsd'] = ASDFDataSet(asdf_file_dict['obsd'])
        dataset['synt'] = ASDFDataSet(asdf_file_dict['synt'])
        for deriv_par in self.par_list:
            dataset[deriv_par] = ASDFDataSet(asdf_file_dict[deriv_par])
        return dataset

    @staticmethod
    def load_winfile(flexwin_file):
        """
        old way of loading flexwin outputfile
        """
        win_list = []
        with open(flexwin_file, "r") as f:
            num_file = int(f.readline().strip())
            if num_file == 0:
                logger.warnning("Nothing in flexwinfile: %s" %flexwin_file)
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
                    win_list.append(win_obj)
        return win_list

    def load_data_from_sac(self, win_obj, tag=None):
        """
        Old way of loading obsd and synt data...

        :param win_obj:
        :return:
        """
        win_obj.datalist = {}
        win_obj.tag = {}
        obsd_fn = win_obj.obsd_fn
        synt_fn = win_obj.synt_fn
        # obsd
        obsd = read(obsd_fn)[0]
        win_obj.datalist['obsd'] = obsd
        win_obj.tag['obsd'] =tag
        win_obj.station = obsd.stats.station
        win_obj.network = obsd.stats.network
        win_obj.component = obsd.stats.channel
        win_obj.location = obsd.stats.location
        # synt
        win_obj.datalist['synt'] = read(synt_fn)[0]
        win_obj.tag['synt'] = tag
        # other synt data will be referred as key value: Mrr, Mtt, Mpp, Mrt, Mrp, Mtp, dep, lat, lon, ctm, hdr
        for deriv_par in self.par_list:
            synt_dev_fn = synt_fn + "." + deriv_par
            win_obj.datalist[deriv_par] = read(synt_dev_fn)[0]
            win_obj.tag[deriv_par] = tag

        # station information
        win_obj.longitude = win_obj.datalist['synt'].stats.sac['stlo']
        win_obj.latitude = win_obj.datalist['synt'].stats.sac['stla']
        
        # specify metadata infor
        win_obj.source = "sac"

    def load_data_from_asdf(self, win, asdf_ds, obsd_tag=None, synt_tag=None):
        """
        load data from asdf file

        :return:
        """
        # trace
        win.datalist = {}
        win.tag = {}
        win.datalist['obsd'], win.tag['obsd'] = self.get_trace_from_asdf(win.obsd_fn, asdf_ds['obsd'], obsd_tag)
        win.datalist['synt'], win.tag['synt'] = self.get_trace_from_asdf(win.synt_fn, asdf_ds['synt'], synt_tag)
        for deriv_par in self.par_list:
            win.datalist[deriv_par], win.tag[deriv_par] = self.get_trace_from_asdf(win.synt_fn, asdf_ds[deriv_par], synt_tag)

        win.station = win.datalist['obsd'].stats.station
        win.network = win.datalist['obsd'].stats.network
        win.component = win.datalist['obsd'].stats.channel
        win.location = win.datalist['obsd'].stats.location

        # station information
        inv = self.get_stationxml_from_asdf(win.obsd_fn, asdf_ds['obsd'])
        win.latitude = float(inv[0][0].latitude)
        win.longitude = float(inv[0][0].longitude)

        # specify metadata infor
        win.source = "asdf"

    def get_stationxml_from_asdf(self, station_string, asdf_handle):
        """
        Used to extrace station location information from stationxml in asdf

        """
        station_string = os.path.basename(station_string)
        station_info = station_string.split(".")
        if len(station_info) == 5:
            [network, station, loc, comp, type] = station_info
        elif len(station_info) == 4:
            [network, station, comp, type] = station_info
        else:
            raise ValueError("station string not correct:%s" %station_info)

        if len(network) >= 3 and len(station) <=2:
            # in case people have different naming conventions
            temp_string = network
            network = station
            station = temp_string

        station_name = network + "_" + station
        # get the tag
        st = getattr(asdf_handle.waveforms, station_name)
        inv = getattr(st,'StationXML')
        return inv

    def get_trace_from_asdf(self, station_string, asdf_handle, tag):
        """
        Used to extract a specific trace out of an asdf file.

        :param station_string:
        :param asdf_handle:
        :param tag:
        :return:
        """
        # just in case people put the whole path, which has no meaning if asy14ydf is used
        station_string = os.path.basename(station_string)
        station_info = station_string.split(".")
        if len(station_info) == 5:
            [network, station, loc, comp, type] = station_info
        elif len(station_info) == 4:
            [network, station, comp, type] = station_info
        else:
            raise ValueError("station string not correct:%s" %station_info)

        if len(network) >= 3 and len(station) <=2:
            # in case people have different naming conventions
            temp_string = network
            network = station
            station = temp_string

        station_name = network + "_" + station
        # get the tag
        st = getattr(asdf_handle.waveforms, station_name)
        attr_list = dir(st)
        attr_list.remove('StationXML')
        if tag is None or tag == "":
            if len(attr_list) != 1:
                raise ValueError("More that 1 data tags in obsd asdf file. For this case, you need specify obsd_tag:%s" 
                                % attr_list)
            stream = getattr(st, attr_list[0])
        else:
            stream = getattr(st, tag)
        if len(station_info) == 5:
            tr = stream.select(network=network, station=station, channel=comp, location=loc)[0]
        else:
            tr = stream.select(network=network, station=station, channel=comp)[0]

        return tr, tag
    
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
