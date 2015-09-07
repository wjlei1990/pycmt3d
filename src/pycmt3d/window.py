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

        self.tshift = None
        self.dlnA = None

    def win_energy(self, mode='data_and_synt'):
        """
        Calculate energy inside the window

        :param mode: if mode == 'data_and_synt', then energy is calculated by sqrt(obsd_energy * synt_energy); \
        if mode == 'data_only', then energy is calculated by obsd_energy only
        """
        obsd = self.datalist['obsd']
        synt = self.datalist['synt']
        dt = obsd.stats.delta
        for _idx in range(self.num_wins):
            istart_d = int(self.win_time[_idx, 0]/dt)
            iend_d = int(self.win_time[_idx, 1]/dt)
            if iend_d - istart_d <= 1:
                raise ValueError("Window length < 1, incorrect!")
            if mode.lower() == "data_and_synt":
                self.energy[_idx] = np.sqrt(np.sum(obsd.data[istart_d:iend_d]**2*dt) *
                                            np.sum(synt.data[istart_d:iend_d]**2*dt))
            elif mode.lower() == "data_only":
                self.energy[_idx] = np.sum(obsd.data[istart_d:iend_d]**2*dt)
            elif mode.lower() == "data_average_only":
                self.energy[_idx] = np.sum(obsd.data[istart_d:iend_d]**2*dt/(iend_d - istart_d))
            elif mode.lower() == "data_abs_only":
                self.energy[_idx] = np.sum(np.abs(obsd.data[istart_d:iend_d]))
            else:
                raise ValueError("Weight mode incorrect: 1) data_and_synt; 2) data_only "
                                "3) data_average_only; 4) data_abs_only")

    def get_location_info(self, cmtsource):
        """
        calculating azimuth and distance, and then store it

        :param cmtsource: cmt source
        :return:
        """
        self.event_latitude = cmtsource.latitude
        self.event_longitude = cmtsource.longitude
        # calculate location
        dist_in_m, az, baz = gps2DistAzimuth(self.event_latitude, self.event_longitude, self.latitude, self.longitude)
        self.dist_in_km = dist_in_m / 1000.0
        self.azimuth = az


class DataContainer(object):
    """
    Class that contains methods that load data and window information
    """
    def __init__(self, par_list=[]):
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

    def add_measurements_from_sac(self, flexwinfile, tag="untaged", initial_weight=1.0, load_mode="obsolute_time"):
        """
        Add measurments(window and data) from the given flexwinfile and the data format should be sac

        :param flexwinfile:
        :return:
        """
        load_mode = load_mode.lower()
        if load_mode not in ["obsolute_time", "relative_time"]:
            raise ValueError("load_winfile mode incorrect: 1)obsolute_time 2)relative_time")

        t1 = time.time()
        win_list = self.load_winfile(flexwinfile, initial_weight=initial_weight)
        for win_obj in win_list:
            self.load_data_from_sac(win_obj, tag=tag, mode=load_mode)

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
        logger.info("Elapsed time: %5.2f s" % (t2-t1))
        logger.info("Number of files and window added: [%d, %d]" % (nfiles, nwins))

    def add_measurements_from_asdf(self, flexwinfile, asdf_file_dict, obsd_tag=None, synt_tag=None,
                                   stationfile=None, initial_weight=1.0):
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
        win_list = self.load_winfile(flexwinfile, initial_weight=initial_weight)
        # load in the asdf data
        asdf_dataset = self.check_and_load_asdf_file(asdf_file_dict)
        if stationfile is not None:
            station_info = self.load_station_from_text(stationfile)
        else:
            station_info = None
        # load data for each window
        for win_obj in win_list:
            self.load_data_from_asdf(win_obj, asdf_dataset, obsd_tag=obsd_tag, synt_tag=synt_tag,
                                     station_dict=station_info)

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
        logger.info("Elapsed time: %5.2f s" % (t2-t1))
        logger.info("Number of files and window added: [%d, %d]" % (nfiles, nwins))

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
    def load_winfile(flexwin_file, initial_weight=1.0):
        """
        old way of loading flexwin outputfile
        """
        win_list = []
        with open(flexwin_file, "r") as f:
            num_file = int(f.readline().strip())
            if num_file == 0:
                logger.warning("Nothing in flexwinfile: %s" % flexwin_file)
                return
            for idx in range(num_file):
                # keep the old format of cmt3d input
                obsd_fn = f.readline().strip()
                synt_fn = f.readline().strip()
                num_wins = int(f.readline().strip())
                win_time = np.zeros((num_wins, 2))
                for iwin in range(num_wins):
                    content = f.readline().strip().split()
                    win_time[iwin, 0] = float(content[0])
                    win_time[iwin, 1] = float(content[1])
                win_weight = initial_weight * np.ones(num_wins)
                win_obj = Window(num_wins=num_wins, win_time=win_time,
                                 obsd_fn=obsd_fn, synt_fn=synt_fn,
                                 weight=win_weight)
                win_list.append(win_obj)
        return win_list

    def load_data_from_sac(self, win_obj, tag=None, mode=None):
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
        win_obj.tag['obsd'] = tag
        win_obj.station = obsd.stats.station
        win_obj.network = obsd.stats.network
        win_obj.component = obsd.stats.channel
        win_obj.location = obsd.stats.location

        #calibrate window time if needed
        if mode == "relative_time":
            b_tshift = obsd.stats.sac['b']
            for _ii in range(win_obj.win_time.shape[0]):
                for _jj in range(win_obj.win_time.shape[1]):
                    win_obj.win_time[_ii, _jj] -= b_tshift
                    # WJL: not a good way
                    win_obj.win_time[_ii, _jj] = max(win_obj.win_time[_ii, _jj], 0.0)

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

    def load_data_from_asdf(self, win, asdf_ds, obsd_tag=None, synt_tag=None, station_dict=None):
        """
        load data from asdf file

        :return:
        """
        # trace
        win.datalist = {}
        win.tag = {}
        #print "obsd"
        win.datalist['obsd'], win.tag['obsd'] = self.get_trace_from_asdf(win.obsd_fn, asdf_ds['obsd'], obsd_tag)
        #print "synt"
        win.datalist['synt'], win.tag['synt'] = self.get_trace_from_asdf(win.synt_fn, asdf_ds['synt'], synt_tag)
        for deriv_par in self.par_list:
            #print deriv_par
            win.datalist[deriv_par], win.tag[deriv_par] = self.get_trace_from_asdf(win.synt_fn, asdf_ds[deriv_par], synt_tag)

        win.station = win.datalist['obsd'].stats.station
        win.network = win.datalist['obsd'].stats.network
        win.component = win.datalist['obsd'].stats.channel
        win.location = win.datalist['obsd'].stats.location

        # station information
        if station_dict is None:
            inv = self.get_stationxml_from_asdf(win.obsd_fn, asdf_ds['obsd'])
            win.latitude = float(inv[0][0].latitude)
            win.longitude = float(inv[0][0].longitude)
        else:
            key = "_".join([win.network, win.station])
            win.latitude = station_dict[key][0]
            win.longitude = station_dict[key][1]

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
            raise ValueError("station string not correct:%s" % station_info)

        if len(network) >= 3 and len(station) <= 2:
            # in case people have different naming conventions
            temp_string = network
            network = station
            station = temp_string

        station_name = network + "_" + station
        # get the tag
        st = getattr(asdf_handle.waveforms, station_name)
        inv = getattr(st, 'StationXML')
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
            raise ValueError("station string not correct:%s" % station_info)

        if len(network) >= 3 and len(station) <= 2:
            # in case people have different naming conventions
            temp_string = network
            network = station
            station = temp_string

        station_name = network + "_" + station
        # get the tag
        st = getattr(asdf_handle.waveforms, station_name)
        attr_list = dir(st)
        if 'StationXML' in attr_list:
            attr_list.remove('StationXML')
        if tag is None or tag == "":
            if len(attr_list) != 1:
                raise ValueError("More that 1 data tags in obsd asdf file. For this case, you need specify obsd_tag:%s"
                                 % attr_list)
            stream = getattr(st, attr_list[0])
            tag = attr_list[0]
        else:
            stream = getattr(st, tag)
        if len(station_info) == 5:
            tr = stream.select(network=network, station=station, channel="*%s" %comp[-1:], location=loc)[0]
        else:
            tr = stream.select(network=network, station=station, channel="*%s" %comp[-1:])[0]

        return tr, tag

    @staticmethod
    def load_station_from_text(stationfile):
        station_dict = {}
        with open(stationfile, 'r') as f:
            content = f.readlines()
            content = [ line.rstrip('\n') for line in content ]
            for line in content:
                info = line.split()
                key = "_".join([info[1], info[0]])
                #print key
                station_dict[key] = [float(info[2]), float(info[3]), float(info[4])]
        return station_dict

    def write_new_syn_file(self, format="sac", outputdir=".", eventname=None, suffix=None):
        """
        Write out new synthetic file based on new cmtsolution
        :return:
        """
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        # sort the new synthetic data
        new_synt_dict = {}
        for window in self.window:
            tag = window.tag['synt']
            if tag not in new_synt_dict.keys():
                new_synt_dict[tag] = []
            new_synt_dict[tag].append(window)

        if format.upper() == "SAC":
            for tag, win_array in new_synt_dict.iteritems():
                if eventname is None:
                    targetdir = os.path.join(outputdir, tag)
                else:
                    targetdir = os.path.join(outputdir, "%s_%s" %(eventname, tag))
                if not os.path.exists(targetdir):
                    os.makedirs(targetdir)
                for window in win_array:
                    sta = window.station
                    nw = window.network
                    component = window.component
                    location = window.location
                    filename = "%s.%s.%s.%s.sac" % (sta, nw, location, component)
                    #filename = "%s.%s.BH%s" % (sta, nw, component[2:3])
                    outputfn = os.path.join(targetdir, filename)
                    new_synt = window.datalist['new_synt']
                    new_synt.write(outputfn, format='SAC')
        elif format.upper() == "ASDF":
            from pyasdf import ASDFDataSet
            for tag, win_array in new_synt_dict.iteritems():
                if eventname is None:
                    outputfn = os.path.join(outputdir, "new_synt.%s.h5" % tag)
                else:
                    if suffix is None:
                        outputfn = os.path.join(outputdir, "%s.new_synt.%s.h5" % (eventname, tag))
                    else:
                        outputfn = os.path.join(outputdir, "%s.%s.new_synt.%s.h5" % (eventname, suffix, tag))
                if os.path.exists(outputfn):
                    os.remove(outputfn)
                ds = ASDFDataSet(outputfn)
                for window in win_array:
                    ds.add_waveforms(window.datalist['new_synt'], tag=tag)
                # add stationxml


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
