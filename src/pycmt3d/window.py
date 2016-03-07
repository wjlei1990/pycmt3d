#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from __init__ import logger
from obspy import read
import time
import os
from obspy.core.util.geodetics import gps2DistAzimuth
try:
    from pyasdf import ASDFDataSet
except ImportError:
    print("Can not import pyasdf. ASDF not supported then")


class Window(object):
    """
    Obsd, synt, deriv synt trace and window information from one
    component of one station. Also, window weighting, station azimuth,
    distance is also included.
    """

    def __init__(self, station=None, network=None, location=None,
                 component=None, num_wins=0, win_time=None, weight=None,
                 obsd_id=None, synt_id=None, datalist=None, tag=None,
                 source=None):

        self.station = station
        self.network = network
        self.location = location
        self.component = component
        self.num_wins = num_wins    # number of windows
        self.win_time = win_time    # window time
        self.weight = weight
        self.obsd_id = obsd_id
        self.synt_id = synt_id
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

        :param mode: if mode == 'data_and_synt', then energy is calculated
        by sqrt(obsd_energy * synt_energy); if mode == 'data_only', then
        energy is calculated by obsd_energy only
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
                self.energy[_idx] = np.sqrt(
                    np.sum(obsd.data[istart_d:iend_d]**2*dt) *
                    np.sum(synt.data[istart_d:iend_d]**2*dt))
            elif mode.lower() == "data_only":
                self.energy[_idx] = np.sum(obsd.data[istart_d:iend_d]**2*dt)
            elif mode.lower() == "data_average_only":
                self.energy[_idx] = np.sum(
                    obsd.data[istart_d:iend_d]**2*dt/(iend_d - istart_d))
            elif mode.lower() == "data_abs_only":
                self.energy[_idx] = np.sum(np.abs(obsd.data[istart_d:iend_d]))
            else:
                raise ValueError(
                        "Weight mode incorrect: 1) data_and_synt; 2) "
                        "data_only 3) data_average_only; 4) data_abs_only")

    def get_location_info(self, cmtsource):
        """
        calculating azimuth and distance, and then store it

        :param cmtsource: cmt source
        :return:
        """
        self.event_latitude = cmtsource.latitude
        self.event_longitude = cmtsource.longitude
        # calculate location
        dist_in_m, az, baz = gps2DistAzimuth(
            self.event_latitude, self.event_longitude,
            self.latitude, self.longitude)
        self.dist_in_km = dist_in_m / 1000.0
        self.azimuth = az


class DataContainer(object):
    """
    Class that contains methods that load data and window information
    """
    def __init__(self, par_list=None):
        """
        :param par_list: derivative parameter name list
        """
        if par_list is None:
            par_list = []
        self.par_list = par_list

        self.window = []
        self.npar = len(par_list)
        self.nfiles = 0
        self.nwins = 0

        self.elapse_time = 0.0

        # store asdf dataset if asdf mode
        self.asdf_file_dict = None

    def add_measurements_from_sac(self, flexwinfile, tag="untaged",
                                  initial_weight=1.0,
                                  load_mode="obsolute_time"):
        """
        Add measurments(window and data) from the given flexwinfile
        and the data format should be sac

        :param flexwinfile:
        :return:
        """
        load_mode = load_mode.lower()
        if load_mode not in ["obsolute_time", "relative_time"]:
            raise ValueError("load_winfile mode incorrect: 1)obsolute_time"
                             "2)relative_time")

        t1 = time.time()
        win_list = self.load_winfile(flexwinfile,
                                     initial_weight=initial_weight)
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
        logger.info("Number of files and window added: [%d, %d]"
                    % (nfiles, nwins))

    def add_measurements_from_asdf(self, flexwinfile, asdf_file_dict,
                                   obsd_tag=None, synt_tag=None,
                                   external_stationfile=None,
                                   initial_weight=1.0,
                                   winfile_format="txt"):
        """
        Add measurments(window and data) from the given flexwinfile and
        the data format should be asdf. Usually, you can leave the
        obsd_tag=None and synt_tag=None unless if you have multiple tags in
        asdf file.

        :param flexwinfile:
        :param asdf_file_dict:
        :return:
        """
        t1 = time.time()
        # load window information
        win_list = self.load_winfile(flexwinfile,
                                     initial_weight=initial_weight,
                                     file_format=winfile_format)
        # load in the asdf data
        asdf_dataset = self.check_and_load_asdf_file(asdf_file_dict)
        self.asdf_file_dict = asdf_file_dict
        if external_stationfile is not None:
            station_info = \
                self.load_station_from_text(external_stationfile)
        else:
            station_info = None

        # load data for each window
        for win_obj in win_list:
            self.load_data_from_asdf(
                win_obj, asdf_dataset, obsd_tag=obsd_tag,
                synt_tag=synt_tag, station_dict=station_info)

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
        logger.info("Number of files and window added: [%d, %d]"
                    % (nfiles, nwins))

    def check_and_load_asdf_file(self, asdf_file_dict):

        if not isinstance(asdf_file_dict, dict):
            raise ValueError("asdf_file_dict should be dictionary. Key from "
                             "par_list and value is the asdf file name")
        if len(asdf_file_dict) != (self.npar+2):
            raise ValueError("par_list is not consistent with asdf_file_dict")
        for key in self.par_list:
            if key not in asdf_file_dict.keys():
                raise ValueError("key in par_list is not in asdf_file_dict")
        if 'obsd' not in asdf_file_dict.keys():
            raise ValueError("No obsd asdf file found in asdf_file_dict")
        if 'synt' not in asdf_file_dict.keys():
            raise ValueError("No synt asdf file found in asdf_file_dict")
        dataset = dict()
        dataset['obsd'] = ASDFDataSet(asdf_file_dict['obsd'])
        dataset['synt'] = ASDFDataSet(asdf_file_dict['synt'])
        for deriv_par in self.par_list:
            dataset[deriv_par] = ASDFDataSet(asdf_file_dict[deriv_par])
        return dataset

    def load_winfile(self, flexwin_file, initial_weight=1.0,
                     file_format="txt"):
        """
        loading window file. Currently supports two format:
        1) txt; 2) json
        """
        file_format = file_format.lower()
        if file_format not in ['txt', 'json']:
            raise ValueError("Supported format: 1) txt; 2) json.")
        if file_format == "txt":
            win_list = self.load_winfile_txt(flexwin_file,
                                             initial_weight=initial_weight)
        elif file_format == "json":
            win_list = self.load_winfile_json(flexwin_file,
                                              initial_weight=initial_weight)
        else:
            raise NotImplementedError("Window file format not support:"
                                      "%s" % file_format)
        return win_list

    @staticmethod
    def load_winfile_txt(flexwin_file, initial_weight=1.0):
        """
        Read the txt format of  window file(see the documentation
        online).

        :param flexwin_file:
        :param initial_weight:
        :return:
        """
        win_list = []
        with open(flexwin_file, "r") as f:
            num_file = int(f.readline().strip())
            if num_file == 0:
                logger.warning("Nothing in flexwinfile: %s" % flexwin_file)
                return []
            for idx in range(num_file):
                # keep the old format of cmt3d input
                obsd_id = f.readline().strip()
                synt_id = f.readline().strip()
                num_wins = int(f.readline().strip())
                win_time = np.zeros((num_wins, 3))
                win_weight = np.zeros(num_wins)
                for iwin in range(num_wins):
                    content = f.readline().strip().split()
                    win_time[iwin, 0] = float(content[0])
                    win_time[iwin, 1] = float(content[1])
                    if len(content) == 3:
                        win_weight[iwin] = float(content[2])
                    else:
                        win_weight[iwin] = initial_weight
                win_obj = Window(num_wins=num_wins, win_time=win_time,
                                 obsd_id=obsd_id, synt_id=synt_id,
                                 weight=win_weight)
                win_list.append(win_obj)
        return win_list

    @staticmethod
    def load_winfile_json(flexwin_file, initial_weight=1.0):
        """
        Read the json format of window file

        :param flexwin_file:
        :param initial_weight:
        :return:
        """
        import json
        win_list = []
        with open(flexwin_file, 'r') as fh:
            content = json.load(fh)
            for _sta, _channel in content.iteritems():
                for _chan_win in _channel:
                    num_wins = len(_chan_win)
                    obsd_id = _chan_win[0]["obsd_id"]
                    synt_id = _chan_win[0]["synt_id"]
                    win_time = np.zeros([num_wins, 2])
                    win_weight = np.zeros(num_wins)
                    for _idx, _win in enumerate(_chan_win):
                        win_time[_idx, 0] = _win["relative_starttime"]
                        win_time[_idx, 1] = _win["relative_endtime"]
                        if "initial_weighting" in _win.keys():
                            win_weight[_idx] = _win["initial_weighting"]
                        else:
                            win_weight[_idx] = initial_weight
                    win_obj = Window(num_wins=num_wins, win_time=win_time,
                                     obsd_id=obsd_id, synt_id=synt_id,
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
        obsd_id = win_obj.obsd_id
        synt_id = win_obj.synt_id
        # obsd
        obsd = read(obsd_id)[0]
        win_obj.datalist['obsd'] = obsd
        win_obj.tag['obsd'] = tag
        win_obj.station = obsd.stats.station
        win_obj.network = obsd.stats.network
        win_obj.component = obsd.stats.channel
        win_obj.location = obsd.stats.location

        # calibrate window time if needed
        if mode == "relative_time":
            b_tshift = obsd.stats.sac['b']
            for _ii in range(win_obj.win_time.shape[0]):
                for _jj in range(win_obj.win_time.shape[1]):
                    win_obj.win_time[_ii, _jj] -= b_tshift
                    # WJL: not a good way
                    win_obj.win_time[_ii, _jj] = \
                        max(win_obj.win_time[_ii, _jj], 0.0)

        # synt
        win_obj.datalist['synt'] = read(synt_id)[0]
        win_obj.tag['synt'] = tag
        # other synt data will be referred as key value:
        # Mrr, Mtt, Mpp, Mrt, Mrp, Mtp, dep, lat, lon, ctm, hdr
        for deriv_par in self.par_list:
            synt_dev_fn = synt_id + "." + deriv_par
            win_obj.datalist[deriv_par] = read(synt_dev_fn)[0]
            win_obj.tag[deriv_par] = tag

        # station information
        win_obj.longitude = win_obj.datalist['synt'].stats.sac['stlo']
        win_obj.latitude = win_obj.datalist['synt'].stats.sac['stla']
        # specify metadata infor
        win_obj.source = "sac"

    def load_data_from_asdf(self, win, asdf_ds, obsd_tag=None, synt_tag=None,
                            station_dict=None):
        """
        load data from asdf file

        :return:
        """
        # trace
        win.datalist = dict()
        win.tag = dict()
        win.datalist['obsd'], win.tag['obsd'] = \
            self.get_trace_from_asdf(win.obsd_id, asdf_ds['obsd'], obsd_tag)
        win.datalist['synt'], win.tag['synt'] = \
            self.get_trace_from_asdf(win.synt_id, asdf_ds['synt'], synt_tag)
        for deriv_par in self.par_list:
            win.datalist[deriv_par], win.tag[deriv_par] = \
                self.get_trace_from_asdf(win.synt_id, asdf_ds[deriv_par],
                                         synt_tag)

        win.station = win.datalist['obsd'].stats.station
        win.network = win.datalist['obsd'].stats.network
        win.component = win.datalist['obsd'].stats.channel
        win.location = win.datalist['obsd'].stats.location

        # load station information
        if station_dict is None:
            win.latitude, win.longitude = \
                self.get_station_loc_from_asdf(win.obsd_id, asdf_ds['synt'])
        else:
            key = "_".join([win.network, win.station])
            win.latitude = station_dict[key][0]
            win.longitude = station_dict[key][1]

        # specify metadata infor
        win.source = "asdf"

    @staticmethod
    def get_station_loc_from_asdf(station_string, asdf_handle):
        """
        Used to extrace station location information from stationxml in asdf
        """
        station_string = os.path.basename(station_string)
        station_info = station_string.split(".")
        if len(station_info) == 4:
            [network, station, _, _] = station_info
        else:
            raise ValueError("Station string should be 'NW.STA.LOC.COMP'."
                             "But current is not correct:%s" % station_info)

        if len(network) >= 3 or len(station) <= 2:
            raise ValueError("Station string should be 'NW.STA.LOC.COMP'"
                             "But current is: %s" % station_info +
                             "You may place the network and station name in"
                             "the wrong order")

        station_name = network + "_" + station
        # get the tag
        st = getattr(asdf_handle.waveforms, station_name)
        if "coordinates" in dir(st):
            latitude = st.coordinates["latitude"]
            longitude = st.coordinates["longitude"]
        elif "StationXML" in dir(st):
            inv = getattr(st, 'StationXML')
            latitude = float(inv[0][0].latitude)
            longitude = float(inv[0][0].longitude)
        else:
            raise ValueError("Can't extract station location")
        return latitude, longitude

    @staticmethod
    def get_trace_from_asdf(station_string, asdf_handle, tag):
        """
        Used to extract a specific trace out of an asdf file.

        :param station_string:
        :param asdf_handle:
        :param tag:
        :return:
        """
        # just in case people put the whole path, which has no meaning
        # if asy14ydf is used
        station_string = os.path.basename(station_string)
        station_info = station_string.split(".")
        if len(station_info) == 4:
            [network, station, loc, channel] = station_info
        else:
            raise ValueError("Station string should be 'NW.STA.LOC.COMP'."
                             "But current is not correct:%s" % station_info)

        if len(network) >= 3 and len(station) <= 2:
            raise ValueError("Station string should be 'NW.STA.LOC.COMP'"
                             "But current is: %s" % station_info +
                             "You may place the network and station name in"
                             "the wrong order")

        station_name = network + "_" + station
        # get the tag
        st = getattr(asdf_handle.waveforms, station_name)
        tag_list = st.get_waveform_tags()
        if tag is None or tag == "":
            if len(tag_list) != 1:
                raise ValueError("More that 1 data tags in obsd asdf file. "
                                 "For this case, you need specify obsd_tag:%s"
                                 % tag_list)
            stream = getattr(st, tag_list[0])
            tag = tag_list[0]
        else:
            stream = getattr(st, tag)
        tr = stream.select(network=network, station=station, location=loc,
                           channel=channel)[0]
        return tr.copy(), tag

    @staticmethod
    def load_station_from_text(stationfile):
        station_dict = {}
        with open(stationfile, 'r') as f:
            content = f.readlines()
            content = [line.rstrip('\n') for line in content]
            for line in content:
                info = line.split()
                key = "_".join([info[1], info[0]])
                station_dict[key] = \
                    [float(info[2]), float(info[3]), float(info[4])]
        return station_dict

    def write_new_syn_file(self, file_format="sac", outputdir=".",
                           eventname=None, suffix=None):
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

        if file_format.upper() == "SAC":
            for tag, win_array in new_synt_dict.iteritems():
                if eventname is None:
                    targetdir = os.path.join(outputdir, tag)
                else:
                    targetdir = os.path.join(outputdir, "%s_%s"
                                             % (eventname, tag))
                if not os.path.exists(targetdir):
                    os.makedirs(targetdir)
                for window in win_array:
                    sta = window.station
                    nw = window.network
                    component = window.component
                    location = window.location
                    filename = "%s.%s.%s.%s.sac" \
                               % (sta, nw, location, component)
                    outputfn = os.path.join(targetdir, filename)
                    new_synt = window.datalist['new_synt']
                    new_synt.write(outputfn, format='SAC')
        elif file_format.upper() == "ASDF":
            for tag, win_array in new_synt_dict.iteritems():

                if eventname is None:
                    _event_str = "new_synt"
                else:
                    _event_str = "%s.new_synt" % eventname
                if suffix is None:
                    _suffix_str = ""
                else:
                    _suffix_str = "%s" % suffix
                outputfn = os.path.join(
                    outputdir, "%s.%s.%s.h5" 
                    % (_event_str, _suffix_str, tag))

                if os.path.exists(outputfn):
                    print "Output file exists, removed:%s" % outputfn
                    os.remove(outputfn)
                ds = ASDFDataSet(outputfn)
                for window in win_array:
                    ds.add_waveforms(window.datalist['new_synt'], tag=tag)
                # add stationxml
                _staxml_asdf = self.asdf_file_dict['synt']
                ds_sta = ASDFDataSet(_staxml_asdf)
                self.add_staxml_from_other_asdf(ds, ds_sta)
                ds.flush()
        else:
            raise NotImplementedError

    @staticmethod
    def add_staxml_from_other_asdf(ds, ds_sta):
        sta_tag_list = dir(ds.waveforms)
        for sta_tag in sta_tag_list:
            _sta_data = getattr(ds_sta.waveforms, sta_tag)
            staxml = _sta_data.StationXML
            ds.add_stationxml(staxml)

    def print_summary(self):
        """
        Print summary of data container

        :return:
        """
        nfiles_r = 0
        nfiles_t = 0
        nfiles_z = 0
        nwins_r = 0
        nwins_t = 0
        nwins_z = 0
        for window in self.window:
            if window.component[2:3] == "R":
                nfiles_r += 1
                nwins_r += window.num_wins
            elif window.component[2:3] == "T":
                nfiles_t += 1
                nwins_t += window.num_wins
            elif window.component[2:3] == "Z":
                nfiles_z += 1
                nwins_z += window.num_wins
            else:
                raise ValueError(
                    "Unrecognized compoent in windows: %s.%s.%s"
                    % (window.station, window.network, window.component))

        logger.info("="*10 + "  Data Summary  " + "="*10)
        logger.info("Number of Deriv synt: %d" % len(self.par_list))
        logger.info("   Par: [%s]" % (', '.join(self.par_list)))
        logger.info("Number of data pairs: %d" % self.nfiles)
        logger.info("   [Z, R, T] = [%d, %d, %d]"
                    % (nfiles_z, nfiles_r, nfiles_t))
        logger.info("Number of windows: %d" % self.nwins)
        logger.info("   [Z, R, T] = [%d, %d, %d]"
                    % (nwins_z, nwins_r, nwins_t))
        logger.info("Loading takes %6.2f seconds" % self.elapse_time)
