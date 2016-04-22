#!/usr/bin/env python
# -*- coding: utf-8 -*-

from source import CMTSource
import const
from __init__ import logger
import util
from util import construct_taper
from plot_util import PlotUtil
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import math
from obspy.core.util.geodetics import gps2DistAzimuth
import matplotlib.gridspec as gridspec
from spaceweight import SphereAziBin, Point


class Weight(object):
    """
    Class that handles the solver part of source inversion

    :param cmtsource: earthquake source
    :type cmtsource: :class:`pycmt3d.CMTSource`
    :param data_container: all data and window
    :type data_container: :class:`pycmt3d.DataContainer`
    :param config: configuration for source inversion
    :type config: :class:`pycmt3d.Config`
    """
    def __init__(self, cmtsource, data_container, metas, config):
        self.cmtsource = cmtsource
        self.data_container = data_container
        self.metas = metas
        self.config = config

    def setup_weight(self, weight_mode="num_wins"):
        """
        Use Window information to setup weight.

        :returns:
        """
        logger.info("*" * 15)
        logger.info("Start weighting...")
        if self.config.weight_data:
            self.prepare_for_weighting()
            azi_weight = self.setup_weight_for_azimuth()
            loc_weight = self.setup_weight_for_location()

            if self.config.normalize_by_category:
                self.setup_weight_for_category(window)

            if self.config.normalize_by_energy:
                window.weight = window.weight / window.obsd_energy

            # normalization of data weights
            self.normalize_weight()

        # prepare the weight array(for each window)
        self.weight_array = np.zeros([self.data_container.nwins])
        _idx = 0
        for window in self.traces:
            for win_idx in range(window.num_wins):
                self.weight_array[_idx] = window.weight[win_idx]
                _idx += 1

    def setup_weight_for_location(self, window):
        """
        setup weight from location information, including distance,
        component and azimuth

        :param window:
        :param naz_bin:
        :param naz_bin_all:
        :return:
        """
        idx_naz = self.get_azimuth_bin_number(window.azimuth)
        if self.config.normalize_category:
            tag = window.tag['obsd']
            naz = naz_bin[tag][idx_naz]
        else:
            naz = naz_bin_all[idx_naz]
        logger.debug("%s.%s.%s, num_win, dist, naz: %d, %.2f, %d" % (
                window.station, window.network, window.component,
                window.num_wins, window.dist_in_km, naz))

        if self.config.normalize_window:
            mode = "uniform"
        else:
            # if the weight is not normalized by energy,
            # then use the old weighting method(exponential)
            mode = "exponential"
            # weighting on compoent, distance and azimuth
        window.weight = \
            window.weight * self.config.weight_function(
                window.component, window.dist_in_km, naz, window.num_wins,
                dist_weight_mode=mode)

    def setup_weight_for_category(self, window):
        """
        Setup weight for each category if config.normalize_category
        window_weight = window_weight / N_windows_in_category

        :param window:
        :return:
        """
        if self.config.normalize_category:
            tag = window.tag['obsd']
            num_cat = self.bin_category[tag]
            window.weight = window.weight/num_cat

    def prepare_for_weighting(self):
        """
        Prepare necessary information for weighting, e.x.,
        calculating azimuth, distance and energty of a window.
        Also, based on the tags, sort window into different categories.

        :return:
        """
        for trace in self.traces:
            trace.event_latitude = self.cmtsource.latitude
            trace.event_longitude = self.cmtsource.longitude

        self.naz_files, self.naz_wins = self.calculate_azimuth_bin()
        # add all category together
        # if not weighted by category, then use total number
        self.naz_files_all = np.zeros(const.NREGIONS)
        self.naz_wins_all = np.zeros(const.NREGIONS)
        for key in self.naz_files.keys():
            self.naz_files_all += self.naz_files[key]
            self.naz_wins_all += self.naz_wins[key]
            logger.info("Category: %s" % key)
            logger.info("Azimuth file bin: [%s]"
                        % (', '.join(map(str, self.naz_files[key]))))
            logger.info("Azimuth win bin: [%s]"
                        % (', '.join(map(str, self.naz_wins[key]))))

        # stat different category
        bin_category = {}
        for window in self.traces:
            tag = window.tag['obsd']
            if tag in bin_category.keys():
                bin_category[tag] += window.num_wins
            else:
                bin_category[tag] = window.num_wins
        self.bin_category = bin_category

    @staticmethod
    def get_azimuth_bin_number(azimuth):
        """
        Calculate the bin number of a given azimuth

        :param azimuth: test test test
        :return:
        """
        # the azimth ranges from [0,360]
        # so a little modification here
        daz = 360.0 / const.NREGIONS
        k = int(math.floor(azimuth / daz))
        if k < 0 or k > const.NREGIONS:
            if azimuth - 360.0 < 0.0001:
                k = const.NREGIONS - 1
            else:
                raise ValueError('Error bining azimuth')
        return k

    def calculate_azimuth_bin(self):
        """
        Calculate the azimuth and sort them into bins

        :return:
        """
        naz_traces = {}
        naz_wins = {}
        for trace in self.traces:
            tag = trace.full_tag['obsd']
            bin_idx = self.get_azimuth_bin_number(window.azimuth)
            if tag not in naz_files.keys():
                naz_files[tag] = np.zeros(const.NREGIONS)
                naz_wins[tag] = np.zeros(const.NREGIONS)
            naz_files[tag][bin_idx] += 1
            naz_wins[tag][bin_idx] += window.num_wins
        return naz_files, naz_wins

    def normalize_weight(self):
        """
        Normalize the weighting and make the maximum to 1

        :return:
        """
        max_weight = 0.0
        for window in self.traces:
            max_temp = np.max(window.weight)
            if max_temp > max_weight:
                max_weight = max_temp

        logger.debug("Global Max Weight: %f" % max_weight)

        for window in self.traces:
            logger.debug("%s.%s.%s, weight: [%s]"
                         % (window.network, window.station, window.component,
                            ', '.join(map(self._float_to_str, window.weight))))
            window.weight /= max_weight
            logger.debug("Updated, weight: [%s]"
                         % (', '.join(map(self._float_to_str, window.weight))))

    def get_station_info(self, datalist):
        """
        Using the event location and station information to
        calculate azimuth and distance
        !!! Obsolete, not used any more !!!

        :param datalist: data dictionary(referred to pycmt3d.Window.datalist)
        :return:
        """
        # this might be related to datafile type(sac, mseed or asdf)
        event_lat = self.cmtsource.latitude
        event_lon = self.cmtsource.longitude
        # station location from synthetic file
        sta_lat = datalist['synt'].stats.sac['stla']
        sta_lon = datalist['synt'].stats.sac['stlo']
        dist_in_m, az, baz = \
            gps2DistAzimuth(event_lat, event_lon, sta_lat, sta_lon)
        return [dist_in_m / 1000.0, az]
