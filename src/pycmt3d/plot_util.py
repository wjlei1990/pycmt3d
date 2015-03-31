#!/usr/bin/env python
# -*- coding: utf-8 -*-

import obspy
from obspy.core.util.geodetics.base import gps2DistAzimuth
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import math
from obspy.imaging.beachball import Beach

# earth half circle
EARTH_HC, _, _ = gps2DistAzimuth(0,0,0, 180)


class PlotUtil(object):

    def __init__(self, data_container=None, cmtsource=None, nregions=12):
        self.data_container = data_container
        self.cmtsource = cmtsource
        self.window = data_container.window
        self.nregions = nregions

        self.moment_tensor = [cmtsource.m_rr, cmtsource.m_tt, cmtsource.m_pp, cmtsource.m_rt, cmtsource.m_rp,
                              cmtsource.m_tp]

        self.prepare_array()

    def prepare_array(self):
        # station
        sta_dict = {}
        for window in self.window:
            key = window.network + "." + window.component
            if key not in sta_dict.keys():
                sta_dict[key] = [window.latitude, window.longitude]

        self.sta_lat = []
        self.sta_lon = []
        for key, sta in sta_dict.iteritems():
            self.sta_lat.append(sta[0])
            if sta[1] < 0:
                self.sta_lon.append(sta[1]+360.0)
            else:
                self.sta_lon.append(sta[1])

        self.calc_sta_dist_azi()

    def calc_sta_dist_azi(self):
        self.sta_theta = []
        self.sta_dist = []
        self.sta_azi = []
        for i in range(len(self.sta_lat)):
            dist, az, baz = gps2DistAzimuth(self.cmtsource.latitude, self.cmtsource.longitude,
                                            self.sta_lat[i], self.sta_lon[i])
            self.sta_azi.append(az)
            self.sta_theta.append(az/180.0*np.pi)
            self.sta_dist.append(dist/EARTH_HC)

    def get_azimuth_bin_number(self, azimuth):
        """
        Calculate the bin number of a given azimuth

        :param azimuth: test test test
        :return:
        """
        # the azimth ranges from [0,360]
        # so a little modification here
        daz = 360.0 / self.nregions
        k = int(math.floor(azimuth / daz))
        if k < 0 or k > self.nregions:
            if azimuth - 360.0 < 0.0001:
                k = self.nregions - 1
            else:
                raise ValueError('Error bining azimuth')
        return k

    def calculate_azimuth_bin(self, azimuth_array):
        """
        Calculate the azimuth and sort them into bins

        :return:
        """
        delta = 2*np.pi/self.nregions
        bins = [ delta*i for i in range(self.nregions)]

        naz_wins = np.zeros(self.nregions)
        for azimuth in azimuth_array:
            bin_idx = self.get_azimuth_bin_number(azimuth[0])
            naz_wins[bin_idx] += azimuth[1]
        return bins, naz_wins

    def plot_global_map(self):
        """
        Plot global map of event and stations
        """
        #ax = plt.subplot(211)
        plt.title(self.cmtsource.eventname)
        m = Basemap(projection='cyl', lon_0=142.36929, lat_0=0.0,
                    resolution='c')
        m.drawcoastlines()
        m.fillcontinents()
        m.drawparallels(np.arange(-90., 120., 30.))
        m.drawmeridians(np.arange(0., 420., 60.))
        m.drawmapboundary()

        x, y = m(self.sta_lon, self.sta_lat)
        m.scatter(x, y, 30, color="r", marker="^", edgecolor="k", linewidth='0.3', zorder=3)

        cmt_lat = self.cmtsource.latitude
        cmt_lon = self.cmtsource.longitude
        if cmt_lon < 0:
            cmt_lon += 360
        focmecs = self.moment_tensor
        ax = plt.gca()
        bb = Beach(focmecs, xy=(cmt_lon, cmt_lat), width=10, linewidth=1)
        bb.set_zorder(10)
        ax.add_collection(bb)

    def plot_sta_dist_azi(self):
        print "dd:", self.sta_theta
        print "dd:", self.sta_dist
        print "dd:", self.sta_azi
        c = plt.scatter(self.sta_theta, self.sta_dist, marker=u'^', c='r', s=50, edgecolor='k', linewidth='0.3')
        c.set_alpha(0.75)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=6)

    def plot_sta_azi(self):
        # set plt.subplot(***, polar=True)
        azimuth_array = []
        for azi in self.sta_azi:
            azimuth_array.append([azi, 1])
        bins, naz = self.calculate_azimuth_bin(azimuth_array)

        bars = plt.bar(bins, naz, width=(bins[1]-bins[0]), bottom=0.0)
        for r, bar in zip(naz, bars):
            bar.set_facecolor(plt.cm.jet(r/16.))
            bar.set_alpha(0.5)
            bar.set_linewidth(0.3)
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=6)

    def plot_win_azi(self):
        # set plt.subplot(***, polar=True)
        win_azi = []
        for window in self.window:
            win_azi.append([window.azimuth, window.num_wins])
        bins, naz = self.calculate_azimuth_bin(win_azi)

        bars = plt.bar(bins, naz, width=(bins[1]-bins[0]), bottom=0.0)
        for r, bar in zip(naz, bars):
            bar.set_facecolor(plt.cm.jet(r/16.))
            bar.set_alpha(0.5)
            bar.set_linewidth(0.3)
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=6)

    def plot_all_stat(self):
        plt.figure()
        ax = plt.subplot(211)
        self.plot_global_map()
        ax = plt.subplot(234,  polar=True)
        self.plot_sta_dist_azi()
        ax = plt.subplot(235, polar=True)
        self.plot_sta_azi()
        ax = plt.subplot(236, polar=True)
        self.plot_win_azi()
        plt.show()



