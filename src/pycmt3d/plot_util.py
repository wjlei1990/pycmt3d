#!/usr/bin/env python
# -*- coding: utf-8 -*-
from obspy.core.util.geodetics.base import gps2DistAzimuth
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import math
from obspy.imaging.beachball import Beach
import matplotlib.gridspec as gridspec

# earth half circle
EARTH_HC, _, _ = gps2DistAzimuth(0, 0, 0, 180)


class PlotUtil(object):

    def __init__(self, data_container=None, cmtsource=None, config=None,
                 nregions=12, new_cmtsource=None, bootstrap_mean=None,
                 bootstrap_std=None, var_reduction=0.0, mode="regional"):
        self.data_container = data_container
        self.cmtsource = cmtsource
        self.window = data_container.window
        self.config = config
        self.nregions = nregions

        self.new_cmtsource = new_cmtsource
        self.bootstrap_mean = bootstrap_mean
        self.bootstrap_std = bootstrap_std
        self.var_reduction = var_reduction

        self.moment_tensor = [cmtsource.m_rr, cmtsource.m_tt, cmtsource.m_pp,
                              cmtsource.m_rt, cmtsource.m_rp, cmtsource.m_tp]

        if mode.lower() not in ["global", "regional"]:
            raise ValueError("Plot mode: 1) global; 2) regional")
        self.mode = mode.lower()

        self.sta_lat = []
        self.sta_lon = []
        self.sta_theta = []
        self.sta_dist = []
        self.sta_azi = []
        self.prepare_array()

    def prepare_array(self):
        # station
        sta_dict = {}
        for window in self.window:
            key = window.network + "." + window.station
            if key not in sta_dict.keys():
                sta_dict[key] = [window.latitude, window.longitude]

        for key, sta in sta_dict.iteritems():
            self.sta_lat.append(sta[0])
            self.sta_lon.append(sta[1])

        self.calc_sta_dist_azi()

    def calc_sta_dist_azi(self):
        for i in range(len(self.sta_lat)):
            dist, az, baz = gps2DistAzimuth(self.cmtsource.latitude,
                                            self.cmtsource.longitude,
                                            self.sta_lat[i], self.sta_lon[i])
            self.sta_azi.append(az)
            self.sta_theta.append(az/180.0*np.pi)
            if self.mode == "regional":
                # if regional, then use original distance(in km)
                self.sta_dist.append(dist/1000.0)
            elif self.mode == "global":
                # if global, then use degree as unit
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
        bins = [delta*i for i in range(self.nregions)]

        naz_wins = np.zeros(self.nregions)
        for azimuth in azimuth_array:
            bin_idx = self.get_azimuth_bin_number(azimuth[0])
            naz_wins[bin_idx] += azimuth[1]
        return bins, naz_wins

    @staticmethod
    def plot_si_bb(ax, cmt):
        # get moment tensor
        mt = [cmt.m_rr, cmt.m_tt, cmt.m_pp, cmt.m_rt, cmt.m_rp, cmt.m_tp]
        # plot beach ball
        b = Beach(mt, linewidth=1, xy=(0, 0.6), width=1, size=2,
                  facecolor='r')
        ax.add_collection(b)
        # set axis
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1.5])
        ax.set_aspect('equal')
        # magnitude
        text = "Mw=%4.3f" % cmt.moment_magnitude
        plt.text(-0.9, -0.3, text, fontsize=7)
        # lat and lon
        text = "lat=%6.3f$^\circ$; lon=%6.3f$^\circ$" \
               % (cmt.latitude, cmt.longitude)
        plt.text(-0.9, -0.5, text, fontsize=7)
        # depth
        text = "dep=%6.3f km;" % (cmt.depth_in_m/1000.0)
        plt.text(-0.9, -0.7, text, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        # title
        text = "Init CMT"
        plt.text(-0.9, 1.3, text, fontsize=7)

    @staticmethod
    def plot_si_bb_comp(ax, cmt, cmt_init, tag):
        # get moment tensor
        mt = [cmt.m_rr, cmt.m_tt, cmt.m_pp, cmt.m_rt, cmt.m_rp, cmt.m_tp]
        # plot beach ball
        b = Beach(mt, linewidth=1, xy=(0, 0.6), width=1, size=2,
                  facecolor='r')
        ax.add_collection(b)
        # set axis
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1.5])
        ax.set_aspect('equal')
        # magnitude
        text = r"$\Delta$Mw=%4.3f" % (
            cmt.moment_magnitude-cmt_init.moment_magnitude)
        plt.text(-0.9, -0.3, text, fontsize=7)
        # lat and lon
        text = r"$\Delta$lat=%6.3f$^\circ$; $\Delta$lon=%6.3f$^\circ$" \
               % (cmt.latitude-cmt_init.latitude,
                  cmt.longitude-cmt_init.longitude)
        plt.text(-0.9, -0.5, text, fontsize=7)
        # depth
        text = r"$\Delta$dep=%6.3f km;" % (
            (cmt.depth_in_m-cmt_init.depth_in_m)/1000.0)
        plt.text(-0.9, -0.7, text, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        text = tag
        plt.text(-0.9, 1.3, text, fontsize=7)

    def plot_table(self):
        par_mean = self.bootstrap_mean
        par_std = self.bootstrap_std
        std_over_mean = np.zeros(par_mean.shape)
        for _i in range(par_mean.shape[0]):
            if par_mean[_i] != 0:
                std_over_mean[_i] = par_std[_i]/np.abs(par_mean[_i])
            else:
                std_over_mean[_i] = 0.0
        fontsize = 9
        incre = 0.06
        pos = 1.00
        format1 = "%15.4e  %15.4e  %15.4e  %15.4e   %10.2f%%"
        format2 = "%16.3f  %16.3f  %20.3f  %20.3f   %15.2f%%"

        text = "Number of stations: %d          Number of widnows: %d" \
               % (len(self.sta_lat), self.data_container.nwins)
        plt.text(0, pos, text, fontsize=fontsize)

        pos -= incre
        text = "Number of Parameter:%3d         Zero-Trace:%6s" \
               "                  Double-couple:%6s " \
               % (self.config.npar, self.config.zero_trace,
                  self.config.double_couple)
        plt.text(0, pos, text, fontsize=fontsize)

        pos -= incre
        text = "Station Correction:%6s         Norm_window:%6s" \
               "              Norm_Category:%6s" \
               % (self.config.station_correction, self.config.normalize_window,
                  self.config.normalize_category)
        plt.text(0, pos, text, fontsize=fontsize)

        pos -= incre
        energy_change = \
            (self.new_cmtsource.M0 - self.cmtsource.M0) / self.cmtsource.M0
        text = "Inversion Damping:%6.3f       Energy Change: %6.2f%%" \
               "       Variance Reduction: %6.2f%%" \
               % (self.config.lamda_damping, energy_change*100,
                  self.var_reduction*100)
        plt.text(0, pos, text, fontsize=fontsize)

        pos -= incre
        text = "*"*20 + "   Summary Table    " + "="*20
        plt.text(0, pos, text, fontsize=fontsize)

        pos -= incre
        text = " PAR         Old_CMT          New_CMT       Bootstrap_Mean" \
               "      Bootstrap_STD     STD/Mean"
        plt.text(0, pos, text, fontsize=fontsize)

        pos -= incre
        text = "Mrr:" + format1 % (
                    self.cmtsource.m_rr, self.new_cmtsource.m_rr,
                    par_mean[0], par_std[0], std_over_mean[0] * 100)
        plt.text(0, pos, text, fontsize=fontsize)
        text = "Mtt:" + format1 % (
                    self.cmtsource.m_tt, self.new_cmtsource.m_tt,
                    par_mean[1], par_std[1], std_over_mean[1] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize)
        text = "Mpp:" + format1 % (
                    self.cmtsource.m_pp, self.new_cmtsource.m_pp,
                    par_mean[2], par_std[2], std_over_mean[2] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize)
        text = "Mrt:" + format1 % (
                    self.cmtsource.m_rt, self.new_cmtsource.m_rt,
                    par_mean[3], par_std[3], std_over_mean[3] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize)
        text = "Mrp:" + format1 % (
                    self.cmtsource.m_rp, self.new_cmtsource.m_rp,
                    par_mean[4], par_std[4], std_over_mean[4] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize)
        text = "Mtp:" + format1 % (
                    self.cmtsource.m_tp, self.new_cmtsource.m_tp,
                    par_mean[5], par_std[5], std_over_mean[5] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize)
        text = "DEP:  " + format2 % (
               self.cmtsource.depth_in_m / 1000.0,
               self.new_cmtsource.depth_in_m / 1000.0,
               par_mean[6], par_std[6], std_over_mean[6] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize)
        text = "LON:" + format2 % (
               self.cmtsource.longitude, self.new_cmtsource.longitude,
               par_mean[7], par_std[7], std_over_mean[7] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize)
        text = "LAT:  " + format2 % (
               self.cmtsource.latitude, self.new_cmtsource.latitude,
               par_mean[8], par_std[8], std_over_mean[8] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize)
        text = "CMT:   " + format2 % (
               self.cmtsource.time_shift, self.new_cmtsource.time_shift,
               par_mean[9], par_std[9], std_over_mean[9] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize)
        text = "HDR:   " + format2 % (
               self.cmtsource.half_duration, self.new_cmtsource.half_duration,
               par_mean[10], par_std[10], std_over_mean[10] * 100)
        pos -= incre
        plt.text(0, pos, text, fontsize=fontsize)
        plt.axis('off')

    def plot_global_map(self):
        """
        Plot global map of event and stations
        """
        # ax = plt.subplot(211)
        plt.title(self.cmtsource.eventname)
        m = Basemap(projection='cyl', lon_0=0.0, lat_0=0.0,
                    resolution='c')
        m.drawcoastlines()
        m.fillcontinents()
        m.drawparallels(np.arange(-90., 120., 30.))
        m.drawmeridians(np.arange(0., 420., 60.))
        m.drawmapboundary()

        x, y = m(self.sta_lon, self.sta_lat)
        m.scatter(x, y, 30, color="r", marker="^", edgecolor="k",
                  linewidth='0.3', zorder=3)

        cmt_lat = self.cmtsource.latitude
        cmt_lon = self.cmtsource.longitude
        focmecs = self.moment_tensor
        ax = plt.gca()
        bb = Beach(focmecs, xy=(cmt_lon, cmt_lat), width=20, linewidth=1,
                   alpha=1.0)
        bb.set_zorder(10)
        ax.add_collection(bb)

    def plot_sta_dist_azi(self):
        plt.title("Station Dist and Azi", fontsize=10)
        ax = plt.gca()
        c = plt.scatter(self.sta_theta, self.sta_dist, marker=u'^', c='r',
                        s=20, edgecolor='k', linewidth='0.3')
        c.set_alpha(0.75)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=6)
        if self.mode == "regional":
            ax.set_rmax(1.10 * max(self.sta_dist))
        elif self.mode == "global":
            ax.set_rmax(1.0)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

    def plot_sta_azi(self):
        # set plt.subplot(***, polar=True)
        plt.title("Station Azimuth", fontsize=10)
        azimuth_array = []
        for azi in self.sta_azi:
            azimuth_array.append([azi, 1])
        bins, naz = self.calculate_azimuth_bin(azimuth_array)
        norm_factor = np.max(naz)

        bars = plt.bar(bins, naz, width=(bins[1]-bins[0]), bottom=0.0)
        for r, bar in zip(naz, bars):
            bar.set_facecolor(plt.cm.jet(r/norm_factor))
            bar.set_alpha(0.5)
            bar.set_linewidth(0.3)
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=6)
        ax = plt.gca()
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

    def plot_win_azi(self):
        # set plt.subplot(***, polar=True)
        plt.title("Window Azimuth", fontsize=10)
        win_azi = []
        for window in self.window:
            win_azi.append([window.azimuth, window.num_wins])
        bins, naz = self.calculate_azimuth_bin(win_azi)
        norm_factor = np.max(naz)

        bars = plt.bar(bins, naz, width=(bins[1]-bins[0]), bottom=0.0)
        for r, bar in zip(naz, bars):
            bar.set_facecolor(plt.cm.jet(r/norm_factor))
            bar.set_alpha(0.5)
            bar.set_linewidth(0.3)
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=6)
        ax = plt.gca()
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

    def plot_all_stat(self, figurename=None):
        plt.figure(figsize=(10, 7), facecolor='w', edgecolor='k')
        g = gridspec.GridSpec(2, 3)
        plt.subplot(g[0, :-1])
        self.plot_global_map()
        plt.subplot(g[1, 0],  polar=True)
        self.plot_sta_dist_azi()
        plt.subplot(g[1, 1], polar=True)
        self.plot_sta_azi()
        plt.subplot(g[1, 2], polar=True)
        self.plot_win_azi()
        ax = plt.subplot(g[0, 2])
        self.plot_si_bb(ax, self.cmtsource)
        if figurename is None:
            plt.show()
        else:
            plt.savefig(figurename)

    def plot_inversion_summary(self, figurename=None):
        if self.new_cmtsource is None:
            raise ValueError("No new cmtsource...Can't plot summary")
        plt.figure(figsize=(10, 11), facecolor='w', edgecolor='k')
        g = gridspec.GridSpec(3, 3)
        plt.subplot(g[0, :-1])
        self.plot_global_map()
        plt.subplot(g[1, 0],  polar=True)
        self.plot_sta_dist_azi()
        plt.subplot(g[1, 1], polar=True)
        self.plot_sta_azi()
        plt.subplot(g[1, 2], polar=True)
        self.plot_win_azi()
        ax = plt.subplot(g[0, 2])
        self.plot_si_bb(ax, self.cmtsource)
        ax = plt.subplot(g[2, 2])
        self.plot_si_bb_comp(ax, self.new_cmtsource, self.cmtsource,
                             "Inversion")
        plt.subplot(g[2, :-1])
        self.plot_table()
        if figurename is None:
            plt.show()
        else:
            plt.savefig(figurename)
