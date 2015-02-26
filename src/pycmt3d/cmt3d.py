#!/usr/bin/env python
# -*- coding: utf-8 -*-

from window import Window
import os
import numpy as np
from source import CMTSource
from obspy import read
import math
from obspy.core.util.geodetics import gps2DistAzimuth
import const

class cmt3d(object):

    def __init__(self, cmtsource, flexwin_file, config):
        self.config = config
        self.cmtsource = cmtsource
        self.flexwin_file = flexwin_file
        self.data = []
        self.window = []
        self.num_file = 0

    def load_winfile(self):
        self.window = []
        with open(self.flexwin_file, "r") as f:
            num_file = int(f.readline().strip())
            if num_file == 0:
                return
            for idx in range(num_file):
                # keep the old format of cmt3d input
                obsd_fn = f.readline().strip()
                synt_fn = f.readline().strip()
                sta_info = os.path.basename(obsd_fn)
                [sta, nw, loc, comp, type] = sta_info.split(".")
                num_win = int(f.readline().strip())
                win_time = np.zeros((num_win,2))
                for iwin in range(num_win):
                    [left, right] = f.readline().strip()
                    win_time[iwin, 0] = left
                    win_time[iwin, 1] = right
                    win_obj = Window(sta, nw, loc, comp, win_time,
                                     obsd=obsd_fn, synt=synt_fn)
                self.window.append(win_obj)
                self.data.append(self.load_data(win_obj))
        self.num_file = len(self.window)

    def load_data(self, win_obj):
        """
        Old way of loading obsd and synt data...
        :param win_obj:
        :return:
        """
        datalist = {}
        obsd_fn = win_obj.obsd_fn
        synt_fn = win_obj.synt_fn
        npar = self.config.npar
        par_list = self.config.par_name[:npar]
        datalist['obsd'] = read(obsd_fn)[0]
        datalist['synt'] = read(synt_fn)[0]
        for i in range(npar):
            synt_dev_fn = synt_fn + "." + par_list[i]
            datalist[par_list[i]] = read(synt_dev_fn)[0]

        return datalist

    def setup_weight(self):
        """
        Use Window to setup weight
        :return:
        """
        if self.config.weight_data:
            for idx, window in enumerate(self.window):
                kcmpnm = window.component
                [dist_in_km, azimuth] = self.get_station_loc_info(self.data[idx])
                weight = self.config.weight_function(kcmpnm, azimuth, dist_in_km)
                self.window[idx].weight = weight
        else:
            for idx in range(len(self.window)):
                self.window[idx].weight = 1.0

    def get_station_info(self, datalist):
        # this might be related to datafile type(sac, mseed or asdf)
        event_lat = self.cmtsource.latitude
        event_lon = self.cmtsource.longitude
        # station location from synthetic file
        sta_lat = self.datalist['synt'].stats.sac['stla']
        sta_lon = self.datalist['synt'].stats.sac['stlo']
        dist_in_m, az, baz = gps2DistAzimuth(event_lat, event_lon, sta_lat, sta_lon)
        return [dist_in_m/1000.0, az]

    def setup_matrix(self):
        self.A = np.zeros(self.config.npar, self.config.npar)
        self.b = np.zeros(self.config.npar, 1)
        for idx, window in enumerate(self.window):
            data = self.data[idx]
            [A1, b1] = self.compute_A_b(window, data)
            self.A += A1
            self.b += b1

        # we setup the full array, but based on npar, only part of it will be used
        cmt = self.cmtsource
        self.cmt_par = np.array([cmt.m_rr, cmt.m_tt, cmt.m_pp, cmt.m_rt, cmt.m_rp, cmt.m_tp,
                               cmt.m_rp, cmt.depth_in_m/1000.0, cmt.longitude,
                               cmt.latitude, cmt.time_shift, cmt.half_duration])

    def compute_A_b(self, window, datalist):

        par_list = self.config.par_name
        npar = self.config.npar
        dcmt_par = self.config.dcmt_par
        obsd = datalist['obsd']
        synt = datalist['synt']
        npts = min(obsd.npts, synt.npts)
        for win in window.win_time:
            istart = max(math.floor(win[0]/obsd.stats.delta),1)
            iend = max(math.ceiling(win[1]/obsd.stats.delta),npts)
            if istart > iend:
                raise ValueError("Check window for %s.%s.%s.%s" %(window.station,
                            window.network, window.location, window.component))

            if self.config.station_correction:
                [nshift, cc, dlna] = self.calculate_criteria(obsd, synt, istart, iend)
                istart_d = max(1, istart + nshift)
                iend_d = min(npts, iend + nshift)
                istart_s = istart_d - nshift
                iend_s = iend_d - nshift
            else:
                istart_d = istart
                iend_d = iend
                istart_s = istart
                iend_s = iend

            dsyn = {}
            for itype in range(self.config.npar):
                type = par_list[itype]
                if itype < const.NML:
                    # check file
                    # check dt, npts
                    dt = datalist['synt'].stats.delta
                if itype < const.NM: # moment tensor
                    dsyn[type] = datalist[type].data[1:npts]/dcmt_par[itype]
                elif itype < const.NML:  # location
                    dsyn[type] = (datalist[type].data[1:npts]-datalist['synt'].data[1:npts])/dcmt_par[itype]
                elif itype == const.NML:  # time shift
                    dsyn[type] = (datalist['synt'].data[2:npts]-datalist['synt'].data[1:(npts-1)])/(dt*dcmt_par[itype])
                    dsyn[type].append(dsyn[type].data[npts-2])
            # hanning taper
            taper = self.construct_hanning_taper(istart_s, iend_s)
            # compute A and b by taking into account data weights
            for i in range(npar):
                typei = par_list[i]
                for j in range(npar):
                    typej = par_list[j]
                    A1ij = window.weight * np.sum(taper * dsyn[typei][istart_s:iend_s] * dsyn[typej][istart_s:iend_s]) * dt
                b1i = window.weight * np.sum(taper * (obsd.data[istart_d:iend_d] - synt.data[istart_s:iend_s]) *
                        dsyn[typei][istart_s:iend_s])




    def invert_cmt(self):
        npar = self.config.npar
        old_par = self.cmt_par[0:npar]/self.config.scale_par[0:npar]

        # scale the A and b matrix
        max_row = np.amax(self.A, axis=1)
        A = self.A/max_row
        b = self.b/max_row

        # setup inversion schema
        if self.config.double_couple:
            linear_inversion = False
            na = npar + 2
        elif self.config.zero_trace:
            linear_inversion = True
            na = npar + 1
        else:
            linear_inversion = True
            na = npar

        # add damping
        trace = np.matrix.trace(A)
        damp_matrix = np.zeros([npar, npar])
        np.fill_diagonal(damp_matrix, trace*self.config.lamda_damping)

        # setup new matrix based on constraints
        AA = np.zeros([na, na])
        bb = np.zeros([na, 1])
        if linear_inversion:
            # if invert for moment tensor with zero-trace constraints
            print "Linear Inversion"
            AA[0:npar, 0:npar] = A
            bb[0:npar, 1] = b
            if self.config.zero_trace:
                bb[na-1] = - np.sum(old_par[0:3])
                AA[0:6, na-1] = np.array([1, 1, 1, 0, 0, 0])
                AA[na-1, 0:6] = np.array([1, 1, 1, 0, 0, 0])
            # use linear solver
            try:
                dm = np.linalg.solve(AA, bb)
            except:
                print "Matrix is singular...LinearAlgError"
                raise ValueError("Check Matrix Singularity")
            new_par = old_par[0:npar] + dm[0:npar]

        else:
            # if invert for moment tensor with double couple constraints
            # setup starting solution, solve directly for moment instead
            # of dm, exact implementation of (A16)
            print "No-linear Inversion"
            mstart = np.copy(old_par)
            m1 = np.copy(mstart)
            lam = np.zeros(2)

            # nolinear solver. Maybe there are already exist code.
            # check later
            for iter in range(NMAX_NL_ITER):
                self.get_f_df()
                bb = - bb
                xout = self.gaussian_elimination()
                m1 = m1 + xout
                lam = lam + xout[npar:na]
            dm = m1 - mstart
            new_par = m1

        new_cmt_par = np.copy(self.cmt_par)
        new_cmt_par = new_par * self.config.scale_par

    def get_f_df(self):
        pass

    def gaussian_elimination(self):
        pass

    def variance_reduction(self):
        """
        Calculate variance recduction after source apdated
        :return:
        """
        fh = open("cmt3d_flexwin.out", "w")
        fh.write("%d\n" %self.num_file)

        nwint = 0
        var_all = 0.0
        var_all_new = 0.0
        for window in self.window:
            obsd_fn = window.obsd_fn
            synt_fn = window.synt_fn

            obsd = read(obsd_fn)[0]
            synt = read(synt_fn)[0]
            new_synt = self.compute_new_syn(obsd, synt, synt_fn, dm)
            for win_time in window.win_time:
                nwint += 1
                tstart = win_time[0]
                tend = win_time[1]
                idx_start = max(math.floor(tstart/obsd.stats.delta),1)
                idx_end = min(math.ceil(tend/obsd.stats.delta), obsd.stats.npts)

                if self.config.station_correction:
                    self.calculate_criteria()
                    self.calculate_criteria()
                else:
                    istart_d = idx_start
                    istart = idx_start
                    iend_d = idx_end
                    iend = idx_end
                    istart_dn = idx_start
                    istart_n = idx_start
                    iend_dn = idx_end
                    iend_n = idx_end

                taper = self.construct_hanning_taper(istart, iend)
                v1 = np.sum()
                v2 = np.sum()
                d1 = np.sum()
                d2 = np.sum()
                var_all += 0.5*v1*window.weight*obsd.stats.delta
                var_all_new += 0.5*v2*window.weight*obsd.stats.delta

                # write out
                fh.write("")
                fh.write("")
                fh.write("")

        # close output fh
        fh.close()

    def compute_new_syn(self, obsd, synt, synt_fn, dm):
        pass

    @staticmethod
    def calculate_criteria(obsd, synt, istart, iend, nshift, cc, dlna):
        pass

    @staticmethod
    def construct_hanning_taper(istart, iend):
        pass

    def calculate_variance_reduction(self):
        pass

    def source_inversion(self):
        self.load_winfile()
        self.setup_weight()
        # setup matrix based on misfit
        self.setup_matrix()
        # calculate new cmt solution
        self.invert_cmt()
        self.calculate_variance_reduction()