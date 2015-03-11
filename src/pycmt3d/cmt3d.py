#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    import numpy as np
except:
    msg = ("No module named numpy. "
           "Please install numpy first, it is needed before using pycmt3d.")
    raise ImportError(msg)

try:
    from obspy import read
except:
    msg = ("No module named obspy. "
           "Please install obspy first, it is needed before using pycmt3d.")
    raise ImportError(msg)

import math
import os
import logging
from source import CMTSource
from obspy.core.util.geodetics import gps2DistAzimuth
from obspy.signal.cross_correlation import xcorr
import const
from window import Window

class cmt3d(object):

    def __init__(self, cmtsource, flexwin_file, config):
        self.config = config
        self.cmtsource = cmtsource
        self.flexwin_file = flexwin_file
        self.window = []

    def load_winfile(self):
        """
        old way of loading flexwin inputfile
        :return:
        """
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
                num_wins = int(f.readline().strip())
                win_time = np.zeros((num_wins,2))
                for iwin in range(num_wins):
                    [left, right] = f.readline().strip().split()
                    win_time[iwin, 0] = float(left)
                    win_time[iwin, 1] = float(right)
                win_obj = Window(sta, nw, loc, comp, num_wins = num_wins, win_time = win_time,
                                  obsd=obsd_fn, synt=synt_fn)
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
        datalist = {}
        obsd_fn = win_obj.obsd_fn
        synt_fn = win_obj.synt_fn
        npar = self.config.npar
        par_list = self.config.par_name[:npar]
        datalist['obsd'] = read(obsd_fn)[0]
        datalist['synt'] = read(synt_fn)[0]
        # other synt data will be referred as key value: Mrr, Mtt, Mpp, Mrt, Mrp, Mtp, dep, lat, lon, ctm, hdr
        for i in range(npar):
            synt_dev_fn = synt_fn + "." + par_list[i]
            datalist[par_list[i]] = read(synt_dev_fn)[0]

        win_obj.datalist = datalist

    def setup_weight(self):
        """
        Use Window to setup weight
        """
        if self.config.weight_data:
            # first calculate azimuth and distance for each data pair
            self.prepare_for_weighting()
            # then calculate azimuth weighting
            naz_list = self.calculate_azimuth_bin()
            for idx, window in enumerate(self.window):
                idx_naz = self.get_azimuth_bin_number(window.azimuth)
                naz = naz_list[idx_naz]
                window.weight = self.config.weight_function(window, naz)
            # normalization of data weights
            # Attention: the code here might be tedious but I just do not know how to make it bette without changing previous codes
            self.normalize_weight()
        else:
            for idx, window in enumerate(self.window):
                # set even weigting
                window.weight = np.ones(window.num_wins)

    def prepare_for_weighting(self):
        """
        Calculate azimuth and distance and store it in the instance
        :return:
        """
        event_lat = self.cmtsource.latitude
        event_lon = self.cmtsource.longitude
        for window in self.window:
            sta_lat = window.datalist['synt'].stats.sac['stla']
            sta_lon = window.datalist['synt'].stats.sac['stlo']
            dist_in_m, az, baz = gps2DistAzimuth(event_lat, event_lon, sta_lat, sta_lon)
            window.dist_in_km = dist_in_m/1000.0
            window.az = az

    @staticmethod
    def get_azimuth_bin_number(azimuth):
        # the azimth ranges from [0,360]
        # so a little modification here
        daz = 360.0 / const.NREGIONS
        k = math.floor(azimuth / daz)
        if ( k<0 or k>const.NREGIONS):
            if abs(azimuth-360.0) < 0.0001:
                k = const.NREGIONS - 1
            else:
                raise ValueError ('Error bining azimuth')
        return k

    def calculate_azimuth_bin(self):
        naz_list = np.zeros(const.NREGIONS)
        for window in self.window:
            bin_idx = self.get_azimuth_bin_number(window.azimuth)
            naz_list[bin_idx] += window.num_wins
        return naz_list

    def normalize_weight(self):
        max_weight = 0.0
        for window in self.windows:
            max_temp = np.max(window.weight)
            if max_temp > max_weight:
                max_weight = max_temp

        for window in self.window:
            window.weight /= max_weight

    def get_station_info(self, datalist):
        # this might be related to datafile type(sac, mseed or asdf)
        event_lat = self.cmtsource.latitude
        event_lon = self.cmtsource.longitude
        # station location from synthetic file
        sta_lat = datalist['synt'].stats.sac['stla']
        sta_lon = datalist['synt'].stats.sac['stlo']
        dist_in_m, az, baz = gps2DistAzimuth(event_lat, event_lon, sta_lat, sta_lon)
        return [dist_in_m/1000.0, az]

    # Setup the matrix A and b
    # If the bootstrap is True, the matrix A and b will be assembled partly for bootstrap evalution
    def setup_matrix(self):
        self.A = np.zeros((self.config.npar, self.config.npar))
        self.b = np.zeros((self.config.npar, 1))
        A1_all = []
        b1_all = []
        random_array = []
        for window in enumerate(self.window):
            # loop over pair of data
            for win_idx in range(window.num_wins):
                # loop over each window
                [A1, b1] = self.compute_A_b(window, win_idx)
                A1_all.append(A1)
                b1_all.append(b1)

        if self.config.bootstrap == True:
            self.A_bootstrap = []
            self.b_bootstrap = []
            for i in range(self.config.bootstrap_repeat):
                random_array = np.random.randint(2, size=(self.config.npar, 1, 1))
                self.A = np.sum(random_array * A1_all, axis=0)
                self.b = np.sum(random_array * b1_all, axis=0)
                self.A_bootstrap.append(self.A)
                self.b_bootstrap.append(self.b)
        # Xin, do you have a type error here?
        elif self.config.bootstrap == False:
            self.A = np.sum(A1_all, axis=0)
            self.b = np.sum(b1_all, axis=0)

        # we setup the full array, but based on npar, only part of it will be used
        cmt = self.cmtsource
        self.cmt_par = np.array([cmt.m_rr, cmt.m_tt, cmt.m_pp, cmt.m_rt, cmt.m_rp, cmt.m_tp,
                               cmt.m_rp, cmt.depth_in_m/1000.0, cmt.longitude,
                               cmt.latitude, cmt.time_shift, cmt.half_duration])

    def compute_A_b(self, window, win_idx):

        par_list = self.config.par_name
        npar = self.config.npar
        dcmt_par = self.config.dcmt_par

        datalist = window.datalist
        obsd = datalist['obsd']
        synt = datalist['synt']
        npts = min(obsd.npts, synt.npts)
        win = [window.win_time[win_idx,0], window.win_time[win_idx,1]]

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

        #dsyn = {}
        dsyn = np.zeros((npar, npar))
        #dsyn = np.zeros(npts,dtype={'names':par_list, 'formats':['f4']*len(par_list))
        for itype in range(self.config.npar):
            type_name = par_list[itype]
            if itype < const.NML:
                # check file
                # check dt, npts
                dt_synt = datalist['synt'].stats.delta
                dt_obsd = datalist['obsd'].stats.delta
                if abs(dt_synt - dt_obsd) > 0.0001:
                    raise ValueError("Delta in synthetic and observed no the same")
                dt = dt_synt
            if itype < const.NM: # moment tensor
                dsyn[itype][0:npts] = datalist[type_name].data[1:npts]/dcmt_par[itype]
            elif itype < const.NML:  # location
                dsyn[itype][0:npts] = (datalist[type_name].data[1:npts]-datalist['synt'].data[1:npts])/dcmt_par[itype]
            elif itype == const.NML:  # time shift
                dsyn[itype][0:npts] = (datalist['synt'].data[2:npts]-datalist['synt'].data[1:(npts-1)])/(dt*dcmt_par[itype])
                dsyn[itype][npts-1] = dsyn[itype][npts-2]
                #dsyn[itype].append(dsyn[type_name].data[npts-2])

        # hanning taper
        taper = self.construct_hanning_taper(iend_s-istart_s)
        A1 = np.zeros((npar, npar))
        b1 = np.zeros((npar, 1))
        # compute A and b by taking into account data weights
        # for i in range(npar):
        #     typei = par_list[i]
        #     for j in range(npar):
        #         typej = par_list[j]
        #         A1[i][j] = window.weight * np.sum(taper * dsyn[typei][istart_s:iend_s] * dsyn[typej][istart_s:iend_s]) * dt
        #     b1i = window.weight * np.sum(taper * (obsd.data[istart_d:iend_d] - synt.data[istart_s:iend_s]) *
        #             dsyn[typei][istart_s:iend_s])
        for i in range(npar):
            A1[i] = window.weight[win_idx] * np.sum(taper * dsyn[i][istart_s:iend_s] * dsyn[:][istart_s:iend_s]) * dt
            b1 = window.weight[win_idx] * np.sum(taper * (obsd.data[istart_d:iend_d] - synt.data[istart_s:iend_s]) *
                                         dsyn[:][istart_s:iend_s])
        return [A1, b1]

    def invert_cmt(self, A, b):

        npar = self.config.npar
        old_par = self.cmt_par[0:npar]/self.config.scale_par[0:npar]

        # scale the A and b matrix
        max_row = np.amax(A, axis=1)
        A = A/max_row
        b = b/max_row

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
        A = A + damp_matrix

        # setup new matrix based on constraints
        AA = np.zeros([na, na])
        bb = np.zeros([na, 1])
        if linear_inversion:
            # if invert for moment tensor with zero-trace constraints or no constraint
            print ("Linear Inversion")
            AA[0:npar, 0:npar] = A
            bb[0:npar, 1] = b
            if self.config.zero_trace:
                bb[na-1] = - np.sum(old_par[0:3])
                AA[0:6, na-1] = np.array([1, 1, 1, 0, 0, 0])
                AA[na-1, 0:6] = np.array([1, 1, 1, 0, 0, 0])
                AA[na-1, na-1] = 0.0
            # use linear solver
            try:
                dm = np.linalg.solve(AA, bb)
            except:
                print ('Matrix is singular...LinearAlgError')
                raise ValueError("Check Matrix Singularity")
            new_par = old_par[0:npar] + dm[0:npar]
            return new_par

        else:
            # if invert for moment tensor with double couple constraints
            # setup starting solution, solve directly for moment instead
            # of dm, exact implementation of (A16)
            print ('No-linear Inversion')
            mstart = np.copy(old_par)
            m1 = np.copy(mstart)
            lam = np.zeros(2)

            # nolinear solver. Maybe there are already existing code.
            # check later
            for iter in range(const.NMAX_NL_ITER):
                self.get_f_df(A, b, m1, lam, mstart, AA, bb)
                bb = - bb
                xout = np.linalg.solve(AA, bb)
                m1 = m1 + xout
                lam = lam + xout[npar:na]
            dm = m1 - mstart
            new_par = m1

        new_cmt_par = np.copy(self.cmt_par)
        new_cmt_par = new_par * self.config.scale_par
        return new_cmt_par

    # The function invert_bootstrap
    # It is used to evaluate the mean, standard deviation,
    # and variance of new parameters
    def invert_bootstrap(self):
        new_par_array = np.zeros((self.config.bootstrap_repeat, self.npar))
        for i in range(self.config.bootstrap_repeat):
            new_par = self.invert_cmt(self.A_bootstrap[i], self.b_bootstrap[i])
            new_par_array[i] = new_par
        self.par_mean = np.mean(new_par_array, axis=0)
        self.par_std = np.std(new_par_array, axis=0)
        self.par_var = np.var(new_par_array, axis=0)

    def get_f_df(self, A, b, m, lam, mstart, fij, f0):

        npar = self.config.npar
        NM = const.NM

        # U_j
        dc1_dm = np.array([1,1,1,0,0,0])

        # V_j
        dc2_dm = np.zeros(6)
        dc2_dm[0] = m[1] * m[2] - m[5]**2
        dc2_dm[1] = m[0] * m[2] - m[4]**2
        dc2_dm[2] = m[0] * m[1] - m[3]**2
        dc2_dm[3] = 2 * m[4] * m[5] - 2 * m[2] * m[3]
        dc2_dm[4] = 2 * m[3] * m[5] - 2 * m[1] * m[4]
        dc2_dm[5] = 2 * m[3] * m[4] - 2 * m[0] * m[5]

        # f(x^i) = H_jk (m_k^i -m_k^0) - b_j + lam_1 * U_j + lam_2 * V_j (A11)
        f0.fill(0)
        f0[0:npar] = np.dot(A[0:npar, 0:npar], m[0:npar]-mstart[0:npar]) - b[0:npar]
        f0[0:const.NM] += lam[0] * dc1_dm[0:const.NM] + lam[1] * dc2_dm[0:const.NM]
        # f_(n+1) and f_(n+2)
        f0[npar] = m[0] + m[1] + m[2]
        moment_tensor = np.array([[m[0], m[3], m[4]],[m[3],m[1],m[5]],[m[4],m[5],m[2]]])
        f0[npar+1] = np.linalg.det(moment_tensor)

        # Y_jk
        dc2_dmi_dmj = np.zeros([6,6])
        dc2_dmi_dmj[0,:] = np.arrays([     0.0,    m[2],     m[1],      0.0,         0.0,    -2.0*m[5]   ])
        dc2_dmi_dmj[1,:] = np.arrays([    m[2],    0.0,      m[0],      0.0,      -2.0*m[4],   0.0     ])
        dc2_dmi_dmj[2,:] = np.arrays([    m[1],    m[0],      0.0,   -2.0*m[3],    0.0,        0.0     ])
        dc2_dmi_dmj[3,:] = np.arrays([     0.0,     0.0,  -2.0*m[3], -2.0*m[2],    2*m[5],    2*m[4]   ])
        dc2_dmi_dmj[4,:] = np.arrays([     0.0, -2.0*m[4],    0.0,    2.0*m[5], -2.0*m[1],    2*m[3]   ])
        dc2_dmi_dmj[5,:] = np.arrays([ -2.0*m[5],   0.0,      0.0,    2.0*m[4],    2.0*m[3],  -2.0*m[0]])

        # ! f_jk = H_jk + lam_2 * Y_jk
        fij.fill(0)
        fij[0:npar, 0:npar] = A[0:npar, 0:npar]
        fij[0:NM, 0:NM] = fij[0:NM, 0:NM] + lam[1] * dc2_dmi_dmj[0:NM, 0:NM]
        fij[0:NM, npar] = dc1_dm
        fij[0:NM, npar+1] = dc2_dm
        fij[npar, 0:NM] = dc1_dm
        fij[npar+1, 0:NM] = dc2_dm

    def variance_reduction(self):
        """
        Calculate variance reduction after source updated
        :return:
        """
        fh = open("cmt3d_flexwin.out", "w")
        fh.write("%d\n" %self.num_file)

        nwint = 0
        var_all = 0.0
        var_all_new = 0.0
        for _idx, window in self.window:
            new_synt = self.compute_new_syn(window.datalist, dm)
            obsd = window.datalist['obsd']
            synt = window.datalist['synt']
            npts = min(obsd.npts, synt.npts)
            for win_time in window.win_time:
                nwint += 1
                tstart = win_time[0]
                tend = win_time[1]
                idx_start = max(math.floor(tstart/obsd.stats.delta),1)
                idx_end = min(math.ceil(tend/obsd.stats.delta), obsd.stats.npts)

                if self.config.station_correction:
                    [nshift, cc, dlnA] = self.calculate_criteria(obsd, synt, idx_start, idx_end)
                    [nshift_new, cc_new, dlnA_new] = self.calculate_criteria(obsd, new_synt, idx_start, idx_end,)
                    istart_d = max(1, idx_start+nshift)
                    iend_d = min(npts, idx_start+nshift)
                    istart_dn = max(1, idx_start+nshift_new)
                    iend_dn = min(npts, idx_end+nshift_new)
                    istart = istart_d - nshift
                    iend = iend_d - nshift
                    istart_n = istart_dn - nshift_new
                    iend_n = iend_dn - nshift - nshift_new
                else:
                    istart_d = idx_start
                    istart = idx_start
                    iend_d = idx_end
                    iend = idx_end
                    istart_dn = idx_start
                    istart_n = idx_start
                    iend_dn = idx_end
                    iend_n = idx_end

                taper = self.construct_hanning_taper(iend-istart+1)
                v1 = np.sum(taper * (synt.data[istart:iend]-obsd.data[istart_d:iend_d])**2)
                v2 = np.sum(taper * (new_synt.data[istart_n:iend_n]-obsd.data[istart_dn:iend_dn])**2)
                d1 = np.sum(taper * obsd.data[istart_d:iend_d]**2)
                d2 = np.sum(taper * obsd.data[istart_dn:iend_dn]**2)

                self.var_all += 0.5*v1*window.weight*obsd.stats.delta
                self.var_all_new += 0.5*v2*window.weight*obsd.stats.delta


    def compute_new_syn(self, datalist, dm):
        # get a dummy copy to keep meta data information
        new_synt = datalist['synt'].copy()

        npar = self.config.npar
        npts = datalist['synt'].stats.npts
        dt = datalist['synt'].stats.delta
        dsyn = np.zeros_like([npts, npar])
        par_list = self.config.par_name
        dcmt_par = self.config.dcmt_par

        for i in range(npar):
            if i < const.NM:
                dsyn[:, i] = datalist[par_list[i].data] / dcmt_par[i]
            elif i < const.NML:
                dsyn[:, i] = ( datalist[par_list[i]].data - datalist['synt'].data) / dcmt_par[i]
            elif i == const.NML:
                dsyn[0:(npts-1), i] = -(datalist['synt'].data[1:npts] - datalist[0:(npts-1)]) / (dt*dcmt_par[i])
                dsyn[npts-1, i] = dsyn[npts-2,i]
            elif i == (const.NML + 1):
                # what the hell here....
                pass

            new_synt.data = datalist['synt'].data + np.dot(dsyn, dm)

    @staticmethod
    def calculate_criteria(obsd, synt, istart, iend):
        # cross-correlation measurement
        len = istart - iend
        zero_padding = np.zeros(len)
        trace1 = np.concatenate((zero_padding, obsd.data[istart:iend], zero_padding), axis=0)
        trace2 = np.concatenate((zero_padding, synt.data[istart:iend], zero_padding), axis=0)
        nshift, max_cc = xcorr(trace1, trace2, len)
        # amplitude anomaly
        dlnA = math.sqrt( np.dot(trace1, trace1)/np.dot(trace2, trace2)) - 1.0

        return [nshift, max_cc, dlnA]

    @staticmethod
    def construct_hanning_taper(npts):
        taper = np.zeros(npts)
        for i in range(npts):
            taper[i] = 0.5 * (1 - math.cos(2 * np.pi * (i / (npts-1))))

    def source_inversion(self):
        self.load_winfile()
        self.setup_weight()
        # setup matrix based on misfit
        self.setup_matrix()
        # calculate new cmt solution
        if self.config.bootstrap == False:
            self.invert_cmt(self.A, self.b)
            self.calculate_variance_reduction()
        elif self.config.bootstrap == True:
            self.invert_bootstrap()
