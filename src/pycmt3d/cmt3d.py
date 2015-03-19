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
from source import CMTSource
from obspy.core.util.geodetics import gps2DistAzimuth
from obspy.signal.cross_correlation import xcorr
import const
from window import Window
from __init__ import logger
import obspy

class Cmt3D(object):

    def __init__(self, cmtsource, window, config):
        self.config = config
        self.cmtsource = cmtsource
        self.window = window

        self.print_source_summary()

    def setup_weight(self):
        """
        Use Window to setup weight
        """
        logger.info("*"*15)
        logger.info("Start weighting...")
        if self.config.weight_data:
            # first calculate azimuth and distance for each data pair
            self.prepare_for_weighting()
            # then calculate azimuth weighting
            naz_list = self.calculate_azimuth_bin()
            logger.info("Azimuth bin: [%s]" %(', '.join(map(str, naz_list))))
            for idx, window in enumerate(self.window):
                idx_naz = self.get_azimuth_bin_number(window.azimuth)
                naz = naz_list[idx_naz]
                logger.debug("%s.%s.%s, num_win, dist, naz: %d, %.2f, %d", window.station, window.network, window.component,
                            window.num_wins, window.dist_in_km, naz)
                window.weight = self.config.weight_function(window.component, window.dist_in_km, naz, window.num_wins)
            # normalization of data weights
            # Attention: the code here might be tedious but I just do not know how to make it bette
            # without changing previous codes
            self.normalize_weight()
        else:
            for idx, window in enumerate(self.window):
                # set even weighting
                window.weight = np.ones(window.num_wins)

    def prepare_for_weighting(self):
        """
        Calculate azimuth and distance and store it
        """
        event_lat = self.cmtsource.latitude
        event_lon = self.cmtsource.longitude
        for window in self.window:
            sta_lat = window.datalist['synt'].stats.sac['stla']
            sta_lon = window.datalist['synt'].stats.sac['stlo']
            dist_in_m, az, baz = gps2DistAzimuth(event_lat, event_lon, sta_lat, sta_lon)
            window.dist_in_km = dist_in_m/1000.0
            window.azimuth = az

    @staticmethod
    def get_azimuth_bin_number(azimuth):
        """
        Calculate the bin number of a give azimuth
        :param azimuth:
        :return:
        """
        # the azimth ranges from [0,360]
        # so a little modification here
        daz = 360.0 / const.NREGIONS
        k = int(math.floor(azimuth / daz))
        if ( k<0 or k>const.NREGIONS):
            if abs(azimuth-360.0) < 0.0001:
                k = const.NREGIONS - 1
            else:
                raise ValueError ('Error bining azimuth')
        return k

    def calculate_azimuth_bin(self):
        """
        Calculate the azimuth and sort them into bins
        :return:
        """
        naz_list = np.zeros(const.NREGIONS)
        for window in self.window:
            bin_idx = self.get_azimuth_bin_number(window.azimuth)
            # 1) weight on window numbers
            #naz_list[bin_idx] += window.num_wins
            # 2) weigth on files
            naz_list[bin_idx] += 1

        return naz_list

    def normalize_weight(self):
        max_weight = 0.0
        for window in self.window:
            max_temp = np.max(window.weight)
            if max_temp > max_weight:
                max_weight = max_temp

        logger.debug("Global Max Weight: %f" %(max_weight))

        for window in self.window:
            logger.debug("%s.%s.%s, weight: [%s]" %(window.network, window.station, window.component,
                                                    ', '.join(map(self.float_to_str, window.weight))))
            window.weight /= max_weight
            logger.debug("Updated, weight: [%s]" %(', '.join(map(self.float_to_str, window.weight))))

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
        logger.info("*"*15)
        logger.info("Set up inversion matrix")
        self.A = np.zeros((self.config.npar, self.config.npar))
        self.b = np.zeros(self.config.npar)
        A1_all = []
        b1_all = []
        for window in self.window:
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
            logger.info("Inversion Matrix A is as follows:")
            logger.info("\n%s" %('\n'.join(map(str, self.A))))
            logger.info("RHS vector b is as follows:")
            logger.info("[%s]" %(', '.join(map(str, self.b))))

        # we setup the full array, but based on npar, only part of it will be used
        cmt = self.cmtsource
        self.cmt_par = np.array([cmt.m_rr, cmt.m_tt, cmt.m_pp, cmt.m_rt, cmt.m_rp, cmt.m_tp,
                               cmt.depth_in_m/1000.0, cmt.longitude,
                               cmt.latitude, cmt.time_shift, cmt.half_duration])

    def compute_A_b(self, window, win_idx):

        par_list = self.config.par_name
        npar = self.config.npar
        dcmt_par = self.config.dcmt_par

        datalist = window.datalist
        obsd = datalist['obsd']
        synt = datalist['synt']
        npts = min(obsd.stats.npts, synt.stats.npts)
        win = [window.win_time[win_idx,0], window.win_time[win_idx,1]]

        istart = int(max(math.floor(win[0]/obsd.stats.delta),1))
        iend = int(min(math.ceil(win[1]/obsd.stats.delta),npts))
        if istart > iend:
            raise ValueError("Check window for %s.%s.%s.%s" %(window.station,
                            window.network, window.location, window.component))

        if self.config.station_correction:
            [nshift, cc, dlna] = self.calculate_criteria(obsd, synt, istart, iend)
            print "shift:", nshift
            istart_d = max(1, istart + nshift)
            iend_d = min(npts, iend + nshift)
            istart_s = istart_d - nshift
            iend_s = iend_d - nshift
        else:
            istart_d = istart
            iend_d = iend
            istart_s = istart
            iend_s = iend
        #print "debug, shift", istart, iend, istart_s, iend_s, nshift

        dsyn = np.zeros((npar, npts))
        for itype in range(npar):
            type_name = par_list[itype]
            if itype < const.NML:
                # check file: check dt, npts
                dt_synt = datalist['synt'].stats.delta
                dt_obsd = datalist['obsd'].stats.delta
                if abs(dt_synt - dt_obsd) > 0.0001:
                    raise ValueError("Delta in synthetic and observed no the same")
                dt = dt_synt
            if itype < const.NM: # moment tensor
                dsyn[itype, 0:npts] = datalist[type_name].data[0:npts]/dcmt_par[itype]
            elif itype < const.NML:  # location
                dsyn[itype, 0:npts] = (datalist[type_name].data[0:npts]-datalist['synt'].data[0:npts])/dcmt_par[itype]
            elif itype == const.NML:  # time shift
                dsyn[itype, 0:npts-1] = (datalist['synt'].data[1:npts]-datalist['synt'].data[0:(npts-1)])/(dt*dcmt_par[itype])
                dsyn[itype, npts-1] = dsyn[itype, npts-2]
            elif itype == const.NML + 1: # half duration
                dsyn[itype, 0:npts-1] = -0.5 * self.cmt_par[itype] * (dsyn[const.NML,1:npts]-dsyn[const.NML,0:npts-1])/dt
                dsyn[itype, npts-1] = dsyn[itype, npts-2]

        # hanning taper
        taper = self.construct_hanning_taper(iend_s-istart_s)
        A1 = np.zeros((npar, npar))
        b1 = np.zeros(npar)
        # compute A and b by taking into account data weights
        for i in range(npar):
             for j in range(npar):
                 #print "debug:", istart_s, iend_s
                 #print "debug:", window.weight[win_idx], dt #, np.sum(taper * dsyn[i, istart_s:iend_s] * dsyn[j,istart_s:iend_s])
                 #print "debug, sum:", taper.shape, taper
                 #print "debug, dsyn:", dsyn[i, istart_s:iend_s].shape, dsyn[i, istart_s:iend_s]
                 A1[i][j] = window.weight[win_idx] * np.sum(taper * dsyn[i, istart_s:iend_s] * dsyn[j,istart_s:iend_s]) * dt
             b1[i] = window.weight[win_idx] * np.sum(taper * (obsd.data[istart_d:iend_d] - synt.data[istart_s:iend_s]) *
                     dsyn[i, istart_s:iend_s])
        #for i in range(npar):
        #    A1[i] = window.weight[win_idx] * np.sum(taper * dsyn[i][istart_s:iend_s] * dsyn[:][istart_s:iend_s]) * dt
        #    b1 = window.weight[win_idx] * np.sum(taper * (obsd.data[istart_d:iend_d] - synt.data[istart_s:iend_s]) *
        #                                 dsyn[:][istart_s:iend_s])
        return [A1, b1]

    def invert_cmt(self, A, b):

        logger.info("*"*20)
        logger.info("Invert cmt parameters")

        npar = self.config.npar
        #old_par = self.cmt_par[0:npar]
        old_par = self.cmt_par[0:npar]/self.config.scale_par[0:npar]

        # scale the A and b matrix
        max_row = np.amax(abs(A), axis=1)
        for i in range(len(b)):
            A[i,:] /= max_row[i]
            b[i] /= max_row[i]

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
        bb = np.zeros(na)
        if linear_inversion:
            # if invert for moment tensor with zero-trace constraints or no constraint
            logger.info("Linear Inversion")
            AA[0:npar, 0:npar] = A
            bb[0:npar] = b
            if self.config.zero_trace:
                bb[na-1] = - np.sum(old_par[0:3])
                AA[0:6, na-1] = np.array([1, 1, 1, 0, 0, 0])
                AA[na-1, 0:6] = np.array([1, 1, 1, 0, 0, 0])
                AA[na-1, na-1] = 0.0
            # use linear solver
            try:
                dm = np.linalg.solve(AA, bb)
            except:
                logger.error('Matrix is singular...LinearAlgError')
                raise ValueError("Check Matrix Singularity")
            # check here
            new_par = old_par[0:npar] + dm[0:npar]
            logger.debug("cmt_par: [%s]" %(', '.join(map(str, self.cmt_par[0:npar]))))
            logger.debug("Scaled old_par: [%s]" %(', '.join(map(str,old_par))))
            logger.debug("dm: [%s]" %(', '.join(map(str, dm))))

        else:
            # if invert for moment tensor with double couple constraints
            # setup starting solution, solve directly for moment instead
            # of dm, exact implementation of (A16)
            logger.info('Non-linear Inversion')
            mstart = np.copy(old_par)
            m1 = np.copy(mstart)
            lam = np.zeros(2)

            # nolinear solver. Maybe there are already existing code.
            # check later
            error = np.zeros([const.NMAX_NL_ITER, na])
            for iter in range(const.NMAX_NL_ITER):
                self.get_f_df(A, b, m1, lam, mstart, AA, bb)
                #logger.info("Inversion Matrix AA is as follows:")
                #logger.info("\n%s" %('\n'.join(map(str, AA))))
                #logger.info("Inversion vector bb is as follows:")
                #logger.info("[%s]" %(', '.join(map(str, bb))))
                bb = - bb
                xout = np.linalg.solve(AA, bb)
                #logger.debug("xout: [%s]" %(', '.join(map(str, xout))))
                m1 = m1 + xout[0:npar]
                lam = lam + xout[npar:na]
                error[iter, :] = np.dot(AA, xout) - bb
            dm = m1 - mstart
            new_par = m1
            #logger.debug("dm: [%s]" %(', '.join(map(str, dm))))
            #logger.debug("Scaled old_par: [%s]" %(', '.join(map(str,old_par))))
            #for iter in range(const.NMAX_NL_ITER):
            #    print "iter", iter, error[iter, :]
            #    print "sum abs error:", np.sum(abs(error[iter, :]))

        new_cmt_par = np.copy(self.cmt_par)
        new_cmt_par[0:npar] = new_par[0:npar] * self.config.scale_par[0:npar]
        self.new_cmt_par = new_cmt_par
        logger.info("Old CMT par: [%s]" %(', '.join(map(str, self.cmt_par))))
        logger.info("dm: [%s]" %(', '.join(map(str, new_cmt_par-self.cmt_par))))
        logger.info("New CMT par: [%s]" %(', '.join(map(str, new_cmt_par))))

        # convert it to CMTSource instance
        self.convert_new_cmt_par()
        logger.info("Trace: %e" %(np.sum(new_cmt_par[0:3])))
        logger.info("Energy change: %f%%" %( (self.new_cmtsource.M0 - self.cmtsource.M0)/self.cmtsource.M0*100.0))

    def convert_new_cmt_par(self):
        """
        Convert self.new_cmt_par array to CMTSource
        :return:
        """
        oldcmt = self.cmtsource
        newcmt = self.new_cmt_par
        time_shift = newcmt[9]
        new_cmt_time = oldcmt.origin_time + time_shift
        # copy old one
        self.new_cmtsource = CMTSource(origin_time=oldcmt.origin_time,
            pde_latitude=oldcmt.pde_latitude, pde_longitude=oldcmt.pde_longitude,
            mb=oldcmt.mb, ms=oldcmt.ms, pde_depth_in_m=oldcmt.pde_depth_in_m,
            region_tag=oldcmt.region_tag, eventname=oldcmt.eventname,
            cmt_time=new_cmt_time,  half_duration=newcmt[10],
            latitude=newcmt[8], longitude=newcmt[7], depth_in_m=newcmt[6],
            m_rr=newcmt[0], m_tt=newcmt[1], m_pp=newcmt[2], m_rt=newcmt[3], m_rp=newcmt[4], m_tp=newcmt[5])

    def invert_bootstrap(self):
        """
        It is used to evaluate the mean, standard deviation, and variance of new parameters
        """
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
        f0.fill(0.)
        f0[0:npar] = np.dot(A[0:npar, 0:npar], m[0:npar]-mstart[0:npar]) - b[0:npar]
        #print "f0 step1:", f0
        f0[0:const.NM] += lam[0] * dc1_dm[0:const.NM] + lam[1] * dc2_dm[0:const.NM]
        #print "f0 step2:", f0
        # f_(n+1) and f_(n+2)
        f0[npar] = m[0] + m[1] + m[2]
        moment_tensor = np.array([[m[0], m[3], m[4]],[m[3],m[1],m[5]],[m[4],m[5],m[2]]])
        f0[npar+1] = np.linalg.det(moment_tensor)
        #print "det1:", f0[npar+1]
        f0[npar+1] = m[0] * ( m[1] * m[2] - m[5] ** 2 ) \
                - m[3] * ( m[3] * m[2] - m[5] * m[4] ) \
                + m[4] * ( m[3] * m[5] - m[4] * m[1] )
        #print "det2:", f0[npar+1]

        # Y_jk
        dc2_dmi_dmj = np.zeros([6,6])
        dc2_dmi_dmj[0,:] = np.array([     0.0,    m[2],     m[1],      0.0,         0.0,    -2.0*m[5]  ])
        dc2_dmi_dmj[1,:] = np.array([    m[2],    0.0,      m[0],      0.0,      -2.0*m[4],   0.0      ])
        dc2_dmi_dmj[2,:] = np.array([    m[1],    m[0],      0.0,   -2.0*m[3],    0.0,        0.0      ])
        dc2_dmi_dmj[3,:] = np.array([     0.0,     0.0,  -2.0*m[3], -2.0*m[2],    2*m[5],    2*m[4]    ])
        dc2_dmi_dmj[4,:] = np.array([     0.0, -2.0*m[4],    0.0,    2.0*m[5], -2.0*m[1],    2*m[3]    ])
        dc2_dmi_dmj[5,:] = np.array([ -2.0*m[5],   0.0,      0.0,    2.0*m[4],    2.0*m[3],  -2.0*m[0] ])

        # ! f_jk = H_jk + lam_2 * Y_jk
        fij.fill(0)
        fij[0:npar, 0:npar] = A[0:npar, 0:npar]
        fij[0:NM, 0:NM] = fij[0:NM, 0:NM] + lam[1] * dc2_dmi_dmj[0:NM, 0:NM]
        fij[0:NM, npar] = dc1_dm
        fij[0:NM, npar+1] = dc2_dm
        fij[npar, 0:NM] = dc1_dm
        fij[npar+1, 0:NM] = dc2_dm

    def calculate_var(self):

        npar = self.config.npar
        dm = self.new_cmt_par[0:npar] - self.cmt_par[0:npar]

        var_all = 0.0
        var_all_new = 0.0
        for _idx, window in enumerate(self.window):
            obsd = window.datalist['obsd']
            synt = window.datalist['synt']
            self.compute_new_syn(window.datalist, dm)
            new_synt = window.datalist['new_synt']
            # calculate old variance
            [v1, d1] = self.calculate_var_reduction_one_trace(obsd, synt, window.win_time)
            # calculate new variance
            [v2, d2] = self.calculate_var_reduction_one_trace(obsd, new_synt, window.win_time)

            var_all += np.sum(0.5*v1*window.weight*obsd.stats.delta)
            var_all_new += np.sum(0.5*v2*window.weight*obsd.stats.delta)

        logger.info("Total Variance Reduced from %e to %e ===== %f %%"
                    %(var_all, var_all_new, (var_all-var_all_new)/var_all*100))

    def calculate_var_reduction_one_trace(self, obsd, synt, win_time):
        """
        Calculate the variance reduction on a pair of obsd and synt and windows
        :param obsd:
        :param synt:
        :param win_time:
        :return: variance v1 and energy d1
        """
        num_wins = win_time.shape[0]
        v1 = np.zeros(num_wins)
        d1 = np.zeros(num_wins)
        npts = min(obsd.stats.npts, synt.stats.npts)
        for _win_idx in range(win_time.shape[0]):
            tstart = win_time[_win_idx, 0]
            tend = win_time[_win_idx, 1]
            idx_start = int(max(math.floor(tstart/obsd.stats.delta),1))
            idx_end = int(min(math.ceil(tend/obsd.stats.delta), obsd.stats.npts))
            if self.config.station_correction:
                [nshift, cc, dlnA] = self.calculate_criteria(obsd, synt, idx_start, idx_end)
                #print "shift:", nshift
                istart_d = max(1, idx_start + nshift)
                iend_d = min(npts, idx_end + nshift)
                istart = istart_d - nshift
                iend = iend_d - nshift
            else:
                istart_d = idx_start
                istart = idx_start
                iend_d = idx_end
                iend = idx_end

            taper = self.construct_hanning_taper(iend-istart)
            v1[_win_idx] = np.sum(taper * (synt.data[istart:iend]-obsd.data[istart_d:iend_d])**2)
            d1[_win_idx] = np.sum(taper * obsd.data[istart_d:iend_d]**2)
            #print "v1, idx:", v1[_win_idx], istart, iend, istart_d, iend_d, _win_idx, nshift
        return [v1, d1]

    def compute_new_syn(self, datalist, dm):
        # get a dummy copy to keep meta data information
        datalist['new_synt'] = datalist['synt'].copy()

        npar = self.config.npar
        npts = datalist['synt'].stats.npts
        dt = datalist['synt'].stats.delta
        dsyn = np.zeros([npts, npar])
        par_list = self.config.par_name
        dcmt_par = self.config.dcmt_par
        dm_scaled = dm/self.config.scale_par[0:npar]

        for i in range(npar):
            if i < const.NM:
                dsyn[:, i] = datalist[par_list[i]].data / dcmt_par[i]
            elif i < const.NML:
                dsyn[:, i] = ( datalist[par_list[i]].data - datalist['synt'].data) / dcmt_par[i]
            elif i == const.NML:
                dsyn[0:(npts-1), i] = -(datalist['synt'].data[1:npts] - datalist[0:(npts-1)]) / (dt*dcmt_par[i])
                dsyn[npts-1, i] = dsyn[npts-2,i]
            elif i == (const.NML + 1):
                # what the hell here....
                pass

        datalist['new_synt'].data = datalist['synt'].data + np.dot(dsyn, dm_scaled)

    @staticmethod
    def calculate_criteria(obsd, synt, istart, iend):
        # cross-correlation measurement
        len = iend - istart
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
            taper[i] = 0.5 * (1 - math.cos(2 * np.pi * (float(i) / (npts-1))))
        return taper

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

    def print_source_summary(self):
        cmt = self.cmtsource
        logger.info("="*10 + "  Event Summary  " + "="*10)
        logger.info("Event name: %s" %cmt.eventname)
        logger.info("   Latitude and longitude: %.2f, %.2f" %(cmt.latitude, cmt.longitude))
        logger.info("   Depth: %.1f km" % (cmt.depth_in_m/1000.0))
        logger.info("   Region tag: %s" %cmt.region_tag)
        logger.info("   Trace: %.3e" %((cmt.m_rr + cmt.m_tt + cmt.m_pp)/cmt.M0))
        logger.info("   Moment Magnitude: %.2f" %(cmt.moment_magnitude))

    @staticmethod
    def float_to_str(value):
        return "%.5f" % value