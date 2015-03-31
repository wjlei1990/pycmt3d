#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
from source import CMTSource
from obspy.core.util.geodetics import gps2DistAzimuth
import const
from __init__ import logger
import util


class Cmt3D(object):
    """
    Class that handles the solver part of source inversion

    :param cmtsource: earthquake source
    :type cmtsource: :class:`pycmt3d.CMTSource`
    :param data_container: all data and window
    :type DataContainer: :class:`pycmt3d.DataContainer`
    :param config: configuration for source inversion
    :type config: :class:`pycmt3d.Config`
    """

    def __init__(self, cmtsource, data_container, config):

        self.config = config
        self.cmtsource = cmtsource
        self.data_container = data_container
        self.window = self.data_container.window
        self.nwins = self.data_container.nwins

        # weight array
        self.weight_array = np.zeros(self.nwins)
        # measurement from each window
        self.A1_all = []
        self.b1_all = []
        # original cmt par array
        self.cmt_par = np.array([cmtsource.m_rr, cmtsource.m_tt, cmtsource.m_pp, cmtsource.m_rt, cmtsource.m_rp,
                                 cmtsource.m_tp, cmtsource.depth_in_m / 1000.0, cmtsource.longitude,
                                 cmtsource.latitude, cmtsource.time_shift, cmtsource.half_duration])
        # new cmt par from the inversion
        self.new_cmt_par = None
        self.new_cmtsource = None

        # bootstrap stat var
        self.par_mean = None
        self.par_std = None
        self.par_var = None
        self.std_over_mean = None

        self.print_cmtsource_summary(self.cmtsource)

    def setup_weight(self):
        """
        Use Window and location information to setup weight

        :returns:
        """
        logger.info("*" * 15)
        logger.info("Start weighting...")
        if self.config.weight_data:
            # first calculate azimuth and distance for each data pair
            self.prepare_for_weighting()
            # then calculate azimuth weighting
            naz_files, naz_wins = self.calculate_azimuth_bin()
            logger.info("Azimuth file bin: [%s]" % (', '.join(map(str, naz_files))))
            logger.info("Azimuth win bin: [%s]" % (', '.join(map(str, naz_wins))))
            for idx, window in enumerate(self.window):
                idx_naz = self.get_azimuth_bin_number(window.azimuth)
                naz = naz_files[idx_naz]
                logger.debug("%s.%s.%s, num_win, dist, naz: %d, %.2f, %d", window.station, window.network,
                             window.component,
                             window.num_wins, window.dist_in_km, naz)

                if self.config.normalize_window:
                    mode = "damping"
                else:
                    mode = "exponential"
                # weighting on compoent, distance and azimuth
                window.weight = self.config.weight_function(window.component, window.dist_in_km, naz, window.num_wins,
                                                            dist_weight_mode=mode)

                if self.config.normalize_window:
                    # normalize each window's measurement by energy
                    window.weight = window.weight/window.energy

            # normalization of data weights
            # Attention: the code here might be tedious but I just do not know how to make it bette
            # without changing previous codes
            self.normalize_weight()

            self.weight_array = np.zeros([self.data_container.nwins])
            _idx = 0
            for window in self.window:
                for win_idx in range(window.num_wins):
                    self.weight_array[_idx] = window.weight[win_idx]
                    _idx += 1

        else:
            for idx, window in enumerate(self.window):
                # set even weighting
                window.weight = np.ones(window.num_wins)

    def prepare_for_weighting(self):
        """
        Prepare necessary information for weighting, e.x., calculating azimuth and distance
        and store it

        :return:
        """
        for window in self.window:
            # calculate energy
            window.win_energy(mode=self.config.norm_mode)
            # calculate location
            window.get_location_info(self.cmtsource)

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
        naz_files = np.zeros(const.NREGIONS)
        naz_wins = np.zeros(const.NREGIONS)
        for window in self.window:
            bin_idx = self.get_azimuth_bin_number(window.azimuth)
            naz_files[bin_idx] += 1
            naz_wins[bin_idx] += window.num_wins
        return naz_files, naz_wins

    def normalize_weight(self):
        """
        Normalize the weighting and make the maximum to 1

        :return:
        """
        max_weight = 0.0
        for window in self.window:
            max_temp = np.max(window.weight)
            if max_temp > max_weight:
                max_weight = max_temp

        logger.debug("Global Max Weight: %f" % max_weight)

        for window in self.window:
            logger.debug("%s.%s.%s, weight: [%s]" % (window.network, window.station, window.component,
                                                     ', '.join(map(self._float_to_str, window.weight))))
            window.weight /= max_weight
            logger.debug("Updated, weight: [%s]" % (', '.join(map(self._float_to_str, window.weight))))

    def get_station_info(self, datalist):
        """
        Using the event location and station information to calculate azimuth and distance

        :param datalist: data dictionary(referred to pycmt3d.Window.datalist)
        :return:
        """
        # this might be related to datafile type(sac, mseed or asdf)
        event_lat = self.cmtsource.latitude
        event_lon = self.cmtsource.longitude
        # station location from synthetic file
        sta_lat = datalist['synt'].stats.sac['stla']
        sta_lon = datalist['synt'].stats.sac['stlo']
        dist_in_m, az, baz = gps2DistAzimuth(event_lat, event_lon, sta_lat, sta_lon)
        return [dist_in_m / 1000.0, az]

    def setup_matrix(self):
        """
        Calculate A and b for all windows

        :return:
        """
        logger.info("*" * 15)
        logger.info("Set up inversion matrix")

        for window in self.window:
            # loop over pair of data
            for win_idx in range(window.num_wins):
                # loop over each window
                # here, A and b are exact measurements and no weightings are applied
                [A1, b1] = self.compute_A_b(window, win_idx)
                self.A1_all.append(A1)
                self.b1_all.append(b1)

    def compute_A_b(self, window, win_idx):
        """
        Calculate the matrix A and vector b based on one pair of observed data and
        synthetic data on a given window.

        :param window: data and window information
        :type window: :class:`pycmt3d.Window`
        :param win_idx: window index(a specific window)
        :type win_idx: integer
        :return:
        """
        par_list = self.config.par_name
        npar = self.config.npar
        dcmt_par = self.config.dcmt_par

        datalist = window.datalist
        obsd = datalist['obsd']
        synt = datalist['synt']
        npts = min(obsd.stats.npts, synt.stats.npts)
        win = [window.win_time[win_idx, 0], window.win_time[win_idx, 1]]

        istart = int(max(math.floor(win[0] / obsd.stats.delta), 1))
        iend = int(min(math.ceil(win[1] / obsd.stats.delta), npts))
        if istart > iend:
            raise ValueError("Check window for %s.%s.%s.%s" % (window.station,
                                                               window.network, window.location, window.component))

        if self.config.station_correction:
            [nshift, cc, dlna] = self.calculate_criteria(obsd, synt, istart, iend)
            # print "shift:", nshift
            istart_d = max(1, istart + nshift)
            iend_d = min(npts, iend + nshift)
            istart_s = istart_d - nshift
            iend_s = iend_d - nshift
        else:
            istart_d = istart
            iend_d = iend
            istart_s = istart
            iend_s = iend
        # print "debug, shift", istart, iend, istart_s, iend_s, nshift

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
            if itype < const.NM:  # moment tensor
                dsyn[itype, 0:npts] = datalist[type_name].data[0:npts] / dcmt_par[itype]
            elif itype < const.NML:  # location
                dsyn[itype, 0:npts] = (datalist[type_name].data[0:npts] - datalist['synt'].data[0:npts]) / dcmt_par[
                    itype]
            elif itype == const.NML:  # time shift
                dsyn[itype, 0:npts - 1] = (datalist['synt'].data[1:npts] - datalist['synt'].data[0:(npts - 1)]) / (
                    dt * dcmt_par[itype])
                dsyn[itype, npts - 1] = dsyn[itype, npts - 2]
            elif itype == const.NML + 1:  # half duration
                dsyn[itype, 0:npts - 1] = -0.5 * self.cmt_par[itype] * (
                    dsyn[const.NML, 1:npts] - dsyn[const.NML, 0:npts - 1]) / dt
                dsyn[itype, npts - 1] = dsyn[itype, npts - 2]

        # hanning taper
        taper = self.construct_hanning_taper(iend_s - istart_s)
        A1 = np.zeros((npar, npar))
        b1 = np.zeros(npar)
        # compute A and b by taking into account data weights
        for j in range(npar):
            for i in range(0, j + 1):
                A1[i, j] = np.sum(taper * dsyn[i, istart_s:iend_s] * dsyn[j, istart_s:iend_s]) * dt
            b1[j] = np.sum(taper * (obsd.data[istart_d:iend_d] -
                                    synt.data[istart_s:iend_s]) * dsyn[j, istart_s:iend_s]) * dt
        for j in range(npar):
            for i in range(j + 1, npar):
                A1[i, j] = A1[j, i]

        # print "debug, idx:", nshift, istart_s, iend_s, istart_d, iend_d, window.weight[win_idx]
        # print "debug, distance", window.dist_in_km
        # print "obsd sum, synt sum, dsyn sum:", np.sum(np.abs(obsd.data[istart_d:iend_d])), \
        #               np.sum(np.abs(synt.data[istart_s:iend_s])), np.sum(np.abs(dsyn[j, istart_s:iend_s]))
        # print "b1:", b1, np.sum(b1)

        return [A1, b1]

    def invert_solver(self, A, b):
        """
        Solver part. Hession matrix A and misfit vector b will be reconstructed here
        based on different constraints.

        :param A: basic Hessian matrix
        :param b: basid misfit vector
        :return:
        """

        npar = self.config.npar
        old_par = self.cmt_par[0:npar] / self.config.scale_par[0:npar]

        # scale the A and b matrix
        max_row = np.amax(abs(A), axis=1)
        for i in range(len(b)):
            A[i, :] /= max_row[i]
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
        np.fill_diagonal(damp_matrix, trace * self.config.lamda_damping)
        A = A + damp_matrix

        # setup new matrix based on constraints
        AA = np.zeros([na, na])
        bb = np.zeros(na)
        if linear_inversion:
            # if invert for moment tensor with zero-trace constraints or no constraint
            # logger.info("Linear Inversion")
            AA[0:npar, 0:npar] = A
            bb[0:npar] = b
            if self.config.zero_trace:
                bb[na - 1] = - np.sum(old_par[0:3])
                AA[0:6, na - 1] = np.array([1, 1, 1, 0, 0, 0])
                AA[na - 1, 0:6] = np.array([1, 1, 1, 0, 0, 0])
                AA[na - 1, na - 1] = 0.0
            # use linear solver
            try:
                dm = np.linalg.solve(AA, bb)
            except:
                logger.error('Matrix is singular...LinearAlgError')
                raise ValueError("Check Matrix Singularity")
            # check here
            new_par = old_par[0:npar] + dm[0:npar]

        else:
            # if invert for moment tensor with double couple constraints
            # setup starting solution, solve directly for moment instead
            # of dm, exact implementation of (A16)
            # logger.info('Non-linear Inversion')
            mstart = np.copy(old_par)
            m1 = np.copy(mstart)
            lam = np.zeros(2)

            # nolinear solver. Maybe there are already existing code.
            # check later
            error = np.zeros([const.NMAX_NL_ITER, na])
            for iter_idx in range(const.NMAX_NL_ITER):
                self.get_f_df(A, b, m1, lam, mstart, AA, bb)
                # logger.info("Inversion Matrix AA is as follows:")
                # logger.info("\n%s" %('\n'.join(map(str, AA))))
                # logger.info("Inversion vector bb is as follows:")
                # logger.info("[%s]" %(', '.join(map(str, bb))))
                bb = - bb
                xout = np.linalg.solve(AA, bb)
                # logger.debug("xout: [%s]" %(', '.join(map(str, xout))))
                m1 = m1 + xout[0:npar]
                lam = lam + xout[npar:na]
                error[iter_idx, :] = np.dot(AA, xout) - bb
            # dm = m1 - mstart
            new_par = m1

        new_cmt_par = np.copy(self.cmt_par)
        new_cmt_par[0:npar] = new_par[0:npar] * self.config.scale_par[0:npar]

        return new_cmt_par

    def invert_cmt(self):
        """
        ensemble all measurements together to form Matrix A and vector b to solve the A * (dm) = b
        A is the Hessian Matrix and b is the misfit

        :return:
        """
        # ensemble A and b
        A = util.sum_matrix(self.weight_array, self.A1_all)
        b = util.sum_matrix(self.weight_array, self.b1_all)
        logger.info("Inversion Matrix A is as follows:")
        logger.info("\n%s" % ('\n'.join(map(self._float_array_to_str, A))))
        logger.info("RHS vector b is as follows:")
        logger.info("[%s]" % (self._float_array_to_str(b)))

        # source inversion
        self.new_cmt_par = self.invert_solver(A, b)
        self.convert_new_cmt_par()

    def invert_bootstrap(self):
        """
        It is used to evaluate the mean, standard deviation, and variance of new parameters

        :return:
        """
        A_bootstrap = []
        b_bootstrap = []
        # Bootstrap to generate subset A and b
        for i in range(self.config.bootstrap_repeat):
            random_array = util.gen_random_array(self.nwins, sample_number=int(0.3 * self.nwins))
            A = util.sum_matrix(random_array * self.weight_array, self.A1_all)
            b = util.sum_matrix(random_array * self.weight_array, self.b1_all)
            A_bootstrap.append(A)
            b_bootstrap.append(b)

        # inversion of each subset
        new_par_array = np.zeros((self.config.bootstrap_repeat, const.NPARMAX))
        for i in range(self.config.bootstrap_repeat):
            new_par = self.invert_solver(A_bootstrap[i], b_bootstrap[i])
            new_par_array[i, :] = new_par

        # statistical analysis
        self.par_mean = np.mean(new_par_array, axis=0)
        self.par_std = np.std(new_par_array, axis=0)
        self.par_var = np.var(new_par_array, axis=0)
        self.std_over_mean = self.par_std / np.abs(self.par_mean)

    def source_inversion(self):
        self.setup_matrix()
        self.setup_weight()
        self.invert_cmt()

        # convert it to CMTSource instance
        self.calculate_variance()

        if self.config.bootstrap:
            self.invert_bootstrap()

        self.print_inversion_summary()

    def get_f_df(self, A, b, m, lam, mstart, fij, f0):
        """
        Iterative solver for Non-linear case(double-couple constraint)

        :param A: basic Hessian matrix
        :param b: basic misfit vector
        :param m: current source array
        :param lam: constraints coefficient for zero-trace and double-couple constraints
        :param mstart: starting source solution
        :param fij: reconstructed Hessian Matrix AA
        :param f0: reconstructed misfit vector bb
        :return:
        """

        npar = self.config.npar
        NM = const.NM

        # U_j
        dc1_dm = np.array([1, 1, 1, 0, 0, 0])

        # V_j
        dc2_dm = np.zeros(6)
        dc2_dm[0] = m[1] * m[2] - m[5] ** 2
        dc2_dm[1] = m[0] * m[2] - m[4] ** 2
        dc2_dm[2] = m[0] * m[1] - m[3] ** 2
        dc2_dm[3] = 2 * m[4] * m[5] - 2 * m[2] * m[3]
        dc2_dm[4] = 2 * m[3] * m[5] - 2 * m[1] * m[4]
        dc2_dm[5] = 2 * m[3] * m[4] - 2 * m[0] * m[5]

        # f(x^i) = H_jk (m_k^i -m_k^0) - b_j + lam_1 * U_j + lam_2 * V_j (A11)
        f0.fill(0.)
        f0[0:npar] = np.dot(A[0:npar, 0:npar], m[0:npar] - mstart[0:npar]) - b[0:npar]
        # print "f0 step1:", f0
        f0[0:const.NM] += lam[0] * dc1_dm[0:const.NM] + lam[1] * dc2_dm[0:const.NM]
        # f_(n+1) and f_(n+2)
        f0[npar] = m[0] + m[1] + m[2]
        moment_tensor = np.array([[m[0], m[3], m[4]], [m[3], m[1], m[5]], [m[4], m[5], m[2]]])
        f0[npar + 1] = np.linalg.det(moment_tensor)
        f0[npar + 1] = m[0] * (m[1] * m[2] - m[5] ** 2) \
                     - m[3] * (m[3] * m[2] - m[5] * m[4]) \
                     + m[4] * (m[3] * m[5] - m[4] * m[1])

        # Y_jk
        dc2_dmi_dmj = np.zeros([6, 6])
        dc2_dmi_dmj[0, :] = np.array([0.0, m[2], m[1], 0.0, 0.0, -2.0 * m[5]])
        dc2_dmi_dmj[1, :] = np.array([m[2], 0.0, m[0], 0.0, -2.0 * m[4], 0.0])
        dc2_dmi_dmj[2, :] = np.array([m[1], m[0], 0.0, -2.0 * m[3], 0.0, 0.0])
        dc2_dmi_dmj[3, :] = np.array([0.0, 0.0, -2.0 * m[3], -2.0 * m[2], 2 * m[5], 2 * m[4]])
        dc2_dmi_dmj[4, :] = np.array([0.0, -2.0 * m[4], 0.0, 2.0 * m[5], -2.0 * m[1], 2 * m[3]])
        dc2_dmi_dmj[5, :] = np.array([-2.0 * m[5], 0.0, 0.0, 2.0 * m[4], 2.0 * m[3], -2.0 * m[0]])

        # ! f_jk = H_jk + lam_2 * Y_jk
        fij.fill(0)
        fij[0:npar, 0:npar] = A[0:npar, 0:npar]
        fij[0:NM, 0:NM] = fij[0:NM, 0:NM] + lam[1] * dc2_dmi_dmj[0:NM, 0:NM]
        fij[0:NM, npar] = dc1_dm
        fij[0:NM, npar + 1] = dc2_dm
        fij[npar, 0:NM] = dc1_dm
        fij[npar + 1, 0:NM] = dc2_dm

    def calculate_variance(self):
        """
        Calculate variance reduction based on old and new source solution

        :return:
        """
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

            var_all += np.sum(0.5 * v1 * window.weight * obsd.stats.delta)
            var_all_new += np.sum(0.5 * v2 * window.weight * obsd.stats.delta)

        logger.info("Total Variance Reduced from %e to %e ===== %f %%"
                    % (var_all, var_all_new, (var_all - var_all_new) / var_all * 100))

    def calculate_var_reduction_one_trace(self, obsd, synt, win_time):
        """
        Calculate the variance reduction on a pair of obsd and synt and windows

        :param obsd: observed data trace
        :type obsd: :class:`obspy.core.trace.Trace`
        :param synt: synthetic data trace
        :type synt: :class:`obspy.core.trace.Trace`
        :param win_time: [win_start, win_end]
        :type win_time: :class:`list` or :class:`numpy.array`
        :return:  waveform misfit reduction and observed data energy [v1, d1]
        :rtype: [float, float]
        """
        num_wins = win_time.shape[0]
        v1 = np.zeros(num_wins)
        d1 = np.zeros(num_wins)
        npts = min(obsd.stats.npts, synt.stats.npts)
        for _win_idx in range(win_time.shape[0]):
            tstart = win_time[_win_idx, 0]
            tend = win_time[_win_idx, 1]
            idx_start = int(max(math.floor(tstart / obsd.stats.delta), 1))
            idx_end = int(min(math.ceil(tend / obsd.stats.delta), obsd.stats.npts))
            if self.config.station_correction:
                [nshift, cc, dlnA] = self.calculate_criteria(obsd, synt, idx_start, idx_end)
                # print "shift:", nshift
                istart_d = max(1, idx_start + nshift)
                iend_d = min(npts, idx_end + nshift)
                istart = istart_d - nshift
                iend = iend_d - nshift
            else:
                istart_d = idx_start
                istart = idx_start
                iend_d = idx_end
                iend = idx_end

            taper = self.construct_hanning_taper(iend - istart)
            v1[_win_idx] = np.sum(taper * (synt.data[istart:iend] - obsd.data[istart_d:iend_d]) ** 2)
            d1[_win_idx] = np.sum(taper * obsd.data[istart_d:iend_d] ** 2)
            # print "v1, idx:", v1[_win_idx], istart, iend, istart_d, iend_d, _win_idx, nshift
        return [v1, d1]

    def compute_new_syn(self, datalist, dm):
        """
        Compute new synthetic data based on new CMTSOLUTION

        :param datalist: dictionary of all data
        :param dm: CMTSolution perterbation, i.e., (self.new_cmt_par-self.cmt_par)
        :return:
        """
        # get a dummy copy to keep meta data information
        datalist['new_synt'] = datalist['synt'].copy()

        npar = self.config.npar
        npts = datalist['synt'].stats.npts
        dt = datalist['synt'].stats.delta
        dsyn = np.zeros([npts, npar])
        par_list = self.config.par_name
        dcmt_par = self.config.dcmt_par
        dm_scaled = dm / self.config.scale_par[0:npar]

        for i in range(npar):
            if i < const.NM:
                dsyn[:, i] = datalist[par_list[i]].data / dcmt_par[i]
            elif i < const.NML:
                dsyn[:, i] = (datalist[par_list[i]].data - datalist['synt'].data) / dcmt_par[i]
            elif i == const.NML:
                dsyn[0:(npts - 1), i] = -(datalist['synt'].data[1:npts] - datalist[0:(npts - 1)]) / (dt * dcmt_par[i])
                dsyn[npts - 1, i] = dsyn[npts - 2, i]
            elif i == (const.NML + 1):
                # what the hell here....
                pass

        datalist['new_synt'].data = datalist['synt'].data + np.dot(dsyn, dm_scaled)

    def calculate_criteria(self, obsd, synt, istart, iend):
        """
        Calculate the time shift, max cross-correlation value and energy differnce

        :param obsd: observed data trace
        :type obsd: :class:`obspy.core.trace.Trace`
        :param synt: synthetic data trace
        :type synt: :class:`obspy.core.trace.Trace`
        :param istart: start index of window
        :type istart: int
        :param iend: end index of window
        :param iend: int
        :return: [number of shift points, max cc value, dlnA]
        :rtype: [int, float, float]
        """
        obsd_trace = obsd.data[istart:iend]
        synt_trace = synt.data[istart:iend]
        max_cc, nshift = self._xcorr_win_(obsd_trace, synt_trace)
        dlnA = self._dlnA_win(obsd_trace, synt_trace)

        return [nshift, max_cc, dlnA]

    def convert_new_cmt_par(self):
        """
        Convert self.new_cmt_par array to CMTSource instance

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
                                       cmt_time=new_cmt_time, half_duration=newcmt[10],
                                       latitude=newcmt[8], longitude=newcmt[7], depth_in_m=newcmt[6] * 1000.0,
                                       m_rr=newcmt[0], m_tt=newcmt[1], m_pp=newcmt[2], m_rt=newcmt[3], m_rp=newcmt[4],
                                       m_tp=newcmt[5])

    @staticmethod
    def _xcorr_win_(obsd, synt):
        cc = np.correlate(obsd, synt, mode="full")
        nshift = cc.argmax() - len(obsd) + 1
        # Normalized cross correlation.
        max_cc_value = cc.max() / np.sqrt((synt ** 2).sum() * (obsd ** 2).sum())
        return max_cc_value, nshift

    @staticmethod
    def _dlnA_win(obsd, synt):
        return 0.5 * np.log(np.sum(obsd ** 2) / np.sum(synt ** 2))

    @staticmethod
    def construct_hanning_taper(npts):
        """
        Hanning taper construct

        :param npts: number of points
        :return:
        """
        taper = np.zeros(npts)
        for i in range(npts):
            taper[i] = 0.5 * (1 - math.cos(2 * np.pi * (float(i) / (npts - 1))))
        return taper

    @staticmethod
    def print_cmtsource_summary(cmt):
        """
        Print CMTSolution source summary

        :return:
        """
        logger.info("=" * 10 + "  Event Summary  " + "=" * 10)
        logger.info("Event name: %s" % cmt.eventname)
        logger.info("   Latitude and longitude: %.2f, %.2f" % (cmt.latitude, cmt.longitude))
        logger.info("   Depth: %.1f km" % (cmt.depth_in_m / 1000.0))
        logger.info("   Region tag: %s" % cmt.region_tag)
        logger.info("   Trace: %.3e" % ((cmt.m_rr + cmt.m_tt + cmt.m_pp) / cmt.M0))
        logger.info("   Moment Magnitude: %.2f" % cmt.moment_magnitude)

    @staticmethod
    def _float_to_str(value):
        """
        Convert float value to a specific precision string

        :param value:
        :return: string of the value
        """
        return "%.5f" % value

    @staticmethod
    def _float_array_to_str(array):
        """
        Convert float array to string

        :return:
        """
        string = "[  "
        for ele in array:
            string += "%10.3e  " % ele
        string += "]"
        return string

    def print_inversion_summary(self):
        """
        Print out the inversion summary

        :return:
        """
        logger.info("*" * 20)
        logger.info("Invert cmt parameters(%d par)" % self.config.npar)

        logger.info("Old CMT par: [%s]" % (', '.join(map(str, self.cmt_par))))
        logger.info("dm: [%s]" % (', '.join(map(str, self.new_cmt_par - self.cmt_par))))
        logger.info("New CMT par: [%s]" % (', '.join(map(str, self.new_cmt_par))))

        logger.info("Trace: %e" % (np.sum(self.new_cmt_par[0:3])))
        logger.info("Energy change: %5.2f%%" % ((self.new_cmtsource.M0 - self.cmtsource.M0) / self.cmtsource.M0 * 100.0))

        self.inversion_result_table()

    def inversion_result_table(self):
        """
        Print out the inversion table

        :return:
        """
        title = "*" * 20 + " Inversion Result Table(%d npar) " % self.config.npar + "*" * 20
        logger.info(title)

        if not self.config.bootstrap:
            logger.info("PAR         Old_CMT        New_CMT")
            logger.info("Mrr:  %15.6e  %15.6e" % (self.cmtsource.m_rr, self.new_cmtsource.m_rr))
            logger.info("Mtt:  %15.6e  %15.6e" % (self.cmtsource.m_tt, self.new_cmtsource.m_tt))
            logger.info("Mpp:  %15.6e  %15.6e" % (self.cmtsource.m_pp, self.new_cmtsource.m_pp))
            logger.info("Mrt:  %15.6e  %15.6e" % (self.cmtsource.m_rt, self.new_cmtsource.m_rt))
            logger.info("Mrp:  %15.6e  %15.6e" % (self.cmtsource.m_rp, self.new_cmtsource.m_rp))
            logger.info("Mtp:  %15.6e  %15.6e" % (self.cmtsource.m_tp, self.new_cmtsource.m_tp))
            logger.info(
                "dep:  %15.3f  %15.3f" % (self.cmtsource.depth_in_m / 1000.0, self.new_cmtsource.depth_in_m / 1000.0))
            logger.info("lon:  %15.3f  %15.3f" % (self.cmtsource.longitude, self.new_cmtsource.longitude))
            logger.info("lat:  %15.3f  %15.3f" % (self.cmtsource.latitude, self.new_cmtsource.latitude))
            logger.info("ctm:  %15.3f  %15.3f" % (self.cmtsource.time_shift, self.new_cmtsource.time_shift))
            logger.info("hdr:  %15.3f  %15.3f" % (self.cmtsource.half_duration, self.new_cmtsource.half_duration))
        else:
            logger.info("PAR         Old_CMT          New_CMT     Bootstrap_Mean     Bootstrap_STD     STD/Mean")
            logger.info(
                "Mrr:  %15.6e  %15.6e  %15.6e  %15.6e   %10.2f%%" % (self.cmtsource.m_rr, self.new_cmtsource.m_rr,
                                                                     self.par_mean[0], self.par_std[0],
                                                                     self.std_over_mean[0] * 100))
            logger.info(
                "Mtt:  %15.6e  %15.6e  %15.6e  %15.6e   %10.2f%%" % (self.cmtsource.m_tt, self.new_cmtsource.m_tt,
                                                                     self.par_mean[1], self.par_std[1],
                                                                     self.std_over_mean[1] * 100))
            logger.info(
                "Mpp:  %15.6e  %15.6e  %15.6e  %15.6e   %10.2f%%" % (self.cmtsource.m_pp, self.new_cmtsource.m_pp,
                                                                     self.par_mean[2], self.par_std[2],
                                                                     self.std_over_mean[2] * 100))
            logger.info(
                "Mrt:  %15.6e  %15.6e  %15.6e  %15.6e   %10.2f%%" % (self.cmtsource.m_rt, self.new_cmtsource.m_rt,
                                                                     self.par_mean[3], self.par_std[3],
                                                                     self.std_over_mean[3] * 100))
            logger.info(
                "Mrp:  %15.6e  %15.6e  %15.6e  %15.6e   %10.2f%%" % (self.cmtsource.m_rp, self.new_cmtsource.m_rp,
                                                                     self.par_mean[4], self.par_std[4],
                                                                     self.std_over_mean[4] * 100))
            logger.info(
                "Mtp:  %15.6e  %15.6e  %15.6e  %15.6e   %10.2f%%" % (self.cmtsource.m_tp, self.new_cmtsource.m_tp,
                                                                     self.par_mean[5], self.par_std[5],
                                                                     self.std_over_mean[5] * 100))
            logger.info("dep:  %15.3f  %15.3f  %15.3f  %15.3f   %10.2f%%" % (
                self.cmtsource.depth_in_m / 1000.0, self.new_cmtsource.depth_in_m / 1000.0,
                self.par_mean[6], self.par_std[6], self.std_over_mean[6] * 100))
            logger.info("lon:  %15.3f  %15.3f  %15.3f  %15.3f   %10.2f%%" % (
                self.cmtsource.longitude, self.new_cmtsource.longitude,
                self.par_mean[7], self.par_std[7], self.std_over_mean[7] * 100))
            logger.info("lat:  %15.3f  %15.3f  %15.3f  %15.3f   %10.2f%%" % (
                self.cmtsource.latitude, self.new_cmtsource.latitude,
                self.par_mean[8], self.par_std[8], self.std_over_mean[8] * 100))
            logger.info("ctm:  %15.3f  %15.3f  %15.3f  %15.3f   %10.2f%%" % (
                self.cmtsource.time_shift, self.new_cmtsource.time_shift,
                self.par_mean[9], self.par_std[9], self.std_over_mean[9] * 100))
            logger.info("hdr:  %15.3f  %15.3f  %15.3f  %15.3f   %10.2f%%" % (
                self.cmtsource.half_duration, self.new_cmtsource.half_duration,
                self.par_mean[10], self.par_std[10], self.std_over_mean[10] * 100))
