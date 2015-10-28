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


class Cmt3D(object):
    """
    Class that handles the solver part of source inversion

    :param cmtsource: earthquake source
    :type cmtsource: :class:`pycmt3d.CMTSource`
    :param data_container: all data and window
    :type data_container: :class:`pycmt3d.DataContainer`
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
        self.cmt_par = np.array(
            [cmtsource.m_rr, cmtsource.m_tt, cmtsource.m_pp,
             cmtsource.m_rt, cmtsource.m_rp, cmtsource.m_tp,
             cmtsource.depth_in_m / 1000.0, cmtsource.longitude,
             cmtsource.latitude, cmtsource.time_shift,
             cmtsource.half_duration])
        # new cmt par from the inversion
        self.new_cmt_par = None
        self.new_cmtsource = None

        # azimuth information
        self.naz_files = None
        self.naz_files_all = None
        self.naz_wins = None
        self.naz_wins_all = None

        # category bin
        self.bin_category = None

        # window stats before and after. For plotting purpose,
        # it include nshift, cc, cc amplitude ratio, power ration,
        # and kai of each window.
        self.stats_after = None
        self.stats_before = None

        # variance information
        self.var_all = None
        self.var_all_new = None
        self.var_reduction = None

        # bootstrap stat var
        self.par_mean = np.zeros(const.NPARMAX)
        self.par_std = np.zeros(const.NPARMAX)
        self.par_var = np.zeros(const.NPARMAX)
        self.std_over_mean = np.zeros(self.par_mean.shape)

        self.print_cmtsource_summary(self.cmtsource)

    def setup_weight(self, weight_mode="num_wins"):
        """
        Use Window information to setup weight.

        :returns:
        """
        logger.info("*" * 15)
        logger.info("Start weighting...")
        if self.config.weight_data:
            # first calculate azimuth and distance for each data pair
            self.prepare_for_weighting()
            # then calculate azimuth weighting
            for idx, window in enumerate(self.window):
                if weight_mode.lower() == "num_files":
                    # weighted by the number of files in each azimuth bin
                    self.setup_weight_for_location(window, self.naz_files,
                                                   self.naz_files_all)
                else:
                    # weighted by the number of windows in each azimuth bin
                    self.setup_weight_for_location(window, self.naz_wins,
                                                   self.naz_wins_all)

                if self.config.normalize_category:
                    self.setup_weight_for_category(window)

                if self.config.normalize_window:
                    window.weight = window.weight/window.energy

            # normalization of data weights
            self.normalize_weight()

        # prepare the weight array
        self.weight_array = np.zeros([self.data_container.nwins])
        _idx = 0
        for window in self.window:
            for win_idx in range(window.num_wins):
                self.weight_array[_idx] = window.weight[win_idx]
                _idx += 1

    def setup_weight_for_location(self, window, naz_bin, naz_bin_all):
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
        for window in self.window:
            # calculate energy
            window.win_energy(mode=self.config.norm_mode)
            # calculate location
            window.get_location_info(self.cmtsource)

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
        for window in self.window:
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
        naz_files = {}
        naz_wins = {}
        for window in self.window:
            tag = window.tag['obsd']
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
        for window in self.window:
            max_temp = np.max(window.weight)
            if max_temp > max_weight:
                max_weight = max_temp

        logger.debug("Global Max Weight: %f" % max_weight)

        for window in self.window:
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

    def setup_matrix(self):
        """
        Calculate A and b for all windows

        :return:
        """
        logger.info("*" * 15)
        logger.info("Set up inversion matrix")

        for window in self.window:
            # loop over pair of data
            dsyn = self.calculate_dsyn(window.datalist)
            for win_idx in range(window.num_wins):
                # loop over each window
                # here, A and b are exact measurements
                # and no weightings are applied
                [A1, b1] = self.compute_A_b(window, win_idx, dsyn)
                self.A1_all.append(A1)
                self.b1_all.append(b1)

    def compute_A_b(self, window, win_idx, dsyn):
        """
        Calculate the matrix A and vector b based on one pair of
        observed data and synthetic data on a given window.

        :param window: data and window information
        :type window: :class:`pycmt3d.Window`
        :param win_idx: window index(a specific window)
        :type win_idx: integer
        :param dsyn: derivative synthetic data matrix
        :type dsyn: numpy.array
        :return:
        """
        npar = self.config.npar

        datalist = window.datalist
        obsd = datalist['obsd']
        synt = datalist['synt']
        npts = min(obsd.stats.npts, synt.stats.npts)
        win = [window.win_time[win_idx, 0], window.win_time[win_idx, 1]]

        istart = int(max(math.floor(win[0] / obsd.stats.delta), 1))
        iend = int(min(math.ceil(win[1] / obsd.stats.delta), npts))
        if istart > iend:
            raise ValueError("Check window for %s.%s.%s.%s" %
                             (window.station, window.network,
                              window.location, window.component))

        # station correction
        istart_d, iend_d, istart_s, iend_s, nshift, cc, dlnA, cc_amp_value = \
            self.apply_station_correction(obsd, synt, istart, iend)

        dt_synt = datalist['synt'].stats.delta
        dt_obsd = datalist['obsd'].stats.delta
        if abs(dt_synt - dt_obsd) > 0.0001:
            raise ValueError("Delta in synthetic and observed no the same")
        dt = dt_synt

        # hanning taper
        taper = construct_taper(iend_s - istart_s, taper_type=const.taper_type)

        A1 = np.zeros((npar, npar))
        b1 = np.zeros(npar)
        # compute A and b
        for j in range(npar):
            for i in range(0, j + 1):
                A1[i, j] = np.sum(taper * dsyn[i, istart_s:iend_s] *
                                  dsyn[j, istart_s:iend_s]) * dt
            b1[j] = np.sum(
                taper * (obsd.data[istart_d:iend_d] -
                         synt.data[istart_s:iend_s]) *
                dsyn[j, istart_s:iend_s]) * dt
        for j in range(npar):
            for i in range(j + 1, npar):
                A1[i, j] = A1[j, i]

        return [A1, b1]

    def calculate_dsyn(self, datalist):
        """
        Calculate dsyn matrix based on perturbed seismograms

        :param datalist:
        :return:
        """
        par_list = self.config.par_name
        npar = self.config.npar
        dcmt_par = self.config.dcmt_par
        obsd = datalist['obsd']
        synt = datalist['synt']
        npts = min(obsd.stats.npts, synt.stats.npts)
        dsyn = np.zeros((npar, npts))
        for itype in range(npar):
            type_name = par_list[itype]
            if itype < const.NML:
                # check file: check dt, npts
                dt_synt = datalist['synt'].stats.delta
                dt_obsd = datalist['obsd'].stats.delta
                if abs(dt_synt - dt_obsd) > 0.0001:
                    raise ValueError("Delta in synthetic and observed no "
                                     "the same")
                dt = dt_synt
            if itype < const.NM:  # moment tensor
                dsyn[itype, 0:npts] = \
                    datalist[type_name].data[0:npts] / dcmt_par[itype]
            elif itype < const.NML:  # location
                dsyn[itype, 0:npts] = \
                    (datalist[type_name].data[0:npts] -
                     datalist['synt'].data[0:npts]) / dcmt_par[itype]
            elif itype == const.NML:  # time shift
                dsyn[itype, 0:npts - 1] = \
                    (datalist['synt'].data[1:npts] -
                     datalist['synt'].data[0:(npts - 1)]) \
                    / (dt * dcmt_par[itype])
                dsyn[itype, npts - 1] = dsyn[itype, npts - 2]
            elif itype == const.NML + 1:  # half duration
                dsyn[itype, 0:npts - 1] = -0.5 * self.cmt_par[itype] * (
                    dsyn[const.NML, 1:npts] - dsyn[const.NML, 0:npts - 1]) / dt
                dsyn[itype, npts - 1] = dsyn[itype, npts - 2]
        return dsyn

    def apply_station_correction(self, obsd, synt, istart, iend):
        """
        Apply station correction on windows based one cross-correlation
        time shift if config.station_correction

        :param obsd:
        :param synt:
        :param istart:
        :param iend:
        :return:
        """
        npts = min(obsd.stats.npts, synt.stats.npts)
        [nshift, cc, dlnA] = self.calculate_criteria(obsd, synt, istart, iend)
        if self.config.station_correction:
            istart_d = max(1, istart + nshift)
            iend_d = min(npts, iend + nshift)
            istart_s = istart_d - nshift
            iend_s = iend_d - nshift
            # recalculate the dlnA and cc_amp_value(considering the shift)
            dlnA = \
                self._dlnA_win_(obsd[istart_d:iend_d], synt[istart_s:iend_s])
            cc_amp_value = \
                10 * np.log10(np.sum(obsd[istart_d:iend_d] *
                                     synt[istart_s:iend_s]) /
                              (synt[istart_s:iend_s] ** 2).sum())
        else:
            istart_d = istart
            iend_d = iend
            istart_s = istart
            iend_s = iend
            cc_amp_value = \
                10 * np.log10(np.sum(obsd[istart_d:iend_d] *
                                     synt[istart_s:iend_s]) /
                              (synt[istart_s:iend_s] ** 2).sum())
        return istart_d, iend_d, istart_s, iend_s, nshift, \
            cc, dlnA, cc_amp_value

    def invert_solver(self, A, b, print_mode=False):
        """
        Solver part. Hession matrix A and misfit vector b will be
        reconstructed here based on different constraints.

        :param A: basic Hessian matrix
        :param b: basic misfit vector
        :param print_mode: if True, then print out log information;
        if False, then no log information
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
        if print_mode:
            logger.info("Condition number of new A: %10.2f"
                        % np.linalg.cond(A))

        if linear_inversion:
            if print_mode:
                logger.info("Linear Inversion...")
            new_par = self.linear_solver(old_par, A, b, npar, na)
        else:
            if print_mode:
                logger.info("Nonlinear Inversion...")
            new_par = self.nonlinear_solver(old_par, A, b, npar, na)

        new_cmt_par = np.copy(self.cmt_par)
        new_cmt_par[0:npar] = new_par[0:npar] * self.config.scale_par[0:npar]

        return new_cmt_par

    def linear_solver(self, old_par, A, b, npar, na):
        """
        if invert for moment tensor with zero-trace constraints
        or no constraint
        """
        AA = np.zeros([na, na])
        bb = np.zeros(na)
        AA[0:npar, 0:npar] = A
        bb[0:npar] = b
        if self.config.zero_trace:
            bb[na - 1] = - np.sum(old_par[0:3])
            AA[0:6, na - 1] = np.array([1, 1, 1, 0, 0, 0])
            AA[na - 1, 0:6] = np.array([1, 1, 1, 0, 0, 0])
            AA[na - 1, na - 1] = 0.0
        try:
            dm = np.linalg.solve(AA, bb)
        except:
            logger.error('Matrix is singular...LinearAlgError')
            raise ValueError("Check Matrix Singularity")
        new_par = old_par[0:npar] + dm[0:npar]
        return new_par

    def nonlinear_solver(self, old_par, A, b, npar, na):
        """
        if invert for moment tensor with double couple constraints
        setup starting solution, solve directly for moment instead
        of dm, exact implementation of (A16)
        logger.info('Non-linear Inversion')

        :return:
        """
        mstart = np.copy(old_par)
        m1 = np.copy(mstart)
        lam = np.zeros(2)
        AA = np.zeros([na, na])
        bb = np.zeros(na)

        error = np.zeros([const.NMAX_NL_ITER, na])
        for iter_idx in range(const.NMAX_NL_ITER):
            self._get_f_df_(A, b, m1, lam, mstart, AA, bb)
            bb = - bb
            xout = np.linalg.solve(AA, bb)
            m1 = m1 + xout[0:npar]
            lam = lam + xout[npar:na]
            error[iter_idx, :] = np.dot(AA, xout) - bb
        # dm = m1 - mstart
        return m1

    def invert_cmt(self):
        """
        ensemble all measurements together to form Matrix A and vector
        b to solve the A * (dm) = b
        A is the Hessian Matrix and b is the misfit

        :return:
        """
        logger.info("*"*15)
        logger.info("CMT Inversion")
        logger.info("*"*15)
        # ensemble A and b
        A = util.sum_matrix(self.weight_array, self.A1_all)
        b = util.sum_matrix(self.weight_array, self.b1_all)
        logger.info("Inversion Matrix A is as follows:")
        logger.info("\n%s" % ('\n'.join(map(self._float_array_to_str, A))))
        logger.info("Condition number of A: %10.2f" % (np.linalg.cond(A)))
        logger.info("RHS vector b is as follows:")
        logger.info("[%s]" % (self._float_array_to_str(b)))

        # source inversion
        self.new_cmt_par = self.invert_solver(A, b, print_mode=True)
        self.convert_new_cmt_par()

    def invert_bootstrap(self):
        """
        It is used to evaluate the mean, standard deviation, and variance
        of new parameters

        :return:
        """
        A_bootstrap = []
        b_bootstrap = []
        n_subset = int(const.BOOTSTRAP_SUBSET_RATIO * self.nwins)
        for i in range(self.config.bootstrap_repeat):
            random_array = util.gen_random_array(
                self.nwins, sample_number=n_subset)
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
        for _ii in range(self.par_mean.shape[0]):
            if self.par_mean[_ii] != 0:
                # in case of 0 value
                self.std_over_mean[_ii] = \
                        np.abs(self.par_std[_ii] / self.par_mean[_ii])
            else:
                self.std_over_mean[_ii] = 0.

    def source_inversion(self):
        """
        the Source Inversion method
        :return:
        """
        from __init__ import logfilename
        print "*"*40 + "\nSee detailed output in %s\n" % logfilename

        self.setup_matrix()
        self.setup_weight(weight_mode=self.config.weight_azi_mode)
        self.invert_cmt()

        self.calculate_variance()

        if self.config.bootstrap:
            self.invert_bootstrap()

        self.print_inversion_summary()

    def _get_f_df_(self, A, b, m, lam, mstart, fij, f0):
        """
        Iterative solver for Non-linear case(double-couple constraint)

        :param A: basic Hessian matrix
        :param b: basic misfit vector
        :param m: current source array
        :param lam: constraints coefficient for zero-trace and
        double-couple constraints
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
        f0[0:npar] = np.dot(A[0:npar, 0:npar], m[0:npar] -
                            mstart[0:npar]) - b[0:npar]
        # print "f0 step1:", f0
        f0[0:const.NM] += \
            lam[0] * dc1_dm[0:const.NM] + lam[1] * dc2_dm[0:const.NM]
        # f_(n+1) and f_(n+2)
        f0[npar] = m[0] + m[1] + m[2]
        moment_tensor = np.array([[m[0], m[3], m[4]],
                                  [m[3], m[1], m[5]], [m[4], m[5], m[2]]])
        f0[npar + 1] = np.linalg.det(moment_tensor)
        f0[npar + 1] = m[0] * (m[1] * m[2] - m[5] ** 2) \
            - m[3] * (m[3] * m[2] - m[5] * m[4]) \
            + m[4] * (m[3] * m[5] - m[4] * m[1])

        # Y_jk
        dc2_dmi_dmj = np.zeros([6, 6])
        dc2_dmi_dmj[0, :] = np.array([0.0, m[2], m[1], 0.0, 0.0, -2.0 * m[5]])
        dc2_dmi_dmj[1, :] = np.array([m[2], 0.0, m[0], 0.0, -2.0 * m[4], 0.0])
        dc2_dmi_dmj[2, :] = np.array([m[1], m[0], 0.0, -2.0 * m[3], 0.0, 0.0])
        dc2_dmi_dmj[3, :] = np.array([0.0, 0.0, -2.0 * m[3], -2.0 * m[2],
                                      2 * m[5], 2 * m[4]])
        dc2_dmi_dmj[4, :] = np.array([0.0, -2.0 * m[4], 0.0, 2.0 * m[5],
                                      -2.0 * m[1], 2 * m[3]])
        dc2_dmi_dmj[5, :] = np.array([-2.0 * m[5], 0.0, 0.0, 2.0 * m[4],
                                      2.0 * m[3], -2.0 * m[0]])

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

        self.stats_before = {}
        self.stats_after = {}
        for _idx, window in enumerate(self.window):
            obsd = window.datalist['obsd']
            synt = window.datalist['synt']
            dt = obsd.stats.delta
            self.compute_new_syn(window.datalist, dm)
            new_synt = window.datalist['new_synt']
            # calculate old variance
            [v1, d1, nshift1, cc1, dlnA1, cc_amp_value1] = \
                self.calculate_var_one_trace(obsd, synt, window.win_time)
            # calculate new variance
            [v2, d2, nshift2, cc2, dlnA2, cc_amp_value2] = \
                self.calculate_var_one_trace(obsd, new_synt, window.win_time)

            var_all += np.sum(0.5 * v1 * window.weight * obsd.stats.delta)
            var_all_new += np.sum(0.5 * v2 * window.weight * obsd.stats.delta)

            # prepare stats
            tag = window.tag['obsd']
            if tag not in self.stats_before.keys():
                self.stats_before[tag] = []
            if tag not in self.stats_after.keys():
                self.stats_after[tag] = []
            for _i in range(window.num_wins):
                self.stats_before[tag].append(
                    [nshift1[_i]*dt, cc1[_i], dlnA1[_i], cc_amp_value1[_i],
                     v1[_i]/d1[_i]])
                self.stats_after[tag].append(
                    [nshift2[_i]*dt, cc2[_i], dlnA2[_i], cc_amp_value2[_i],
                     v2[_i]/d2[_i]])

        for tag in self.stats_before.keys():
            self.stats_before[tag] = np.array(self.stats_before[tag])
            self.stats_after[tag] = np.array(self.stats_after[tag])

        logger.info(
            "Total Variance Reduced from %e to %e ===== %f %%"
            % (var_all, var_all_new, (var_all - var_all_new) / var_all * 100))
        self.var_all = var_all
        self.var_all_new = var_all_new
        self.var_reduction = (var_all - var_all_new) / var_all

    def calculate_kai_total_value(self):
        """
        Calculate the sum of kai value

        :return:
        """
        kai_before = {}
        kai_after = {}
        for tag in self.stats_before.keys():
            kai_before[tag] = np.sum(self.stats_before[tag][:, -1])
            kai_after[tag] = np.sum(self.stats_after[tag][:, -1])
        return kai_before, kai_after

    def calculate_var_one_trace(self, obsd, synt, win_time):
        """
        Calculate the variance reduction on a pair of obsd and
        synt and windows

        :param obsd: observed data trace
        :type obsd: :class:`obspy.core.trace.Trace`
        :param synt: synthetic data trace
        :type synt: :class:`obspy.core.trace.Trace`
        :param win_time: [win_start, win_end]
        :type win_time: :class:`list` or :class:`numpy.array`
        :return:  waveform misfit reduction and observed data
        energy [v1, d1]
        :rtype: [float, float]
        """
        num_wins = win_time.shape[0]
        v1 = np.zeros(num_wins)
        d1 = np.zeros(num_wins)
        nshift_array = np.zeros(num_wins)
        cc_array = np.zeros(num_wins)
        dlnA_array = np.zeros(num_wins)
        cc_amp_value_array = np.zeros(num_wins)
        for _win_idx in range(win_time.shape[0]):
            tstart = win_time[_win_idx, 0]
            tend = win_time[_win_idx, 1]
            idx_start = int(max(math.floor(tstart / obsd.stats.delta), 1))
            idx_end = \
                int(min(math.ceil(tend / obsd.stats.delta), obsd.stats.npts))

            istart_d, iend_d, istart, iend, nshift, cc, dlnA, cc_amp_value = \
                self.apply_station_correction(obsd, synt, idx_start, idx_end)

            taper = construct_taper(iend - istart, taper_type=const.taper_type)
            v1[_win_idx] = \
                np.sum(taper * (synt.data[istart:iend] -
                                obsd.data[istart_d:iend_d]) ** 2)
            d1[_win_idx] = np.sum(taper * obsd.data[istart_d:iend_d] ** 2)
            nshift_array[_win_idx] = nshift
            cc_array[_win_idx] = cc
            dlnA_array[_win_idx] = dlnA
            cc_amp_value_array[_win_idx] = cc_amp_value
        return [v1, d1, nshift_array, cc_array, dlnA_array, cc_amp_value_array]

    def compute_new_syn(self, datalist, dm):
        """
        Compute new synthetic data based on new CMTSOLUTION

        :param datalist: dictionary of all data
        :param dm: CMTSolution perterbation, i.e.,
        (self.new_cmt_par-self.cmt_par)
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
                dsyn[:, i] = (datalist[par_list[i]].data -
                              datalist['synt'].data) / dcmt_par[i]
            elif i == const.NML:
                dsyn[0:(npts - 1), i] = \
                    -(datalist['synt'].data[1:npts] -
                      datalist[0:(npts - 1)]) / (dt * dcmt_par[i])
                dsyn[npts - 1, i] = dsyn[npts - 2, i]
            elif i == (const.NML + 1):
                # not implement yet....
                raise ValueError("For npar == 10 or 11, not implemented yet")

        datalist['new_synt'].data = \
            datalist['synt'].data + np.dot(dsyn, dm_scaled)

    def write_new_syn(self, outputdir=".", file_format="sac"):
        # check first
        print "New synt output dir: %s" % outputdir
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        if 'new_synt' not in self.window[0].datalist.keys():
            raise ValueError("new synt not computed yet")

        eventname = self.cmtsource.eventname
        if self.config.double_couple:
            constr_str = "ZT_DC"
        elif self.config.zero_trace:
            constr_str = "ZT"
        else:
            constr_str = "no_constr"
        suffix = "%dp_%s" % (self.config.npar, constr_str)

        self.data_container.write_new_syn_file(
            file_format=file_format, outputdir=outputdir, eventname=eventname,
            suffix=suffix)

    def calculate_criteria(self, obsd, synt, istart, iend):
        """
        Calculate the time shift, max cross-correlation value and
        energy differnce

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
        dlnA = self._dlnA_win_(obsd_trace, synt_trace)

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
        self.new_cmtsource = CMTSource(
            origin_time=oldcmt.origin_time,
            pde_latitude=oldcmt.pde_latitude,
            pde_longitude=oldcmt.pde_longitude,
            mb=oldcmt.mb, ms=oldcmt.ms, pde_depth_in_m=oldcmt.pde_depth_in_m,
            region_tag=oldcmt.region_tag, eventname=oldcmt.eventname,
            cmt_time=new_cmt_time, half_duration=newcmt[10],
            latitude=newcmt[8], longitude=newcmt[7],
            depth_in_m=newcmt[6] * 1000.0,
            m_rr=newcmt[0], m_tt=newcmt[1], m_pp=newcmt[2], m_rt=newcmt[3],
            m_rp=newcmt[4], m_tp=newcmt[5])

    def write_new_cmtfile(self, outputdir="."):
        """
        Write new_cmtsource into a file
        """
        if self.config.double_couple:
            suffix = "ZT_DC"
        elif self.config.zero_trace:
            suffix = "ZT"
        else:
            suffix = "no_constraint"
        outputfn = "%s.%dp_%s.inv" % (
            self.cmtsource.eventname, self.config.npar, suffix)
        cmtfile = os.path.join(outputdir, outputfn)
        print "New cmt file: %s" % cmtfile

        self.new_cmtsource.write_CMTSOLUTION_file(cmtfile)

    @staticmethod
    def _xcorr_win_(obsd, synt):
        cc = np.correlate(obsd, synt, mode="full")
        nshift = cc.argmax() - len(obsd) + 1
        # Normalized cross correlation.
        max_cc_value = \
            cc.max() / np.sqrt((synt ** 2).sum() * (obsd ** 2).sum())
        return max_cc_value, nshift

    @staticmethod
    def _dlnA_win_(obsd, synt):
        return 10 * np.log10(np.sum(obsd ** 2) / np.sum(synt ** 2))

    @staticmethod
    def print_cmtsource_summary(cmt):
        """
        Print CMTSolution source summary

        :return:
        """
        logger.info("=" * 10 + "  Event Summary  " + "=" * 10)
        logger.info("Event name: %s" % cmt.eventname)
        logger.info("   Latitude and longitude: %.2f, %.2f" % (
            cmt.latitude, cmt.longitude))
        logger.info("   Depth: %.1f km" % (cmt.depth_in_m / 1000.0))
        logger.info("   Region tag: %s" % cmt.region_tag)
        logger.info("   Trace: %.3e" % (
            (cmt.m_rr + cmt.m_tt + cmt.m_pp) / cmt.M0))
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

    @staticmethod
    def _write_log_file_(filename, nshift_list, cc_list, dlnA_list):
        with open(filename, 'w') as f:
            for i in range(len(nshift_list)):
                nshift = nshift_list[i]
                cc = cc_list[i]
                dlnA = dlnA_list[i]
                f.write("%5d %10.3f %10.3f\n" % (nshift, cc, dlnA))

    def print_inversion_summary(self):
        """
        Print out the inversion summary

        :return:
        """
        logger.info("*" * 20)
        logger.info("Invert cmt parameters(%d par)" % self.config.npar)

        logger.info("Old CMT par: [%s]" % (
            ', '.join(map(str, self.cmt_par))))
        logger.info("dm: [%s]" % (
            ', '.join(map(str, self.new_cmt_par - self.cmt_par))))
        logger.info("New CMT par: [%s]" % (
            ', '.join(map(str, self.new_cmt_par))))

        logger.info("Trace: %e" % (np.sum(self.new_cmt_par[0:3])))
        logger.info("Energy change(scalar moment): %5.2f%%" % (
            (self.new_cmtsource.M0 - self.cmtsource.M0) /
            self.cmtsource.M0 * 100.0))

        self.inversion_result_table()

    def inversion_result_table(self):
        """
        Print out the inversion table

        :return:
        """
        title = "*" * 20 + " Inversion Result Table(%d npar) " % \
            self.config.npar + "*" * 20
        logger.info(title)

        if not self.config.bootstrap:
            logger.info("PAR         Old_CMT        New_CMT")
            logger.info("Mrr:  %15.6e  %15.6e" % (
                self.cmtsource.m_rr, self.new_cmtsource.m_rr))
            logger.info("Mtt:  %15.6e  %15.6e" % (
                self.cmtsource.m_tt, self.new_cmtsource.m_tt))
            logger.info("Mpp:  %15.6e  %15.6e" % (
                self.cmtsource.m_pp, self.new_cmtsource.m_pp))
            logger.info("Mrt:  %15.6e  %15.6e" % (
                self.cmtsource.m_rt, self.new_cmtsource.m_rt))
            logger.info("Mrp:  %15.6e  %15.6e" % (
                self.cmtsource.m_rp, self.new_cmtsource.m_rp))
            logger.info("Mtp:  %15.6e  %15.6e" % (
                self.cmtsource.m_tp, self.new_cmtsource.m_tp))
            logger.info(
                "dep:  %15.3f  %15.3f" % (
                    self.cmtsource.depth_in_m / 1000.0,
                    self.new_cmtsource.depth_in_m / 1000.0))
            logger.info("lon:  %15.3f  %15.3f" % (
                self.cmtsource.longitude, self.new_cmtsource.longitude))
            logger.info("lat:  %15.3f  %15.3f" % (
                self.cmtsource.latitude, self.new_cmtsource.latitude))
            logger.info("ctm:  %15.3f  %15.3f" % (
                self.cmtsource.time_shift, self.new_cmtsource.time_shift))
            logger.info("hdr:  %15.3f  %15.3f" % (
                self.cmtsource.half_duration,
                self.new_cmtsource.half_duration))
        else:
            logger.info("PAR         Old_CMT          New_CMT     "
                        "Bootstrap_Mean     Bootstrap_STD     STD/Mean")
            logger.info(
                "Mrr:  %15.6e  %15.6e  %15.6e  %15.6e   %10.2f%%" % (
                    self.cmtsource.m_rr, self.new_cmtsource.m_rr,
                    self.par_mean[0], self.par_std[0],
                    self.std_over_mean[0] * 100))
            logger.info(
                "Mtt:  %15.6e  %15.6e  %15.6e  %15.6e   %10.2f%%" % (
                    self.cmtsource.m_tt, self.new_cmtsource.m_tt,
                    self.par_mean[1], self.par_std[1],
                    self.std_over_mean[1] * 100))
            logger.info(
                "Mpp:  %15.6e  %15.6e  %15.6e  %15.6e   %10.2f%%" % (
                    self.cmtsource.m_pp, self.new_cmtsource.m_pp,
                    self.par_mean[2], self.par_std[2],
                    self.std_over_mean[2] * 100))
            logger.info(
                "Mrt:  %15.6e  %15.6e  %15.6e  %15.6e   %10.2f%%" % (
                    self.cmtsource.m_rt, self.new_cmtsource.m_rt,
                    self.par_mean[3], self.par_std[3],
                    self.std_over_mean[3] * 100))
            logger.info(
                "Mrp:  %15.6e  %15.6e  %15.6e  %15.6e   %10.2f%%" % (
                    self.cmtsource.m_rp, self.new_cmtsource.m_rp,
                    self.par_mean[4], self.par_std[4],
                    self.std_over_mean[4] * 100))
            logger.info(
                "Mtp:  %15.6e  %15.6e  %15.6e  %15.6e   %10.2f%%" % (
                    self.cmtsource.m_tp, self.new_cmtsource.m_tp,
                    self.par_mean[5], self.par_std[5],
                    self.std_over_mean[5] * 100))
            logger.info("dep:  %15.3f  %15.3f  %15.3f  %15.3f   %10.2f%%" % (
                self.cmtsource.depth_in_m / 1000.0,
                self.new_cmtsource.depth_in_m / 1000.0,
                self.par_mean[6], self.par_std[6],
                self.std_over_mean[6] * 100))
            logger.info("lon:  %15.3f  %15.3f  %15.3f  %15.3f   %10.2f%%" % (
                self.cmtsource.longitude, self.new_cmtsource.longitude,
                self.par_mean[7], self.par_std[7],
                self.std_over_mean[7] * 100))
            logger.info("lat:  %15.3f  %15.3f  %15.3f  %15.3f   %10.2f%%" % (
                self.cmtsource.latitude, self.new_cmtsource.latitude,
                self.par_mean[8], self.par_std[8],
                self.std_over_mean[8] * 100))
            logger.info("ctm:  %15.3f  %15.3f  %15.3f  %15.3f   %10.2f%%" % (
                self.cmtsource.time_shift, self.new_cmtsource.time_shift,
                self.par_mean[9], self.par_std[9],
                self.std_over_mean[9] * 100))
            logger.info("hdr:  %15.3f  %15.3f  %15.3f  %15.3f   %10.2f%%" % (
                self.cmtsource.half_duration, self.new_cmtsource.half_duration,
                self.par_mean[10], self.par_std[10],
                self.std_over_mean[10] * 100))

    def plot_summary(self, outputdir=".", figure_format="png",
                     plot_mode="regional"):
        """
        Plot inversion summary

        :param outputdir: output directory
        :return:
        """
        eventname = self.cmtsource.eventname
        npar = self.config.npar
        if self.config.double_couple:
            suffix = "ZT_DC"
        elif self.config.zero_trace:
            suffix = "ZT"
        else:
            suffix = "no_constraint"
        outputfn = "%s.%dp_%s.inv" % (eventname, npar, suffix)
        outputfn = os.path.join(outputdir, outputfn)
        figurename = outputfn + "." + figure_format

        print "Source inversion summary figure: %s" % figurename

        plot_stat = PlotUtil(
            data_container=self.data_container, config=self.config,
            cmtsource=self.cmtsource, nregions=const.NREGIONS,
            new_cmtsource=self.new_cmtsource, bootstrap_mean=self.par_mean,
            bootstrap_std=self.par_std, var_reduction=self.var_reduction,
            mode=plot_mode)
        plot_stat.plot_inversion_summary(figurename=figurename)

    def plot_stats_histogram(self, outputdir=".", figure_format="png"):
        """
        Plot inversion histogram

        :param outputdir:
        :return:
        """
        nrows = len(self.stats_before.keys())
        ncols = self.stats_before[self.stats_before.keys()[0]].shape[1]

        if self.config.double_couple:
            constr_str = "ZT_DC"
        elif self.config.zero_trace:
            constr_str = "ZT"
        else:
            constr_str = "no_constraint"

        if not self.config.normalize_window:
            prefix = "%dp_%s." % (self.config.npar, constr_str) + "no_normwin"
        else:
            prefix = "%dp_%s.%s" % (
                self.config.npar, constr_str, self.config.norm_mode)

        if not self.config.normalize_category:
            prefix += ".no_normcat"
        else:
            prefix += ".normcat"
        figname = "%s.%s.dlnA.%s" % (self.cmtsource.eventname, prefix,
                                     figure_format)
        figname = os.path.join(outputdir, figname)

        print "Inversion histogram figure: %s" % figname

        plt.figure(figsize=(5*ncols, 5*nrows))
        G = gridspec.GridSpec(nrows, ncols)
        irow = 0
        for cat in self.stats_before.keys():
            self._plot_stats_histogram_per_cat_(
                G, irow, cat, self.stats_before[cat], self.stats_after[cat])
            irow += 1
        plt.savefig(figname)

    def _plot_stats_histogram_per_cat_(self, G, irow, cat, data_before,
                                       data_after):
        num_bins = [15, 15, 15, 15, 15]
        vtype_list = ['time shift', 'cc', 'Power_Ratio(dB)',
                      'CC Amplitude Ratio(dB)', 'Kai']
        # plot order
        var_index = [0, 1, 2, 3, 4]
        for _idx, var_idx in enumerate(var_index):
            vtype = vtype_list[var_idx]
            self._plot_stats_histogram_one_(
                G[irow, _idx], cat, vtype, data_before[:, var_idx],
                data_after[:, var_idx],
                num_bins[var_idx])

    @staticmethod
    def _plot_stats_histogram_one_(pos, cat, vtype, data_b, data_a, num_bin):
        plt.subplot(pos)
        plt.xlabel(vtype)
        plt.ylabel(cat)
        if vtype == "cc":
            ax_min = min(min(data_b), min(data_a))
            ax_max = max(max(data_b), max(data_a))
        elif vtype == "Kai":
            ax_min = 0.0
            ax_max = max(max(data_b), max(data_a))
        else:
            ax_min = min(min(data_b), min(data_a))
            ax_max = max(max(data_b), max(data_a))
            abs_max = max(abs(ax_min), abs(ax_max))
            ax_min = -abs_max
            ax_max = abs_max
        binwidth = (ax_max - ax_min) / num_bin
        plt.hist(
            data_b, bins=np.arange(ax_min, ax_max+binwidth/2., binwidth),
            facecolor='blue', alpha=0.3)
        plt.hist(
            data_a, bins=np.arange(ax_min, ax_max+binwidth/2., binwidth),
            facecolor='green', alpha=0.5)

    def _write_weight_log_(self, filename):
        """
        write out weight log file
        """
        with open(filename, 'w') as f:
            for window in self.window:
                sta = window.station
                nw = window.network
                component = window.component
                location = window.location
                sta_info = "%s.%s.%s.%s" % (sta, nw, location, component)
                f.write("%s\n" % sta_info)
                for _idx in range(window.weight.shape[0]):
                    f.write("%10.5e %10.5e\n" % (
                        window.weight[_idx], window.energy[_idx]))

    def plot_new_seismogram(self, outputdir=".", figure_format="png"):
        """
        Plot the new synthetic and old synthetic data together with data
        """
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        # make a check
        if 'new_synt' not in self.window[0].datalist.keys():
            return "New synt not generated...Can't plot"
        else:
            print "Plotting data, synthetics and windows to dir: %s" % \
                  outputdir

        for window in self.window:
            self.plot_new_seismogram_sub(window, outputdir, figure_format)

    def plot_new_seismogram_sub(self, window, outputdir, figure_format):
        obsd = window.datalist['obsd']
        synt = window.datalist['synt']
        new_synt = window.datalist['new_synt']

        station = obsd.stats.station
        network = obsd.stats.network
        channel = obsd.stats.channel
        location = obsd.stats.location
        outputfig = os.path.join(outputdir, "%s.%s.%s.%s.%s" % (
            network, station, location, channel, figure_format))

        offset = self.cmtsource.cmt_time - obsd.stats.starttime
        times = [offset + obsd.stats.delta*i for i in range(obsd.stats.npts)]

        plt.figure(figsize=(15, 2.5))
        plt.plot(times, obsd.data, color="black")
        plt.plot(times, synt.data, color="red")
        plt.plot(times, new_synt.data, color="green")
        plt.xlim(times[0], times[-1])

        for win in window.win_time:
            l = win[0] - offset
            r = win[1] - offset
            re = Rectangle((l, plt.ylim()[0]), r - l,
                           plt.ylim()[1] - plt.ylim()[0], color="blue",
                           alpha=0.25)
            plt.gca().add_patch(re)

        plt.savefig(outputfig)
