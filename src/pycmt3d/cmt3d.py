#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from source import CMTSource
import const
from __init__ import logger
import util
from plot_util import PlotUtil, _plot_new_seismogram
from measure import compute_A_b, get_f_df
from plot_stats import _plot_stats_histogram
from data_container import MetaInfo


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

    def __init__(self, cmtsource, data_container, config,
                 logfile="log.txt"):

        self.cmtsource = cmtsource
        self.data_container = data_container
        self.config = config
        self._logfile = logfile

        self.metas = []

        self.trwins = self.data_container.traces

        # new cmt par from the inversion
        self.new_cmt_par = None
        self.new_cmtsource = None

        # category bin
        self.category_bin = None

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

    @property
    def cmt_par(self):
        cmt = self.cmtsource
        self.cmt_par = np.array(
            [cmt.m_rr, cmt.m_tt, cmt.m_pp,
             cmt.m_rt, cmt.m_rp, cmt.m_tp,
             cmt.depth_in_m / 1000.0, cmt.longitude,
             cmt.latitude, cmt.time_shift,
             cmt.half_duration])

    def setup_weight(self):
        """
        Use Window information to setup weight.

        :returns:
        """
        if not self.config.weight_data:
            # no extra weighting, return
            return
        logger.info("*" * 15)
        logger.info("Start weighting...")
        weight_obj = Weight(self.cmtsource, self.data_container,
                            self.metas, self.config.weight_config)
        weight_obj.calculate_weight()

    def setup_matrix(self):
        """
        Calculate A and b for all windows

        :return:
        """
        logger.info("*" * 15)
        logger.info("Set up inversion matrix")

        for trwin in self.trwins:
            metainfo = MetaInfo(obsd_id=trwin.obsd_id, synt_id=trwin.synt_id)
            for win_idx in range(trace.nwindows):
                # loop over each window
                # here, A and b are from raw measurements
                # and no weightings has been applied yet
                A1, b1 = compute_A_b(trace, traces. win_idx,
                                       self.config.parlist,
                                       self.config.dcmt_par)
                metainfo.A1s.append(A1)
                metainfo.b1s.append(b1)

    def sort_category(self):
        """
        Sort self.trwins into different category bins
        """
        def _get_tag(trwin):
            if trwin.tag is None:
                return "%s_%s" % (trwin.tag, trwin.channel)
            else:
                return trwin.channel

        bins = {}
        for idx, trwin in enumerate(self.trwins):
            cat = _get_tag(trwin)
            if cat not in bins:
                bins[cat] = []
            bins[cat].append(idx)
        return bins

    def invert_solver(self, A, b):
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

        # scale the A and b matrix by the max value
        # not really necessary, should be deleted in the future
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
        logger.info("Condition number of A: %10.2f"
                     % np.linalg.cond(A))
        if self.config.damping > 0:
            trace = np.matrix.trace(A)
            damp_matrix = np.zeros([npar, npar])
            np.fill_diagonal(damp_matrix, trace * self.config.lamda_damping)
            A = A + damp_matrix
            logger.info("Condition number of A after damping: %10.2f"
                         % np.linalg.cond(A))

        if linear_inversion:
            logger.info("Linear Inversion...")
            new_par = self.linear_solver(old_par, A, b, npar, na)
        else:
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
        except Exception as err:
            raise ValueError("Can not solve the linear equation due to:%s"
                             % err)
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
            get_f_df(self.config.npar, A, b, m1, lam, mstart, AA, bb)
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

        A1 = np.zeros([self.config.npar, self.config.npar])
        b1 = np.zeros(self.config.npar)
        # ensemble A and b
        for _meta in self.metas:
            A1_trwin = util.sum_matrix(_meta.A1s, coef=_meta.weight)
            b1_trwin = util.sum_matrix(_meta.b1s, coef=_meta.weight)
            A1 += A1_trwin
            b1 += b1_trwin

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
        self.setup_matrix()
        self.sort_category()
        self.setup_weight()
        self.invert_cmt()

        self.calculate_variance()

        #if self.config.bootstrap:
        #    self.invert_bootstrap()

        self.print_inversion_summary()

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

        # calculate metrics for each trwin
        for meta, trwin in zip(self.metas, self.data_container.trwins):
            obsd = trwin.datalist['obsd']
            synt = trwin.datalist['synt']

            # calculate old variance metrics
            [v1, d1, tshift1, cc1, dlnA1, cc_amp1] = \
                calculate_var_on_trace(obsd, synt, trwin.win_time)
            meta.measure["synt"] = {"v": v1, "d": d1, "nshift": nshift1,
                                    "cc": cc1, "dlnA": dlnA1,
                                    "cc_amp": cc_amp1}

            self.compute_new_syn(trwin.datalist, dm)
            new_synt = trwin.datalist['new_synt']
            # calculate new variance metrics
            [v2, d2, tshift2, cc2, dlnA2, cc_amp_value2] = \
                calculate_var_on_trace(obsd, new_synt, trwin.win_time)
            meta.measure["new_synt"] = {"v": v2, "d": d2, "nshift": nshift2,
                                        "cc": cc2, "dlnA": dlnA2,
                                        "cc_amp": cc_amp2}

            var_all += np.sum(0.5 * v1 * meta.weight)
            var_all_new += np.sum(0.5 * v2 * meta.weight)

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
        # datalist['new_synt'].stats.location = \
        #    datalist['obsd'].stats.location

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
    def _write_log_file_(filename, nshift_list, cc_list, dlnA_list):
        """
        Write the nshift, cc, dlnA out
        """
        with open(filename, 'w') as f:
            for i in range(len(nshift_list)):
                nshift = nshift_list[i]
                cc = cc_list[i]
                dlnA = dlnA_list[i]
                f.write("%5d %10.3f %10.3f\n" % (nshift, cc, dlnA))

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

        _plot_stats_histogram(self.stats_before, self.stats_after,
                              figname, figure_format=figure_format)

    def _write_weight_log_(self, filename):
        """
        write out weight log file
        """
        with open(filename, 'w') as f:
            for window in self.data_container.traces:
                sta = window.station
                nw = window.network
                component = window.component
                location = window.location
                sta_info = "%s.%s.%s.%s" % (sta, nw, location, component)
                f.write("%s\n" % sta_info)
                for _idx in range(window.weight.shape[0]):
                    f.write("%10.5e %10.5e\n" % (
                        window.weight[_idx], window.energy[_idx]))

    def plot_new_synt_seismogram(self, outputdir, figure_format="png"):
        """
        Plot the new synthetic waveform
        """
        _plot_new_seismogram(self.data_container, outputdir, self.cmtsource,
                             figure_format=figure_format)

    def write_new_syn(self, outputdir=".", file_format="sac"):
        """
        Write out the new synthetic waveform
        """
        print "New synt output dir: %s" % outputdir
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        if 'new_synt' not in self.data_container.traces[0].datalist.keys():
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
