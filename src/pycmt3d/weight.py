#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __init__ import logger
import numpy as np
from spaceweight import SphereAziBin, Point
from .config import WeightConfig
from .constant import REF_DIST
from util import distance, normalize_array


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

        if len(data_container) != len(metas):
            raise ValueError("Length of data_container and metas different")
        self.data_container = data_container
        self.metas = metas

        if not isinstance(config, WeightConfig):
            raise TypeError("Input config must be type of "
                            "pycmt3d.WeightConfig")
        self.config = config

        # keep category information
        self.point_bins = {}

    def setup_weight(self):
        """
        Use Window information to setup weight.

        :returns:
        """
        logger.info("*" * 15)
        logger.info("Start weighting...")

        self.sort_into_category()

        self.setup_weight_for_location()

        if self.config.mode == "default":
            self.setup_weight_for_epicenter_distance()
            # according to original version of cmt3d, which has weighting on:
            # components(Z, R, T)
            self.setup_weight_for_component()

        if self.config.normalize_by_energy:
            self.normalize_weight_by_energy()

        self.normalize_weight()

    def normalize_weight(self):
        """
        Normalize all weight value. Normalize the average weighting to 1.
        """
        weight_sum = 0
        for meta in self.metas:
            weight_sum += sum(meta.weights)
        factor = self.data_container.nwindows / weight_sum
        for meta in self.metas:
            meta.weights *= factor

    def normalize_weight_by_energy(self):
        for meta, trwin in zip(self.metas, self.data_container):
            energy = self._calculate_energy_weighting(trwin, mode="window")
            meta.weight /= energy

    def sort_into_category(self):
        """
        Sort data into different cateogeries, by the trwin.tag and
        trwin.channel. trwin.tag is usually the period band, so
        category would be like "27_60.BHZ", "27_60.BHR", "27_60.BHT",
        and "60_120.BHZ", "60_120.BHR", "60_120.BHT".
        """
        pbins = {}
        for idx, trwin in enumerate(self.data_container):
            if self.normalize_by_category:
                cat = "%s.%s" % (trwin.tag, trwin.channel)
            else:
                cat = "all"
            if cat not in pbins:
                pbins[cat] = []
            pbins.append(Point([trwin.latitude, trwin.longitude], tag=idx))

        self.point_bins = pbins

    def setup_weight_for_component(self):
        for cat, points in self.point_bins:
            comp = cat.split('.')[-1][-1]
            comp_weight = self.config.comp_weight[comp]
            for point in points:
                meta = self.metas[point.tag]
                meta.weight *= comp_weight

    def setup_weight_for_azimuth(self):
        """
        Sort station azimuth into bins and assign weight to each bin
        """
        center = Point([self.cmtsource.latitude,
                        self.cmtsource.longitude],
                       tag="cmtsource")
        weights = {}
        idxs = {}
        for cat, points in self.point_bins:
            weight = SphereAziBin(points, center=center, bin_order=0.5,
                                  nbins=12, remove_duplicate=False)
            weight.calculate_weight()
            for point in points:
                if cat not in weights:
                    weights[cat] = []
                    idxs[cat] = []
                weights[cat].append(point.weight)
                idxs[cat].append(point.tag)
        return weights, idxs

    def setup_weight_for_epicenter_distance(self):
        """
        This is just a courtesy functions which works the same as CMT3D
        distance weighting
        """
        center = Point([self.cmtsource.latitude, self.cmtsource.longitude],
                       tag="source")
        for cat, points in self.point_bins:
            comp = cat.split('.')[-1][-1]
            for point in points:
                meta = self.metas[point.tag]
                dist = distance(center.coordinate[0], center.coordinate[1],
                                point.coordinate[0], point.coordinate[1])
                for i in range(len(meta.weight)):
                    if comp == "T":
                        meta.weight *= \
                            (dist/REF_DIST) ** self.config.love_dist_weight
                    elif i == 0:
                        meta.weight *= \
                            (dist/REF_DIST) ** self.config.pnl_dist_weight
                    else:
                        meta.weight *= \
                            (dist/REF_DIST) ** self.config.rayleigh_dist_weight

    def setup_weight_for_location(self, window):
        """
        setup weight from station location information, including distance,
        component and azimuth. This weight applies on station level.

        :param window:
        :param naz_bin:
        :param naz_bin_all:
        :return:
        """
        # set up weight based on azimuth distribution
        weights, idxs = self.setup_weight_for_azimuth()
        # set up weight based on station locations
        for cat in weights:
            cat_weights = weights[cat]
            # normalized by number of stations
            factor = len(cat_weights) / np.sum(cat_weights)
            cat_weights = normalize_array(cat_weights, factor)
            for _idx, _weight in zip(idxs[cat], weights[cat]):
                self.metas[_idx].weight *= cat_weights
