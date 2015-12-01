#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration object for pycmt3d
"""

try:
    import numpy as np
except:
    msg = ("No module named numpy. "
           "Please install numpy first, it is needed before using pycmt3d.")
    raise ImportError(msg)
import const


def default_weight_function(kcmpnm, dist_in_km, azi_count, nwins,
                            comp_r_weight=2.0, comp_t_weight=2.0,
                            comp_z_weight=2.0, az_exp_weight=0.5,
                            pnl_dist_weight=1.15, rayleigh_dist_weight=0.55,
                            love_dist_weight=0.78,
                            dist_weight_mode="exponential"):

    """
    Defualt weighting function

    :param kcmpnm:
    :param dist_in_km:
    :param azi_count:
    :param nwins:
    :param comp_r_weight:
    :param comp_t_weight:
    :param comp_z_weight:
    :param az_exp_weight:
    :param pnl_dist_weight:
    :param rayleigh_dist_weight:
    :param love_dist_weight:
    :return:
    """

    data_weight = np.zeros(nwins)
    # component weight
    comp_direct = kcmpnm[2]
    if comp_direct == 'Z':
        cmp_weight = comp_z_weight
    elif comp_direct == 'R':
        cmp_weight = comp_r_weight
    elif comp_direct == 'T':
        cmp_weight = comp_t_weight
    else:
        raise ValueError('The direction of component of seismic data has '
                         'to be Z, R or T')

    # distance weights
    # for global seismograms, this obviously has to be changed
    for win_idx in range(nwins):
        if comp_direct == 'T':
            dist_exp_weight = love_dist_weight
        else:
            if nwins > 1 and win_idx == 0:
                dist_exp_weight = pnl_dist_weight
            else:
                dist_exp_weight = rayleigh_dist_weight

        if dist_weight_mode.lower() == "exponential":
            # exponential weight on distance
            data_weight[win_idx] = \
                cmp_weight * ((dist_in_km/const.REF_DIST) ** dist_exp_weight) \
                / (azi_count ** az_exp_weight)
        elif dist_weight_mode.lower() == "uniform":
            # no distance weighting
            data_weight[win_idx] = cmp_weight / (azi_count ** az_exp_weight)
        elif dist_weight_mode.lower() == "damping":
            # damping over short distance and long distance
            data_weight[win_idx] = \
                cmp_weight / (azi_count ** az_exp_weight) * \
                dist_damping_function(dist_in_km)

    return data_weight


def dist_damping_function(dist_in_km):
    geo_degree = dist_in_km/112  # 111.325km = 1 degree
    if geo_degree <= 60:
        return geo_degree/60.0
    elif geo_degree <= 120:
        return 1.0
    elif geo_degree <= 180:
        return (180.0 - geo_degree)/60.0
    else:
        return 0
