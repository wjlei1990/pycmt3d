#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Source and Receiver classes of Instaseis.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
    Martin van Driel (Martin@vanDriel.de), 2014
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lgpl.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import collections
import numpy as np
import obspy
from obspy.core.util.geodetics import FlinnEngdahl
from obspy.signal.filter import lowpass
import obspy.xseed
import os
from scipy import interp
import warnings
from obspy import readEvents


DEFAULT_MU = 32e9


class CMTSource(object):
    """
    Class to handle a seismic moment tensor source including a source time
    function.
    """
    def __init__(self, origin_time=obspy.UTCDateTime(0),
                 pde_latitude=0.0, pde_longitude=0.0, mb=0.0, ms=0.0, pde_depth_in_m=None,
                 region_tag=None, eventname=None, time_shift=None, half_duration=0.0,
                 latitude=0.0, longitude=0.0, depth_in_m=None,
                 m_rr=0.0, m_tt=0.0, m_pp=0.0, m_rt=0.0, m_rp=0.0, m_tp=0.0):
        """
        :param latitude: latitude of the source in degree
        :param longitude: longitude of the source in degree
        :param depth_in_m: source depth in m
        :param m_rr: moment tensor components in r, theta, phi in Nm
        :param m_tt: moment tensor components in r, theta, phi in Nm
        :param m_pp: moment tensor components in r, theta, phi in Nm
        :param m_rt: moment tensor components in r, theta, phi in Nm
        :param m_rp: moment tensor components in r, theta, phi in Nm
        :param m_tp: moment tensor components in r, theta, phi in Nm
        :param time_shift: correction of the origin time in seconds. only
            useful in the context of finite sources
        :param sliprate: normalized source time function (sliprate)
        :param dt: sampling of the source time function
        :param origin_time: The origin time of the source. This will be the
            time of the first sample in the final seismogram. Be careful to
            adjust it for any time shift or STF (de)convolution effects.

        """
        self.origin_time = origin_time
        self.pde_latitude = pde_latitude
        self.pde_longitude = pde_longitude
        self.pde_depth_in_m = pde_depth_in_m
        self.mb = mb
        self.ms = ms
        self.region_tag = region_tag
        self.eventname = eventname
        self.time_shift = time_shift
        self.half_duration = half_duration
        self.latitude = latitude
        self.longitude = longitude
        self.depth_in_m = depth_in_m
        self.m_rr = m_rr
        self.m_tt = m_tt
        self.m_pp = m_pp
        self.m_rt = m_rt
        self.m_rp = m_rp
        self.m_tp = m_tp

    @classmethod
    def from_CMTSOLUTION_file(self, filename):
        """
        Initialize a source object from a CMTSOLUTION file.

        :param filename: path to the CMTSOLUTION file
        """

        with open(filename, "rt") as f:
            line = f.readline()
            origin_time = line[4:].strip().split()[:6]
            values = list(map(int, origin_time[:-1])) + \
                [float(origin_time[-1])]
            try:
                origin_time = obspy.UTCDateTime(*values)
            except (TypeError, ValueError):
                warnings.warn("Could not determine origin time from line: %s"
                              % line)
                origin_time = obspy.UTCDateTime(0)
            otherinfo = line[4:].strip().split()[6:]
            print("otherinfo:", otherinfo)
            pde_lat = float(otherinfo[0])
            pde_lon = float(otherinfo[1])
            pde_depth_in_m = float(otherinfo[2]) * 1e3
            mb = float(otherinfo[3])
            ms = float(otherinfo[4])
            region_tag = ' '.join(otherinfo[5:])

            eventname = f.readline().strip().split()[-1]
            time_shift = float(f.readline().strip().split()[-1])
            half_duration = float(f.readline().strip().split()[-1])
            latitude = float(f.readline().strip().split()[-1])
            longitude = float(f.readline().strip().split()[-1])
            depth_in_m = float(f.readline().strip().split()[-1]) * 1e3

            m_rr = float(f.readline().strip().split()[-1]) / 1e7
            m_tt = float(f.readline().strip().split()[-1]) / 1e7
            m_pp = float(f.readline().strip().split()[-1]) / 1e7
            m_rt = float(f.readline().strip().split()[-1]) / 1e7
            m_rp = float(f.readline().strip().split()[-1]) / 1e7
            m_tp = float(f.readline().strip().split()[-1]) / 1e7

        return self(origin_time=origin_time,
                    pde_latitude=pde_lat, pde_longitude=pde_lon, mb=mb, ms=ms, pde_depth_in_m=pde_depth_in_m,
                    region_tag=region_tag, eventname=eventname, time_shift=time_shift, half_duration=half_duration,
                    latitude=latitude, longitude=longitude, depth_in_m=depth_in_m,
                    m_rr=m_rr, m_tt=m_tt, m_pp=m_pp, m_rt=m_rt, m_rp=m_rp, m_tp=m_tp)

    def write_CMTSOLUTION_file(self, filename):
        """
        Initialize a source object from a CMTSOLUTION file.

        :param filename: path to the CMTSOLUTION file
        """
        with open(filename, "w") as f:
            # Reconstruct the first line as well as possible. All
            # hypocentral information is missing.
            f.write('PDE %4i %2i %2i %2i %2i %5.2f %8.4f %9.4f %5.1f %.1f %.1f'
                    ' %s\n' % (
                        self.origin_time.year,
                        self.origin_time.month,
                        self.origin_time.day,
                        self.origin_time.hour,
                        self.origin_time.minute,
                        self.origin_time.second +
                        self.origin_time.microsecond / 1E6,
                        self.pde_latitude,
                        self.pde_longitude,
                        self.pde_depth_in_m / 1e3,
                        # Just write the moment magnitude twice...we don't
                        # have any other.
                        self.mb,
                        self.ms,
                        self.region_tag))
            f.write('event name:  %s\n' % (self.eventname,))
            f.write('time shift:     %5.4f\n' % (self.time_shift,))
            f.write('half duration:  %5.4f\n' % (self.half_duration,))
            f.write('latitude:       %7.4f\n' % (self.latitude,))
            f.write('longitude:      %7.4f\n' % (self.longitude,))
            f.write('depth:          %7.4f\n' % (self.depth_in_m / 1e3,))

            f.write('Mrr:            %7.4f\n' % (self.m_rr * 1e7,))
            f.write('Mtt:            %7.4f\n' % (self.m_tt * 1e7,))
            f.write('Mpp:            %7.4f\n' % (self.m_pp * 1e7,))
            f.write('Mrt:            %7.4f\n' % (self.m_rt * 1e7,))
            f.write('Mrp:            %7.4f\n' % (self.m_rp * 1e7,))
            f.write('Mtp:            %7.4f\n' % (self.m_tp * 1e7,))

    @classmethod
    def from_quakeml_to_cmt(self, quakemlfile):

        if not os.path.exists(quakemlfile):
            raise IOError("Quakemlfile not exists")

        cat = readEvents(quakemlfile)
        event = cat.events[0]
        PDEsolution = event.origins[0]
        CMTsolution = event.event.preferred_origin()

    @property
    def M0(self):
        """
        Scalar Moment M0 in Nm
        """
        return (self.m_rr ** 2 + self.m_tt ** 2 + self.m_pp ** 2
                + 2 * self.m_rt ** 2 + 2 * self.m_rp ** 2
                + 2 * self.m_tp ** 2) ** 0.5 * 0.5 ** 0.5

    @property
    def moment_magnitude(self):
        """
        Moment magnitude M_w
        """
        return 2.0 / 3.0 * np.log10(self.M0) - 6.0

    @property
    def tensor(self):
        """
        List of moment tensor components in r, theta, phi coordinates:
        [m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]
        """
        return np.array([self.m_rr, self.m_tt, self.m_pp, self.m_rt, self.m_rp,
                         self.m_tp])

    @property
    def tensor_voigt(self):
        """
        List of moment tensor components in theta, phi, r coordinates in Voigt
        notation:
        [m_tt, m_pp, m_rr, m_rp, m_rt, m_tp]
        """
        return np.array([self.m_tt, self.m_pp, self.m_rr, self.m_rp, self.m_rt,
                         self.m_tp])

    def __str__(self):
        return_str = 'Instaseis Source:\n'
        return_str += '\tLongitude        : %6.1f deg\n' % (self.longitude,)
        return_str += '\tLatitude         : %6.1f deg\n' % (self.latitude,)
        return_str += '\tDepth            : %6.1e km\n' \
                      % (self.depth_in_m / 1e3,)
        return_str += '\tMoment Magnitude :   %4.2f\n' \
                      % (self.moment_magnitude,)
        return_str += '\tScalar Moment    : %10.2e Nm\n' % (self.M0,)
        return_str += '\tMrr              : %10.2e Nm\n' % (self.m_rr,)
        return_str += '\tMtt              : %10.2e Nm\n' % (self.m_tt,)
        return_str += '\tMpp              : %10.2e Nm\n' % (self.m_pp,)
        return_str += '\tMrt              : %10.2e Nm\n' % (self.m_rt,)
        return_str += '\tMrp              : %10.2e Nm\n' % (self.m_rp,)
        return_str += '\tMtp              : %10.2e Nm\n' % (self.m_tp,)

        return return_str


