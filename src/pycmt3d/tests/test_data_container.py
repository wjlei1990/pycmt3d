#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pycmt3d test suite.

Run with pytest.

:copyright:
    Wenjie Lei (lei@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import print_function, division
import inspect
import os
import numpy as np
import obspy
from pycmt3d.data_container import MetaInfo, TraceWindow, DataContainer
import numpy.testing as npt


# Most generic way to get the data folder path.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe()))), "data")
OBSD_DIR = os.path.join(DATA_DIR, "data_T006_T030")
SYNT_DIR = os.path.join(DATA_DIR, "syn_T006_T030")


@pytest.fixture
def meta():
    meta = MetaInfo(obsd_id="CI.GSC..BHZ", synt_id="CI.GSC..BHZ",
                    weight=np.ones(1))
    return meta 


def test_MetaInfo(meta):
    assert meta.obsd_id == "CI.GSC..BHZ"
    assert meta.synt_id == "CI.GSC..BHZ"
    np.assert_allclose(meta.weight, np.ones(1))


def test_TraceWindow():

    obsd = obspy.read(os.path.join(OBSD_DIR, "GSC.CI.BHZ.sac.d"))[0]
    synt = obspy.read(os.path.join(SYNT_DIR, "GSC.CI.BHZ"))[0]
    datalist = {"obsd": obsd, "synt": synt}

    win_time = np.array([[7.545, 46.1450], [50.0, 60.0]])
    weight = np.ones(win_time.shape[0])
    win = TraceWindow(datalist=datalist, windows=win_time, weight=weight,
                      latitude=0.0, longitude=10.0,
                      event_latitude=0.0, event_longitude=0.0,
                      tag="T006_T030", source="SAC")

    assert win.station == "GSC"
    assert win.network == "CI"
    assert win.location == ""
    assert win.channel == "BHZ"
    assert sorted(win.data_keys) == sorted(["obsd", "synt"])
    assert win.full_tag == "T006_T030_BHZ"
    assert win.nwindows == 2
    assert win.obsd_id == "CI.GSC..BHZ"
    assert win.synt_id == "CI.GSC..BHZ"
    npt.assert_allclose(win.azimuth, 90.0)
    npt.assert_allclose(win.back_azimuth, 270.0)
    npt.assert_allclose(win.distance_in_km, 1113.19490793)
    npt.assert_allclose(win.obsd_energy, [1.10586496e-14, 1.24385582e-12])


def test_DataContainer_simple():
    WINDOW_FILE = os.path.join(DATA_DIR, "flexwin_T006_T030.output.test")
    dcon = DataContainer()
    assert dcon.npar == 0
    assert dcon.ntraces == 0
    assert dcon.nwindows == 0

    os.chdir(DATA_DIR)
    dcon.add_measurements_from_sac(WINDOW_FILE, tag="T006_T030")
    assert dcon.npar == 0
    assert dcon.ntraces == 1
    assert dcon.nwindows == 1


def test_load_winfile_txt():
    WINDOW_FILE = os.path.join(DATA_DIR, "flexwin_T006_T030.output.weight")
    dcon = DataContainer()

    traces = dcon.load_winfile(WINDOW_FILE, initial_weight=1.0,
                               file_format="txt")
    assert len(traces) == 2
    npt.assert_allclose(traces[0].windows[0], [7.5450, 46.1450])
    npt.assert_allclose(traces[0].weight[0], 0.50)
    npt.assert_allclose(traces[1].windows[0], [-0.0550, 44.5950])
    npt.assert_allclose(traces[1].windows[1], [44.5950, 76.8450])
    npt.assert_allclose(traces[1].weight[0], 0.60)
    npt.assert_allclose(traces[1].weight[1], 2.00)

    assert dcon._get_counts(traces) == (2, 3)
