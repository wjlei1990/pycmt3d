from cmt3d import Cmt3D
from source import CMTSource
from config import Config
from window import *
from const import PAR_LIST

eventname = "010304A"
cmt_suffix = ".cmt_input"

cmtfile = "/home/lei/DATA/CMT_BIN/from_quakeml/" + eventname
cmtsource = CMTSource.from_CMTSOLUTION_file(cmtfile)
#print cmtsource.depth_in_m

config = Config(9, dlocation=0.03, dmoment=2.0e23, ddepth=3.0,
                double_couple=True, station_correction=True,
               bootstrap=True, bootstrap_repeat=200)

flexwinfile = "/home/lei/DATA/window/cmt3d_input/" + eventname + cmt_suffix
data_con = DataContainer(flexwinfile, PAR_LIST[0:9])

testcmt = Cmt3D(cmtsource, data_con, config)

testcmt.source_inversion()

