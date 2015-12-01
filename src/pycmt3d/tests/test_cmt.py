import matplotlib as mpl
from pycmt3d.cmt3d import Cmt3D
from pycmt3d.source import CMTSource
from pycmt3d.config import Config
from pycmt3d.window import DataContainer
from pycmt3d.const import PAR_LIST
import os
import inspect

mpl.use('Agg')

basedir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
print "lala:", basedir

event = "010104K"
print "test1"
cmtfile = os.path.join(basedir, "data/CMT/010104K")
original_flexwin_output = os.path.join(basedir, "data/cmt_input/010104K_27_60")


def modify_input_winfile(old_winfile):
    new_winfile = os.path.join(os.path.dirname(old_winfile), "temp.input")
    fh2 = open(new_winfile, 'w')
    with open(old_winfile) as fh1:
        npair = int(fh1.readline())
        fh2.write("%d\n" % npair)
        for ipair in range(npair):
            obs_fn = fh1.readline().rstrip()
            syn_fn = fh1.readline().rstrip()
            nwin = int(fh1.readline().rstrip())
            new_obs_fn = os.path.join(basedir, obs_fn)
            new_syn_fn = os.path.join(basedir, syn_fn)
            fh2.write("%s\n%s\n%d\n" % (
                new_obs_fn, new_syn_fn, nwin))
            for iwin in range(nwin):
                fh2.write("%s" % fh1.readline())
    fh2.close()
    return new_winfile


def test_pycmt3d():
    npar = 9
    cmtsource = CMTSource.from_CMTSOLUTION_file(cmtfile)

    # modify the flexwin output file by adding the absolute path
    flexwin_output = modify_input_winfile(original_flexwin_output)

    data_con = DataContainer(PAR_LIST[0:npar])
    data_con.add_measurements_from_sac(flexwin_output)
    config = Config(npar, dlocation=0.03, ddepth=3.0, dmoment=2.0e+23,
                    weight_data=True, weight_function=None,
                    normalize_window=True, station_correction=True,
                    zero_trace=True, double_couple=False, lamda_damping=0.0,
                    bootstrap=True)

    srcinv = Cmt3D(cmtsource, data_con, config)
    srcinv.source_inversion()
