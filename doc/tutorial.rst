Tutourial
==========================================

To run the inversion, one need to prepare:


1. Original CMTSource: CMTSource used as starting solution(Ref :ref:`my-source-label`).

2. Data: including observed data, synthetic data, derived synthetic data and associated windows(Ref :ref:`my-window-label`)
  
3. Inversion shema: how to inversion is done(Ref :ref:`my-config-label`)

CMTSource
#########################################
Source instance could be loaded as::

  import source
  cmtfile = "path/to/your/cmtfile"
  cmtsource = CMTSource.from_CMTSOLUTION_file(cmtfile)

Data
########################################
The pycmt3d pacakge takes two kinds of data.

1. sac format data

  Then the setup is exactly as the original fortran version. The pycmt3d needs the outputfile from flexwin which contains observed and synthetic data filename and associated windows. Another thing it needs to know is how many number of deriv synthetic files you want to read. The deriv parameter is listed as [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp, depth, longitude, latitude, time shift, hald duration].
  Data could be loaded as::

    from window import *
    from const import PAR_LIST
    npar = 6
    flexwin_output = "path/to/your/flexwin_output"
    data = DataContainer(flexwin_output, PAR_LIST[:npar])

  if npar is 6, only the first 6 deriv parameters will be read in, i.e, [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp]. If you want to try different Inversion shema, a larger number of npar is expected. Thus, you can read all necessary deriv synthetic data into the memory and don't need to load it again. For example, I usually choose npar to be 9, which contains moment tensor, depth, longitude and latitude.

2. ASDF format data

   it will be added later on...

Inversion schema
#########################################
Works partially as the INVERSION.PAR file as the fortran version.

One config example is to 
1. invert 9 parameters(moment tensor + depth + location), with location perturbation 0.03 degree, depth perturbation 3.0km and moment perturbation 2.0e23. 
2. Weighting will be applied and the no weighting function specified(default weighting function used).
3. Station correction will be applied
4. Constrain includes zero trace but no double couple.
5. Damping set to 0(no damping)
6. Bootstrap will not be used.

Code example as following::

  from config import Config
  npar = 9   # 9 paramter inversion
  config = Config(npar, dlocation=0.03, ddepth=3.0, dmoment=2.0e+23,
      weight_data=True, weight_function=None, station_correction=True, 
      zero_trace=True, double_couple=False, lamda_damping=0.0, 
      bootstrap=False)

Source Inversion
########################################
After get the CMTSource, Data and Inversion scheme ready, the source inversion can then be conducted::

  from cmt3d import Cmt3D
  srcinv = Cmt3D(cmtsource, data.window, config)
  srcinv.source_inversion()

Workflow Example
########################################
The complete workflow example is shown below::

  import source
  from window import *
  from config import Config
  from cmt3d import Cmt3D

  # load cmtsource
  cmtfile = "path/to/your/cmtfile"
  cmtsource = CMTSource.from_CMTSOLUTION_file(cmtfile)

  # load data and window from flexwin output file
  from const import PAR_LIST
  data_npar = 9 # read 9 deriv synthetic
  flexwin_output = "path/to/your/flexwin_output"
  data = DataContainer(flexwin_output, PAR_LIST[:data_npar])
  
  # inversion shema
  npar = 9   # 9 paramter inversion
  config = Config(npar, dlocation=0.03, ddepth=3.0, dmoment=2.0e+23,
      weight_data=True, weight_function=None, station_correction=True, 
      zero_trace=True, double_couple=False, lamda_damping=0.0, 
      bootstrap=False)

  # source inversion
  srcinv = Cmt3D(cmtsource, data.window, config)
  srcinv.source_inversion()

