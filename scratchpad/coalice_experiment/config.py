#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test reconstruction code

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import sys


sys.path.append('../')
fnames = ['/data02/MyArchive/coalice/melting_086.h5', \
          '/data02/MyArchive/coalice/flat_fields_melting_086.h5', \
          '/data02/MyArchive/coalice/dark_fields_melting_086.h5']
recon_params = {"mask_ratio" : 0.95, \
                "contrast_s" : 0.01}
recon_path = '/data02/MyArchive/coalice/recons'

hf = h5py.File(fnames[0], 'r')
delta_t = hf['measurement/instrument/detector/exposure_time'][:]
# pixel_size = hf['measurement/instrument/detector/pixel_size'][:]
hf.close()    

EXPOSURE_TIME_PER_PROJ = float(delta_t[0]) # seconds
NTHETA_180 = 361 # these many projections per 180 degree spin
