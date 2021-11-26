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
from recon4D import DataGetter
from tomo_encoders.misc import viewer
from tomo_encoders import DataFile


fnames = ['/data02/MyArchive/coalice/melting_086.h5', \
          '/data02/MyArchive/coalice/flat_fields_melting_086.h5', \
          '/data02/MyArchive/coalice/dark_fields_melting_086.h5']
recon_params = {"mask_ratio" : 0.95, \
                "contrast_s" : 0.01}
recon_path = '/data02/MyArchive/coalice/recons'

