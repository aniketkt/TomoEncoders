#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import sys
import time
import seaborn as sns
import pandas as pd

from tomo_encoders import Patches
from tomo_encoders.misc import viewer
from tomo_encoders import DataFile
import cupy as cp
from tomo_encoders.reconstruction.project import get_projections
from tomo_encoders.reconstruction.recon import recon_binning
from tomo_encoders.misc.voxel_processing import cylindrical_mask, normalize_volume_gpu

read_path = '/data02/MyArchive/aisteer_3Dencoders/tmp_data/mosaic'
save_path = '/data02/MyArchive/aisteer_3Dencoders/tmp_data/mosaic_raw'

if __name__ == "__main__":
    
    
    for ilayer in range(1,6+1):
        fpath = os.path.join(read_path, 'mli_L206_HT_650_L%i_rec_1x1_uint16_tiff'%ilayer)
        ds_full = DataFile(fpath, tiff = True)
        break
        # what should be the number of projections?
        V = ds_full.read_full()
        V = normalize_volume_gpu(V.astype(np.float32), chunk_size=1, normalize_sampling_factor=4, TIMEIT = True)
        
        cylindrical_mask(V, 0.98, mask_val = 0.0)
        theta = np.linspace(0,np.pi,1500,dtype='float32')
        pnz = 4
        center = V.shape[-1]//2.0
        projs, theta, center = get_projections(V, theta, center, pnz)

        hf = h5py.File(os.path.join(save_path,'layer%i'%ilayer), 'w')
        hf.create_dataset("data",data = projs)
        hf.create_dataset("theta", data = theta)
        hf.create_dataset("center", data = center)
        hf.close() 
    
    
    
    
    