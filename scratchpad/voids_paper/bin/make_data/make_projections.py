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


fpath_full = '/data02/MyArchive/tomo_datasets/AM_part_Xuan/data/mli_L206_HT_650_L3_rec_1x1_uint16_tiff'
if __name__ == "__main__":
    
    ds_full = DataFile(fpath_full, tiff = True)
    V = ds_full.read_full()
    V = normalize_volume_gpu(V.astype(np.float32), chunk_size=1, normalize_sampling_factor=4, TIMEIT = True)
    
    # 2k volume
    Vx = V[100:-100-6, 1000:-1000-24, 1000:-1000-24].copy()
    cylindrical_mask(Vx, 0.98, mask_val = 0.0)
    theta = np.linspace(0,np.pi,1500,dtype='float32')
    pnz = 16
    center = Vx.shape[-1]//2.0
    projs, theta, center = get_projections(Vx, theta, center, pnz)

    hf = h5py.File('/data02/MyArchive/aisteer_3Dencoders/tmp_data/projs_2k.hdf5', 'w')
    hf.create_dataset("data",data = projs)
    hf.create_dataset("theta", data = theta)
    hf.create_dataset("center", data = center)
    hf.close() 
    
    # 4k volume
    Vx = V[100+256:-100-256-6, :-104, :-104].copy()
    cylindrical_mask(Vx, 0.98, mask_val = 0.0)
    theta = np.linspace(0,np.pi,3000,dtype='float32')
    pnz = 4
    center = Vx.shape[-1]//2.0
    projs, theta, center = get_projections(Vx, theta, center, pnz)

    hf = h5py.File('/data02/MyArchive/aisteer_3Dencoders/tmp_data/projs_4k.hdf5', 'w')
    hf.create_dataset("data",data = projs)
    hf.create_dataset("theta", data = theta)
    hf.create_dataset("center", data = center)
    hf.close()    
    
    
    
    