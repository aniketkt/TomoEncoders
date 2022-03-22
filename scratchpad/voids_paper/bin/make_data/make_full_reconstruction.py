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
from tomo_encoders.reconstruction.recon import recon_binning, recon_patches_3d

base_path = '/data02/MyArchive/aisteer_3Dencoders/tmp_data'
if __name__ == "__main__":

    
    paths_list = [('projs_2k.hdf5', 'test_x_rec'),\
                 ('projs_4k.hdf5', 'test_x_rec_4k')]
    
    for paths in paths_list:
        fpath_projs = os.path.join(base_path, paths[0])
        fpath_x_rec = os.path.join(base_path, paths[1])
        
        hf = h5py.File(fpath_projs, 'r')
        projs = np.asarray(hf["data"][:])
        theta = np.asarray(hf['theta'][:])
        center = float(np.asarray(hf["center"]))
        hf.close()    

        VOL_SHAPE = (projs.shape[1], projs.shape[2], projs.shape[2])
        PATCH_SIZE = (32,32,32)    

        p_grid = Patches(VOL_SHAPE, initialize_by='regular-grid', patch_size = PATCH_SIZE)
        p_grid = p_grid.filter_by_cylindrical_mask(mask_ratio=1)
        x_rec, p_grid = recon_patches_3d(projs, theta, center, p_grid, TIMEIT = True)
        print(f'total patches reconstructed: {x_rec.shape}')    

        Vx_rec = np.ones(VOL_SHAPE)*x_rec[:,::4,::4,::4].min()    
        p_grid.fill_patches_in_volume(x_rec, Vx_rec)

        ds_rec = DataFile(fpath_x_rec, tiff = True, \
                          d_shape = Vx_rec.shape, d_type = np.float32)
        ds_rec.create_new(overwrite=True)
        ds_rec.write_full(Vx_rec)    
    
    
    
    
    
