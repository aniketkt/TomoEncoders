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

sys.path.append('/data02/MyArchive/aisteer_3Dencoders/TomoEncoders/scratchpad/voids_paper/configs/')
from tomo_encoders import Patches
from tomo_encoders.misc import viewer
from tomo_encoders import DataFile
import cupy as cp
from tomo_encoders.reconstruction.project import get_projections
from tomo_encoders.reconstruction.recon import recon_binning, recon_patches_3d
from tomo_encoders.misc.voxel_processing import cylindrical_mask, normalize_volume_gpu
from params import model_path, get_model_params
from tomo_encoders.neural_nets.surface_segmenter import SurfaceSegmenter
import tensorflow as tf
from cupyx.scipy.ndimage import zoom

######## START GPU SETTINGS ############
########## SET MEMORY GROWTH to True ############
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass        
######### END GPU SETTINGS ############




if __name__ == "__main__":

    
    hf = h5py.File('/data02/MyArchive/aisteer_3Dencoders/tmp_data/projs_2k.hdf5', 'r')
    projs = np.asarray(hf["data"][:])
    theta = np.asarray(hf['theta'][:])
    center = float(np.asarray(hf["center"]))
    hf.close()
    VOL_SHAPE = (projs.shape[1], projs.shape[2], projs.shape[2])
    Vp = np.ones(VOL_SHAPE, dtype=np.uint8)
    Vp_mask = np.zeros(VOL_SHAPE, dtype = np.uint8)

    
    t000 = time.time()
    Vx_bin = recon_binning(projs, theta, center, 4, 4, 4)    
    model_tag = "M_a07"
    model_names = {"segmenter" : "segmenter_Unet_%s"%model_tag}
    model_params = get_model_params(model_tag)

    fe = SurfaceSegmenter(model_initialization = 'load-model', \
                         model_names = model_names, \
                         model_path = model_path)    
    fe.test_speeds(128,n_reps = 5, input_size = (32,32,32))    
    
    t00 = time.time()
    ## ASSUMES v has dimensions that are a multiple of the patch_size !!
    min_max = Vx_bin.min(), Vx_bin.max()

    p_sel = Patches(Vx_bin.shape, initialize_by='regular-grid', patch_size = (32,32,32))
    x = p_sel.extract(Vx_bin, (32,32,32)) # effectively 128^3 patches in a full volume
    x = fe.predict_patches("segmenter", x[...,np.newaxis], 256, None, min_max = min_max)[...,0]
    x = np.asarray([zoom(cp.array(_x),4, mode = 'constant', order = 1).get() for _x in x])    
    p_sel = p_sel.rescale(4, VOL_SHAPE)
    print(f'upsampled x array shape {x.shape}')
    p_sel.fill_patches_in_volume(x, Vp)
    t11 = time.time()
    print(f"time for segmentation: {(t11 - t00):.2f} seconds")    

    # improve segmentation on the surface voxels
    p_surf = Patches(VOL_SHAPE, initialize_by='regular-grid', patch_size=(32,32,32))
    tot_patches = len(p_surf)
    edge_mask = np.std(p_surf.extract(Vp,(8,8,8)),axis = (1,2,3)) > 0 # downsample by 4 to save time in computing np.std
    p_surf = p_surf.filter_by_condition(edge_mask)
    eff = len(p_surf)/tot_patches
    print(f"voxels in the neighborhood of surface: {eff*100.0:.2f} pc of total")        
    
    t111 = time.time()
    x_surf, p_surf = recon_patches_3d(projs, theta, center, p_surf, TIMEIT = True)
    min_max = x_surf[:,::4,::4,::4].min(), x_surf[:,::4,::4,::4].max()
    t00 = time.time()
    x_surf = fe.predict_patches("segmenter", x_surf[...,np.newaxis], 256, None, min_max = min_max)[...,0]
    print(f'total patches reconstructed and segmented around surface: {x_surf.shape}')    
    print(f"time for segmentation: {(time.time() - t00):.2f} seconds")    
    p_surf.fill_patches_in_volume(x_surf, Vp)    
    p_surf.fill_patches_in_volume(np.ones((len(x_surf),32,32,32)), Vp_mask)
    
       
    
    t222 = time.time()
    
    print(f'TOTAL PROCESSING TIME: {t222-t000:.2f} secs')
    print(f'FULL-RES PROCESSING TIME: {t222-t111:.2f} secs')
    ds_save = DataFile('/data02/MyArchive/aisteer_3Dencoders/tmp_data/test_y_pred', tiff = True, d_shape = Vp.shape, d_type = np.uint8)
    ds_save.create_new(overwrite=True)
    ds_save.write_full(Vp)
    
    ds_save = DataFile('/data02/MyArchive/aisteer_3Dencoders/tmp_data/test_y_surf', tiff = True, d_shape = Vp_mask.shape, d_type = np.uint8)
    ds_save.create_new(overwrite=True)
    ds_save.write_full(Vp_mask)
    
    
    
    
    
    
    
    
