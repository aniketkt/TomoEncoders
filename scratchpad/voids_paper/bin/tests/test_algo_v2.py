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
from tomo_encoders import Grid
from tomo_encoders.misc import viewer
from tomo_encoders import DataFile
import cupy as cp
from tomo_encoders.reconstruction.project import get_projections
from tomo_encoders.reconstruction.recon import recon_binning, recon_patches_3d, recon_patches_3d_2
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

model_tag = "M_a07"
model_names = {"segmenter" : "segmenter_Unet_%s"%model_tag}
model_params = get_model_params(model_tag)
# patch size
wd = 32
# guess surface parameters
b = 4
b_K = 4


def upsample(p_sel, V_bin, zoom_fac):
    x = p_sel.extract(V_bin)
    sums = np.sum(x, axis = (1,2,3))/np.prod(x[0].shape)
    x = np.ones(tuple([len(p_sel)]+[p_sel.wd*zoom_fac]*3), dtype = x.dtype)
    x[sums == 0,...] = 0.0
    # x = zoom(cp.array(x), (1,zoom_fac,zoom_fac,zoom_fac), mode='constant', order=1).get()
    # cp._default_memory_pool.free_all_blocks()    
    p_sel = p_sel.rescale(zoom_fac)
    return x, p_sel


def guess_surface(projs, theta, center, fe, b, b_K, wd, V_bin):
    ## P-GUESS ##
    # reconstruction
    V_bin[:] = recon_binning(projs, theta, center, b_K, b, b)    
    # segmentation
    min_max = V_bin.min(), V_bin.max()
    p3d_bin = Grid(V_bin.shape, width = wd)
    x = p3d_bin.extract(V_bin) # effectively 128^3 patches in a full volume
    x = fe.predict_patches("segmenter", x[...,np.newaxis], 256, None, min_max = min_max)[...,0]
    p3d_bin.fill_patches_in_volume(x, V_bin)

    # find patches on surface
    VOL_SHAPE = (projs.shape[1], projs.shape[2], projs.shape[2])
    p3d = Grid(V_bin.shape, width = int(wd//b))
    x = p3d.extract(V_bin)
    is_surf = np.std(x, axis = (1,2,3)) > 0.0
    is_cyl = p3d._is_within_cylindrical_crop(0.98, 1.0)
    is_surf = is_surf & is_cyl
    p3d_surf = p3d.filter_by_condition(is_surf)
    
    p3d = p3d.filter_by_condition(~is_surf)
    return p3d_surf, p3d
    

if __name__ == "__main__":


    # initialize segmenter fCNN
    fe = SurfaceSegmenter(model_initialization = 'load-model', \
                         model_names = model_names, \
                         model_path = model_path)    
    fe.test_speeds(128,n_reps = 5, input_size = (wd,wd,wd))    

    # read data and initialize output arrays
    ## to-do: ensure reconstructed object has dimensions that are a multiple of the (wd,wd,wd) !!    
    hf = h5py.File('/data02/MyArchive/aisteer_3Dencoders/tmp_data/projs_2k.hdf5', 'r')
    projs = np.asarray(hf["data"][:])
    theta = np.asarray(hf['theta'][:])
    center = float(np.asarray(hf["center"]))
    hf.close()
    VOL_SHAPE = (projs.shape[1], projs.shape[2], projs.shape[2])
    VOL_B_SHAPE = (projs.shape[1]//b, projs.shape[2]//b, projs.shape[2]//b)
    Vp = np.ones(VOL_SHAPE, dtype=np.uint8)
    Vp_bin = np.zeros(VOL_B_SHAPE, dtype = np.float32)

    
    ##### BEGIN ALGORITHM ########
    # guess surface
    start_guess = cp.cuda.Event(); end_guess = cp.cuda.Event(); start_guess.record()
    p_surf_bin, p_not_surf_bin = guess_surface(projs, theta, center, fe, b, b_K, wd, Vp_bin)
    end_guess.record(); end_guess.synchronize(); t_guess = cp.cuda.get_elapsed_time(start_guess,end_guess)
    print(f'time for guessing neighborhood of surface: {t_guess/1000.0:.2f} seconds')
    
    # simply upsample those patches that don't belong to the surface
    start_upsample = cp.cuda.Event(); end_upsample = cp.cuda.Event(); start_upsample.record()
    x_not_surf, p_not_surf = upsample(p_not_surf_bin, Vp_bin, b)
    p_not_surf.fill_patches_in_volume(x_not_surf, Vp)
    end_upsample.record(); end_upsample.synchronize(); t_upsample = cp.cuda.get_elapsed_time(start_upsample,end_upsample)
    print(f'time for upsampling the binned segmentation outside the surface: {t_upsample/1000.0:.2f} seconds')
    # reconstruct patches on the surface
    start_rec = cp.cuda.Event(); end_rec = cp.cuda.Event(); start_rec.record()
    p_surf = p_surf_bin.rescale(b)
    x_surf, p_surf = recon_patches_3d_2(projs, theta, center, p_surf, apply_fbp =True)
    end_rec.record(); end_rec.synchronize(); t_rec_surf = cp.cuda.get_elapsed_time(start_rec,end_rec)
    print(f'time for surface reconstruction: {t_rec_surf/1000.0:.2f} seconds')
    # segment patches on the surface
    t_start_seg = time.time()
    min_max = x_surf[:,::4,::4,::4].min(), x_surf[:,::4,::4,::4].max()
    x_surf = fe.predict_patches("segmenter", x_surf[...,np.newaxis], 256, None, min_max = min_max)[...,0]
    p_surf.fill_patches_in_volume(x_surf, Vp)    
    print(f"time for surface segmentation: {(time.time() - t_start_seg):.2f} seconds")    

    # complete: show time stats
    print(f'total patches reconstructed and segmented around surface: {x_surf.shape}')    
    eff = len(p_surf)/(len(p_surf) + len(p_not_surf))
    print(f"voxels in the neighborhood of surface: {eff*100.0:.2f} pc of total")        
    
    ds_save = DataFile('/data02/MyArchive/aisteer_3Dencoders/tmp_data/test_y_pred', tiff = True, d_shape = Vp.shape, d_type = np.uint8)
    ds_save.create_new(overwrite=True)
    ds_save.write_full(Vp)

    # Save for illustration purposes the guessed neighborhood of the surface
    Vp_mask = np.zeros(VOL_SHAPE, dtype = np.uint8)
    p_surf.fill_patches_in_volume(np.ones((len(x_surf),wd,wd,wd)), Vp_mask)
    ds_save = DataFile('/data02/MyArchive/aisteer_3Dencoders/tmp_data/test_y_surf', tiff = True, d_shape = Vp_mask.shape, d_type = np.uint8)
    ds_save.create_new(overwrite=True)
    ds_save.write_full(Vp_mask)
    
    
    
    
    
    
    
    
