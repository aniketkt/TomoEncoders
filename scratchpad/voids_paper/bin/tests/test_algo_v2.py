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

sys.path.append('/home/atekawade/TomoEncoders/scratchpad/voids_paper/configs/')
from tomo_encoders import Grid
from tomo_encoders.misc import viewer
from tomo_encoders import DataFile
import cupy as cp
from tomo_encoders.reconstruction.recon import recon_binning, recon_patches_3d
from params import model_path, get_model_params
from tomo_encoders.neural_nets.surface_segmenter import SurfaceSegmenter
import tensorflow as tf
# from cupyx.scipy.ndimage import zoom
from skimage.filters import threshold_otsu

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
sparse_flag = True


def guess_surface(projs, theta, center, b, b_K, wd):
    ## P-GUESS ##
    # reconstruction
    V_bin = recon_binning(projs, theta, center, b_K, b)    
    # segmentation
    
    thresh = threshold_otsu(V_bin.reshape(-1))
    V_bin[:] = (V_bin < thresh).astype(np.uint8)
    
    # find patches on surface
    wdb = int(wd//b)
    p3d = Grid(V_bin.shape, width = wdb)
    x = p3d.extract(V_bin)

    is_surf = np.std(x, axis = (1,2,3)) > 0.0
    is_ones = np.sum(x, axis = (1,2,3))/(wdb**3) == 1
    is_zeros = np.sum(x, axis = (1,2,3))/(wdb**3) == 0
    is_cyl = p3d._is_within_cylindrical_crop(0.98, 1.0)
    
    p3d = p3d.rescale(b)
    p3d_surf = p3d.filter_by_condition(is_surf & is_cyl)
    p3d_ones = p3d.filter_by_condition(is_ones | (~is_cyl))
    p3d_zeros = p3d.filter_by_condition(is_zeros)

    return p3d_surf, p3d_ones, p3d_zeros
    


def determine_surface(projs, theta, center, fe, p_surf, p_zeros):

    # allocate binary volume
    Vp = np.ones(p_surf.vol_shape, dtype=np.uint8)

    # assign zeros in the metal region; the void region will be left with ones
    if p_zeros is not None:
        s = p_zeros.slices()
        for i in range(len(p_zeros)):
            Vp[tuple(s[i])] = 0.0
    
    # reconstruct patches on the surface
    start_rec = cp.cuda.Event(); end_rec = cp.cuda.Event(); start_rec.record()
    x_surf, p_surf = recon_patches_3d(projs, theta, center, p_surf, \
                                      apply_fbp = True, segmenter = fe, \
                                      segmenter_batch_size = 256)

    end_rec.record(); end_rec.synchronize(); t_rec_surf = cp.cuda.get_elapsed_time(start_rec,end_rec)
    # fill segmented patches into volume
    p_surf.fill_patches_in_volume(x_surf, Vp)    
    print(f"\t TIME: reconstruction + segmentation - {t_rec_surf/1000.0:.2f} secs")
    # print(f'total patches reconstructed and segmented around surface: {len(p_surf)}')    
    eff = len(p_surf)*(wd**3)/np.prod(Vp.shape)
    print(f"\t STAT: r value: {eff*100.0:.2f}")        
    return Vp


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

    # make sure projection shapes are divisible by the patch width (both binning and full steps)
    print(f'SHAPE OF PROJECTION DATA: {projs.shape}')
    
    ##### BEGIN ALGORITHM ########
    # guess surface
    print("STEP: guess surface")
    start_guess = cp.cuda.Event(); end_guess = cp.cuda.Event(); start_guess.record()
    if sparse_flag:
        p_surf, p_ones, p_zeros = guess_surface(projs, theta, center, b, b_K, wd)
    else:
        p_surf = Grid((projs.shape[1], projs.shape[2], projs.shape[2]), width = wd)
        p_ones = None
        p_zeros = None
    end_guess.record(); end_guess.synchronize(); t_guess = cp.cuda.get_elapsed_time(start_guess,end_guess)
    print(f'TIME: guessing neighborhood of surface: {t_guess/1000.0:.2f} seconds')

    # determine surface
    print("STEP: determine surface")
    start_determine = cp.cuda.Event(); end_determine = cp.cuda.Event(); start_determine.record()
    Vp = determine_surface(projs, theta, center, fe, p_surf, p_zeros)
    end_determine.record(); end_determine.synchronize(); t_determine = cp.cuda.get_elapsed_time(start_determine,end_determine)
    print(f'TIME: determining surface: {t_determine/1000.0:.2f} seconds')

    # complete: save stuff    
    ds_save = DataFile('/data02/MyArchive/aisteer_3Dencoders/tmp_data/test_y_pred', tiff = True, d_shape = Vp.shape, d_type = np.uint8, VERBOSITY=0)
    ds_save.create_new(overwrite=True)
    ds_save.write_full(Vp)
    Vp_mask = np.zeros(p_surf.vol_shape, dtype = np.uint8) # Save for illustration purposes the guessed neighborhood of the surface
    p_surf.fill_patches_in_volume(np.ones((len(p_surf),wd,wd,wd)), Vp_mask)
    ds_save = DataFile('/data02/MyArchive/aisteer_3Dencoders/tmp_data/test_y_surf', tiff = True, d_shape = Vp_mask.shape, d_type = np.uint8, VERBOSITY=0)
    ds_save.create_new(overwrite=True)
    ds_save.write_full(Vp_mask)
    
