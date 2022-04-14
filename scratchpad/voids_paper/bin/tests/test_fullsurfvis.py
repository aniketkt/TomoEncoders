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

sys.path.append('/home/atekawade/TomoEncoders/scratchpad/voids_paper/configs')
from params import model_path, get_model_params
sys.path.append('/home/atekawade/TomoEncoders/scratchpad/voids_paper')
from tomo_encoders.tasks.surface_determination import guess_surface, determine_surface
from tomo_encoders import DataFile
import cupy as cp
from tomo_encoders.neural_nets.surface_segmenter import SurfaceSegmenter

# from cupyx.scipy.ndimage import zoom


######## START GPU SETTINGS ############
########## SET MEMORY GROWTH to True ############
import tensorflow as tf
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

# handy code for timing stuff
# st_chkpt = cp.cuda.Event(); end_chkpt = cp.cuda.Event(); st_chkpt.record()    
# end_chkpt.record(); end_chkpt.synchronize(); t_chkpt = cp.cuda.get_elapsed_time(st_chkpt,end_chkpt)
# print(f"time checkpoint {t_chkpt/1000.0:.2f} secs")


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
        from tomo_encoders import Grid
        p_surf = Grid((projs.shape[1], projs.shape[2], projs.shape[2]), width = wd)
        p_ones = None
        p_zeros = None
    end_guess.record(); end_guess.synchronize(); t_guess = cp.cuda.get_elapsed_time(start_guess,end_guess)
    print(f'TIME: guessing neighborhood of surface: {t_guess/1000.0:.2f} seconds')

    
    # determine surface
    print("STEP: determine surface")
    start_determine = cp.cuda.Event(); end_determine = cp.cuda.Event(); start_determine.record()
    
    # allocate binary volume
    Vp = np.ones(p_surf.vol_shape, dtype=np.uint8)

    # assign zeros in the metal region; the void region will be left with ones
    if p_zeros is not None:
        s = p_zeros.slices()
        for i in range(len(p_zeros)):
            Vp[tuple(s[i])] = 0.0
    Vp = determine_surface(projs, theta, center, fe, p_surf, Vp = Vp)
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
    
