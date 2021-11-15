#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import sys 
import matplotlib.pyplot as plt 
import numpy as np 
import cupy as cp 
import time 
import os 
import tensorflow as tf


######## START GPU SETTINGS ############
########## SET MEMORY GROWTH to True ############
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass        
######### END GPU SETTINGS ############


### READING DATA

projs_path = '/data02/MyArchive/AM_part_Xuan/projs' 
from tomo_encoders import DataFile, Patches
import h5py 


### DETECTOR / RECONSTRUCTION
DET_BINNING = 4 # detector binning
THETA_BINNING = 4
from tomo_encoders.tasks.sparse_segmenter.recon import recon_patches_3d, recon_binning

### SEGMENTATION
sys.path.append('../trainer')
from params import *
from tomo_encoders.tasks import SparseSegmenter
INF_INPUT_SIZE_b = (64,64,64) # input size during inference for binned pass
INF_INPUT_SIZE_1 = (64,64,64)
CHUNK_SIZE_b = 32 # at detector binning
CHUNK_SIZE_1 = 32 # full volume
NORM_SAMPLING_FACTOR = 4
model_name = "M_a05_64-64-64"

### VOID ANALYSIS
from tomo_encoders.tasks.sparse_segmenter.detect_voids import wrapper_label, to_regular_grid, upsample_sub_vols
N_MAX_DETECT = 3 # 3 for 2 voids
N_VOIDS_IGNORE = 1

### VISUALIZATION
from tomo_encoders.misc_utils.feature_maps_vis import view_midplanes 
demo_out_path = '/data02/MyArchive/AM_part_Xuan/demo_output'
plot_out_path = '/home/atekawade/Dropbox/Arg/transfers/runtime_plots/'
fname = 'mli_L206_HT_650_L3_projs_bin2_ntheta1500.hdf5'
read_fpath = os.path.join(projs_path, fname)
import matplotlib as mpl
mpl.use('Agg')


### TIMING
TIMEIT_lev1 = True
TIMEIT_lev2 = False

########## SAMPLE CODE ###########

# # sub_vols shape is (batch_size, nz, ny, nx) so use newaxis to get to (batch_size, nz, ny, nx, 1)
# y_pred = self.predict_patches(sub_vols[...,np.newaxis], \
#                               chunk_size, out_arr, \
#                               min_max = min_max, \
#                               TIMEIT = False)[...,0]
# self.fill_patches_in_volume(y_pred, p, vol_out)

def calc_vol_shape(projs_shape):
    ntheta, nz, nx = projs_shape
    return (nz, nx, nx)

    

if __name__ == "__main__":

    
    model_names = {"segmenter" : "segmenter_Unet_%s"%model_name}

    print("#"*55, "\nWorking on model %s\n"%model_name, "#"*55)
    fe = SparseSegmenter(model_initialization = 'load-model', \
                         model_names = model_names, \
                         model_path = model_path, \
                         input_size = INF_INPUT_SIZE_b)    
    fe.test_speeds(CHUNK_SIZE_b)
    
    with h5py.File(read_fpath, 'r') as hf:
        projs = np.asarray(hf['data'][:])
        theta = np.asarray(hf['theta'][:])
        center = projs.shape[-1]/2.0
#         center = float(np.asarray(hf['center'][()]))    
    
    
    t000 = time.time()
    print("\n\nSTART PROCESSING\n\n")
    
    PROJS_SHAPE_1 = projs.shape
    VOL_SHAPE_1 = calc_vol_shape(PROJS_SHAPE_1)
    print("projections shape: ", PROJS_SHAPE_1)
    print("reconstructed volume shape: ", VOL_SHAPE_1)
    
    
    # BINNED RECONSTRUCTION
    vol_rec_b = recon_binning(projs, theta, center, \
                              THETA_BINNING, \
                              DET_BINNING, \
                              DET_BINNING, \
                              apply_fbp = True, \
                              TIMEIT = TIMEIT_lev1)
    print("vol_rec_b shape: ", vol_rec_b.shape)
    
    # SEGMENTATION OF BINNED RECONSTRUCTION
    p3d_grid_b = Patches(vol_rec_b.shape, \
                         initialize_by = "grid", \
                         patch_size = INF_INPUT_SIZE_b)
    sub_vols_x_b = p3d_grid_b.extract(vol_rec_b, INF_INPUT_SIZE_b)
    min_max = fe.calc_voxel_min_max(vol_rec_b, NORM_SAMPLING_FACTOR, TIMEIT = False)
    
#     print("\n\nDEBUG: min, max for binned pass: %.4f, %.4f\n\n"%min_max)
    
    sub_vols_y_pred_b, _ = fe.predict_patches(sub_vols_x_b[...,np.newaxis], \
                                           CHUNK_SIZE_b, None, \
                                           min_max = min_max, \
                                           TIMEIT = TIMEIT_lev1)
    sub_vols_y_pred_b = sub_vols_y_pred_b[...,0]
    
    
    vol_seg_b = np.zeros(vol_rec_b.shape, dtype = np.uint8)
    p3d_grid_b.fill_patches_in_volume(sub_vols_y_pred_b, \
                                      vol_seg_b, TIMEIT = False)
    
    assert vol_seg_b.dtype == np.uint8, "data type check failed for vol_seg_b"
    
    print("vol_seg_b shape: ", vol_seg_b.shape)
    fig, ax = plt.subplots(1, 3, figsize = (8,4))
    view_midplanes(vol = fe.rescale_data(vol_rec_b, *min_max), ax = ax)
    view_midplanes(vol = vol_seg_b, cmap = 'copper', alpha = 0.3, ax = ax)
    plt.savefig(os.path.join(plot_out_path, "vols_b_%s.png"%model_name))
    plt.close()

    ##### VOID DETECTION STEP ############
    sub_vols_voids_b, p3d_voids_b = wrapper_label(vol_seg_b, \
                                                  N_MAX_DETECT, \
                                                  TIMEIT = TIMEIT_lev1, \
                                                  N_VOIDS_IGNORE = N_VOIDS_IGNORE)
    p3d_grid_1_voids = to_regular_grid(sub_vols_voids_b, \
                                       p3d_voids_b, \
                                       INF_INPUT_SIZE_1, \
                                       VOL_SHAPE_1, \
                                       DET_BINNING)
    
    #### SELECTED PATCHES RECONSTRUCTED AT FULL RESOLUTION
    sub_vols_grid_voids_1, p3d_grid_voids_1 = recon_patches_3d(projs, theta, center, \
                                                          p3d_grid_1_voids, \
                                                          apply_fbp = True, \
                                                          TIMEIT = TIMEIT_lev1)

    print("length of sub_vols reconstructed %i"%len(sub_vols_grid_voids_1))
    
    #### SEGMENT RECONSTRUCTED PATCHES AT FULL RESOLUTION  
    
    min_val = sub_vols_grid_voids_1[:,::NORM_SAMPLING_FACTOR].min()
    max_val = sub_vols_grid_voids_1[:,::NORM_SAMPLING_FACTOR].max()
    min_max = (min_val, max_val)
    
#     print("\n\nDEBUG: min, max for full pass: %.4f, %.4f\n\n"%min_max)
    
    fe.input_size = INF_INPUT_SIZE_1 # this may not be necessary to assign
    sub_vols_y_pred_1, _ = fe.predict_patches(sub_vols_grid_voids_1[...,np.newaxis], \
                                           CHUNK_SIZE_1, None, \
                                           min_max = min_max, \
                                           TIMEIT = TIMEIT_lev1)
    sub_vols_y_pred_1 = sub_vols_y_pred_1[...,0]
    
    
    
    vol_seg_1 = np.ones(VOL_SHAPE_1, dtype = np.uint8)
    p3d_grid_voids_1.fill_patches_in_volume(sub_vols_y_pred_1, vol_seg_1, TIMEIT = TIMEIT_lev1)
    
    print("vol_seg_1 shape: ", vol_seg_1.shape)
    fig, ax = plt.subplots(1, 3, figsize = (8,4))
    view_midplanes(vol = vol_seg_1, cmap = 'copper', ax = ax)
    plt.savefig(os.path.join(plot_out_path, "vols_1_%s.png"%model_name))
    plt.close()
    
    
    print("TOTAL TIME ELAPSED: %.2f seconds"%(time.time() - t000))
    exit()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
