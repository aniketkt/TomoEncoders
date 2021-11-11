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


### READING DATA
from tomo_encoders import DataFile
import h5py 

### DETECTOR / RECONSTRUCTION
DET_BINNING = 4
THETA_BINNING = 4
from tomo_encoders.tasks.sparse_segmenter.recon import recon_patches_3d, recon_binning

### SEGMENTATION
from tomo_encoders.tasks import SparseSegmenter
INF_INPUT_SIZE_b = (32,32,32) # input size during inference for binned pass
INF_INPUT_SIZE_1 = (64,64,64) # input size during inference for full pass (binning = 1)
CHUNK_SIZE = 32
NORM_SAMPLING_FACTOR = 4
model_tag = "M_a02"

### VOID ANALYSIS
from tomo_encoders.tasks.sparse_segmenter.detect_voids import wrapper_label, to_regular_grid, upsample_sub_vols



### VISUALIZATION
from tomo_encoders.misc_utils.feature_maps_vis import view_midplanes 


projs_path = '/data02/MyArchive/AM_part_Xuan/projs' 
fname = 'mli_L206_HT_650_L3_projs_bin2_ntheta1500.hdf5'
read_fpath = os.path.join(projs_path, fname)


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

    
        model_params = get_model_params(model_tag)
        model_names = {"segmenter" : "segmenter_Unet_%s"%model_tag}
        
        print("#"*55, "\nWorking on model %s\n"%model_tag, "#"*55)
        fe = SparseSegmenter(model_initialization = 'load-model', \
                             model_names = model_names, \
                             model_path = model_path, \
                             input_size = INF_INPUT_SIZE_b)    
        fe.test_speeds(CHUNK_SIZE)
    
    
    
    
    with h5py.File(read_fpath, 'r') as hf:
        projs = np.asarray(hf['data'][:])
        theta = np.asarray(hf['theta'][:])
        center = float(np.asarray(hf['center'][()]))    

    PROJS_SHAPE = projs.shape
    VOL_SHAPE = calc_vol_shape(PROJS_SHAPE)
    print("projections shape: ", PROJS_SHAPE)
    print("reconstructed volume shape: ", VOL_SHAPE)
    
    
    # BINNED RECONSTRUCTION
    vol_rec_b = recon_binning(projs, theta, center, \
                              THETA_BINNING, \
                              DET_BINNING, \
                              DET_BINNING, \
                              apply_fbp = True, \
                              TIMEIT = TIMEIT_lev1)
    
    # SEGMENTATION OF BINNED RECONSTRUCTION
    p3d_grid_b = Patches(vol_rec_b.shape, \
                         initialize_by = "regular-grid", \
                         patch_size = INF_INPUT_SIZE_b)
    sub_vols_x_b = p3d_grid_b.extract(vol_rec_b, INF_INPUT_SIZE_b)
    min_max = fe.calc_voxel_min_max(vol_rec_b, NORM_SAMPLING_FACTOR, TIMEIT = False)
    sub_vols_y_pred_b = fe.predict_patches(sub_vols_x_b[...,np.newaxis], \
                                      CHUNK_SIZE, None, \
                                      min_max = min_max, \
                                      TIMEIT = TIMEIT_lev1)[...,0]
    vol_seg_b = np.zeros(vol_rec_b.shape, dtype = np.uint8)
    p3d_grid_b.fill_patches_in_volume(sub_vols_y_pred_b, \
                                      vol_seg_b, TIMEIT = False)
    
    assert vol_seg_b.dtype == np.uint8, "data type check failed for vol_seg_b"
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
