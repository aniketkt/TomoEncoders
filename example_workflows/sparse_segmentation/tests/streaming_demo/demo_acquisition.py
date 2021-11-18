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

read_fpath = '/data02/MyArchive/AM_part_Xuan/data/mli_L206_HT_650_L3_rec_1x1_uint16.hdf5' 
from tomo_encoders import DataFile, Patches
import h5py 
import pdb

### DETECTOR / RECONSTRUCTION
DET_BINNING = 4 # detector binning # ONLY VALUES THAT DIVIDE 64 into whole parts.
THETA_BINNING = 4
DET_NTHETA = 2000
DET_FOV = (1920,1000)
DET_PNZ = 4 # may speed up projection compututation (or simulation of acquisition)
from tomo_encoders.tasks.sparse_segmenter.recon import recon_patches_3d, recon_binning
from tomo_encoders.tasks.sparse_segmenter.project import acquire_data
from tomo_encoders.tasks.sparse_segmenter.sparse_segmenter import modified_autocontrast, normalize_volume_gpu

### SEGMENTATION
sys.path.append('../trainer')
from params import *
from tomo_encoders.tasks import SparseSegmenter
INF_INPUT_SIZE_b = (64,64,64) # input size during inference for binned pass
INF_INPUT_SIZE_1 = (64,64,64)
CHUNK_SIZE_b = 32 # at detector binning
CHUNK_SIZE_1 = 32 # full volume
NORM_SAMPLING_FACTOR = 4
model_name = "M_a02_64-64-64"

### VOID ANALYSIS
from tomo_encoders.tasks.sparse_segmenter.detect_voids import wrapper_label, to_regular_grid, upsample_sub_vols
N_MAX_DETECT = 25 # 3 for 2 voids - first one is surface
N_VOIDS_IGNORE = 1

### VISUALIZATION
from tomo_encoders.misc_utils.feature_maps_vis import view_midplanes 
demo_out_path = '/data02/MyArchive/AM_part_Xuan/demo_output'
plot_out_path = '/home/atekawade/Dropbox/Arg/transfers/runtime_plots/'
#import matplotlib as mpl
#mpl.use('Agg')


### TIMING
TIMEIT_lev1 = True
TIMEIT_lev2 = False


def calc_vol_shape(projs_shape):
    ntheta, nz, nx = projs_shape
    return (nz, nx, nx)



def process_data(projs, theta, center, fe):

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

    ##### CLIP WITH AUTOCONTRAST #####
    clip_vals = modified_autocontrast(vol_rec_b, s = 0.05, \
                                      normalize_sampling_factor = NORM_SAMPLING_FACTOR)
    vol_rec_b = np.clip(vol_rec_b, *clip_vals)

    # SEGMENTATION OF BINNED RECONSTRUCTION
    p3d_grid_b = Patches(vol_rec_b.shape, \
                         initialize_by = "grid", \
                         patch_size = INF_INPUT_SIZE_b)
    sub_vols_x_b = p3d_grid_b.extract(vol_rec_b, INF_INPUT_SIZE_b)
    min_max = fe.calc_voxel_min_max(vol_rec_b, NORM_SAMPLING_FACTOR, TIMEIT = False)


    print("MIN MAX at binned reconstruction step: ", min_max)
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
#     plt.show()
    plt.close()

    ##### VOID DETECTION STEP ############
    sub_vols_voids_b, p3d_voids_b = wrapper_label(vol_seg_b, \
                                                  N_MAX_DETECT, \
                                                  TIMEIT = TIMEIT_lev1, \
                                                  N_VOIDS_IGNORE = N_VOIDS_IGNORE)
    
    import vedo
    surf = vedo.Volume(sub_vols_voids_b[0], \
                       mode = 0).isosurface(0.5).smooth().subdivide()
    import pdb; pdb.set_trace()
    vedo.show(surf, bg = 'wheat', bg2 = 'lightblue')
    
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

    ##### CLIP WITH AUTOCONTRAST #####
    clip_vals = modified_autocontrast(np.asarray(sub_vols_grid_voids_1), s = 0.05, \
                                      normalize_sampling_factor = NORM_SAMPLING_FACTOR)
    sub_vols_grid_voids_1 = np.clip(sub_vols_grid_voids_1, *clip_vals)

    #### SEGMENT RECONSTRUCTED PATCHES AT FULL RESOLUTION  

    min_val = sub_vols_grid_voids_1[:,::NORM_SAMPLING_FACTOR].min()
    max_val = sub_vols_grid_voids_1[:,::NORM_SAMPLING_FACTOR].max()
    min_max = (min_val, max_val)

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
    print("TOTAL TIME ELAPSED: %.2f seconds"%(time.time() - t000))
#     plt.show()
    plt.close()
    return













if __name__ == "__main__":

    
    model_names = {"segmenter" : "segmenter_Unet_%s"%model_name}

    print("#"*55, "\nWorking on model %s\n"%model_name, "#"*55)
    fe = SparseSegmenter(model_initialization = 'load-model', \
                         model_names = model_names, \
                         model_path = model_path, \
                         input_size = INF_INPUT_SIZE_b)    
    fe.test_speeds(CHUNK_SIZE_b)
    
    ds = DataFile(read_fpath, data_tag = 'data', tiff = False, VERBOSITY = 0)
    vol = ds.read_full().astype(np.float32)
    vol = normalize_volume_gpu(vol, normalize_sampling_factor = NORM_SAMPLING_FACTOR, chunk_size = 1)
    
    iter_count = 0
    while True:
        print("\n\n", "#"*55, "\n")
        print("ITERATION %i: \n"%iter_count, vol.shape)
        print("\nDOMAIN SHAPE: ", vol.shape)
#         pdb.set_trace()
#         point = (550, 2000, 1800)
        point = (550, 2100, 2100)
        projs, theta, center = acquire_data(vol, point, DET_NTHETA, FOV = DET_FOV, pnz = DET_PNZ)    
        process_data(projs, theta, center, fe)
        iter_count += 1
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    