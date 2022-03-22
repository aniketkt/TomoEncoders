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
import vedo

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
read_fpath = '/data02/MyArchive/tomo_datasets/AM_part_Xuan/data/mli_L206_HT_650_L3_rec_1x1_uint16.hdf5' 
from tomo_encoders.misc.voxel_processing import normalize_volume_gpu
from tomo_encoders import DataFile, Patches
import h5py 
import pdb
from tomo_encoders.labeling.detect_voids import to_regular_grid

### SEGMENTATION
sys.path.append('../trainer')
from params import *
from tomo_encoders.neural_nets.sparse_segmenter import SparseSegmenter
INF_INPUT_SIZE = (64,64,64)
INF_CHUNK_SIZE = 32 # full volume
NORM_SAMPLING_FACTOR = 4
model_name = "M_a02_64-64-64"

### VOID ANALYSIS
N_MAX_DETECT = 25 # 3 for 2 voids - first one is surface
CIRC_MASK_FRAC = 0.75

### VISUALIZATION
HEADLESS = False

from tomo_encoders.misc.feature_maps_vis import view_midplanes 

if HEADLESS:
    import matplotlib as mpl
    mpl.use('Agg')


### TIMING
TIMEIT_lev1 = True
TIMEIT_lev2 = False

def visualize_vedo(sub_vols, p3d):
    '''
    make vol_seg_b.
    to-do - select a specific void as IDX_VOID_SELECT and show it in different color.
    '''

    vol_vis = np.zeros(p3d.vol_shape, dtype = np.uint8)
    p3d.fill_patches_in_volume(sub_vols, vol_vis)    

    surf = vedo.Volume(vol_vis).isosurface(0.5).smooth().subdivide()
    return surf

def select_voids(sub_vols, p3d, s_sel):
    
    s_sel = list(s_sel)
    if s_sel[0] is None:
        s_sel[0] = 0
    if s_sel[1] is None:
        s_sel[1] = len(p3d)
    assert len(p3d) == len(sub_vols), "error while selecting voids. the length of patches and sub_vols must match"
    s_sel = tuple(s_sel)
    
    idxs = np.arange(s_sel[0], s_sel[1]).tolist()
    return [sub_vols[ii] for ii in idxs], p3d.select_by_range(s_sel)

from tomo_encoders.tasks import VoidMetrology
def calc_vol_shape(projs_shape):
    ntheta, nz, nx = projs_shape
    return (nz, nx, nx)

def process_data(vol, fe):

    t000 = time.time()
    print("\n\nSTART PROCESSING\n\n")
    VOL_SHAPE_1 = vol.shape
    print("reconstructed volume shape: ", VOL_SHAPE_1)
    
    ##### CLIP WITH AUTOCONTRAST #####
    clip_vals = modified_autocontrast(vol, s = 0.05, normalize_sampling_factor = NORM_SAMPLING_FACTOR)
    vol = np.clip(vol, *clip_vals)
    
    # SEGMENTATION STEP
    p3d_grid = Patches(vol.shape, initialize_by = "grid", patch_size = INF_INPUT_SIZE)
    sub_vols_x = p3d_grid.extract(vol, INF_INPUT_SIZE)
    min_max = fe.calc_voxel_min_max(vol, NORM_SAMPLING_FACTOR, TIMEIT = False)
    sub_vols_y_pred = fe.predict_patches("segmenter", sub_vols_x[...,np.newaxis], INF_CHUNK_SIZE, None, min_max = min_max, TIMEIT = TIMEIT_lev2)
    if TIMEIT_lev2: # unpack if time is returned
        sub_vols_y_pred, _ = sub_vols_y_pred

    sub_vols_y_pred = sub_vols_y_pred[...,0]
    vol_seg = np.zeros(vol.shape, dtype = np.uint8)

    p3d_grid.fill_patches_in_volume(sub_vols_y_pred, vol_seg, TIMEIT = False)
    assert vol_seg.dtype == np.uint8, "data type check failed for vol_seg_b"
    cylindrical_mask(vol_seg, CIRC_MASK_FRAC, mask_val = 0)
        
    # VISUALIZE STEP
    fig, ax = plt.subplots(1, 3, figsize = (8,4))
    view_midplanes(vol = vol, ax = ax)
    view_midplanes(vol = vol_seg, cmap = 'copper', alpha = 0.3, ax = ax)
    if not HEADLESS:
        plt.show()
    plt.close()
    
    
    return vol_seg



if __name__ == "__main__":

    
    model_names = {"segmenter" : "segmenter_Unet_%s"%model_name}

    print("#"*55, "\nWorking on model %s\n"%model_name, "#"*55)
    fe = SparseSegmenter(model_initialization = 'load-model', \
                         model_names = model_names, \
                         model_path = model_path)    
    fe.test_speeds(INF_CHUNK_SIZE)
    
    ds = DataFile(read_fpath, data_tag = 'data', tiff = False, VERBOSITY = 0)
    vol = ds.read_full().astype(np.float32)
    vol = normalize_volume_gpu(vol, normalize_sampling_factor = NORM_SAMPLING_FACTOR, chunk_size = 1)
    
    
    vol_seg = process_data(vol, fe)
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
