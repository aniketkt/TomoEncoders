#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import time
from tomo_encoders import DataFile
import os
import numpy as np
import sys
sys.path.append('/data02/MyArchive/aisteer_3Dencoders/TomoEncoders/scratchpad/voids_paper/configs/')
from params import model_path, get_model_params
import tensorflow as tf

from tomo_encoders.neural_nets.surface_segmenter import SurfaceSegmenter
from tomo_encoders import Grid, Patches
from tomo_encoders.labeling.detect_voids import export_voids


######## START GPU SETTINGS ############
########## SET MEMORY GROWTH to True ############
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass        
######### END GPU SETTINGS ############

# FILE I/O
dir_path = '/data02/MyArchive/tomo_datasets/AM_part_Xuan/data/xzhang_feb22_rec/wheel1_sam1'
save_path = '/data02/MyArchive/tomo_datasets/AM_part_Xuan/seg_data/xzhang_feb22_rec/wheel1_sam1'
if not os.path.exists(save_path): os.makedirs(save_path)


# STITCHING PARAMETERS
id_start = [0,75,75]
id_end = [849,849,924]


# SEGMENTATION PARAMETERS
model_tag = "M_a02"
model_names = {"segmenter" : "segmenter_Unet_%s"%model_tag}
model_params = get_model_params(model_tag)
# patch size
wd = 32

# VOID DETECTION PARAMETERS
N_MAX_DETECT = 10000



def make_stitched(dir_path, id_start, id_end):
    n_layers = len(id_start)
    Vx_full = []
    for il in range(n_layers):
        ds = DataFile(os.path.join(dir_path, f'layer{il+1}'), tiff=True)        
        Vx_full.append(ds.read_chunk(axis=0, slice_start=id_start[il], slice_end=id_end[il], return_slice=False).astype(np.float32))
    Vx_full = np.concatenate(Vx_full, axis=0)

    print(Vx_full.shape)
    return Vx_full


def segment_volume(Vx_full, fe, wd):

    p_grid = Grid(Vx_full.shape, width = wd)
    min_max = Vx_full[::4,::4,::4].min(), Vx_full[::4,::4,::4].max()
    x = p_grid.extract(Vx_full)
    x = fe.predict_patches("segmenter", x[...,np.newaxis], 256, None, min_max = min_max)[...,0]
    print(f"shape of x array is {x.shape}")
    p_grid.fill_patches_in_volume(x, Vx_full) # values in Vx_full are converted to binary (0 and 1) in-place
    Vx_full = Vx_full.astype(np.uint8)
    return Vx_full

if __name__ == "__main__":

    

    # do stuff
    # STEP 1
    # make a big volume that stitches together all layers in one volume; Vx_full.shape will be (tot_ht, ny, nx)
    t_start = time.time()
    Vx_full = make_stitched(dir_path, id_start, id_end)
    

    # make sure Vx_full shape is divisible by 32
    nz, ny, nx = Vx_full.shape
    print(f"shape of Vx_full was {Vx_full.shape}")
    Vx_full = Vx_full[:-(nz%wd), :-(ny%wd), :-(nx%wd)].copy()
    print(f"after cropping, shape of Vx_full is {Vx_full.shape}")
    
    print(f"TIME stitching: {time.time()-t_start:.2f} seconds")


    ds_save = DataFile(os.path.join(save_path, "stitched"), tiff = True, d_shape = Vx_full.shape, d_type = Vx_full.dtype)
    ds_save.create_new(overwrite=True)
    ds_save.write_full(Vx_full)

    # STEP 2
    # Process Vx_full into Vy_full where Vy_full contains only ones (inside void) and zeros (inside metal)
    # initialize segmenter fCNN
    fe = SurfaceSegmenter(model_initialization = 'load-model', \
                         model_names = model_names, \
                         model_path = model_path)    
    fe.test_speeds(128,n_reps = 5, input_size = (wd,wd,wd))    
    
    t0 = time.time()
    Vx_full = segment_volume(Vx_full, fe, wd)
    print(f"TIME segmentation: {time.time()-t_start:.2f} seconds")

    ds_save = DataFile(os.path.join(save_path, "segmented"), tiff = True, d_shape = Vx_full.shape, d_type = Vx_full.dtype)
    ds_save.create_new(overwrite=True)
    ds_save.write_full(Vx_full)


    # STEP 3
    # Process Vy_full into void_vols where void_vols is a list of many ndarrays with different shapes (pz, py, px) representing each void
    # Also output cz, cy, cx for each void_vol in void_vols giving the center of the void volume w.r.t. the coordinates in Vy_full
    x_voids, p_voids = export_voids(Vx_full, N_MAX_DETECT, TIMEIT = True, invert = False)






    # STEP 4
    # Process all void_vols into void_surfs in the form of a single .ply file and save
