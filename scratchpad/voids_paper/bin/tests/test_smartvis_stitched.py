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
from tomo_encoders.tasks.void_mapping import process_patches, segment_otsu
from tomo_encoders.structures.voids import Voids
import cupy as cp
from tomo_encoders.neural_nets.surface_segmenter import SurfaceSegmenter
from tomo_encoders.reconstruction.recon import recon_coarse
from tomo_encoders.misc.voxel_processing import TimerGPU

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
b = 8
b_K = 8
sparse_flag = True
pixel_res = 1.17
size_um = -1 # um
void_rank = 1
radius_around_void_um = 1000.0 # um
blur_size = 0.5
BIT_DEPTH = 16
# handy code for timing stuff
# st_chkpt = cp.cuda.Event(); end_chkpt = cp.cuda.Event(); st_chkpt.record()    
# end_chkpt.record(); end_chkpt.synchronize(); t_chkpt = cp.cuda.get_elapsed_time(st_chkpt,end_chkpt)
# print(f"time checkpoint {t_chkpt/1000.0:.2f} secs")

## Output for vis
ply_lowres = '/home/atekawade/Dropbox/Arg/transfers/runtime_plots/coarse_map.ply'
ply_highres = '/home/atekawade/Dropbox/Arg/transfers/runtime_plots/high_detail_around_void_%i.ply'%void_rank
voids_highres = '/data02/MyArchive/aisteer_3Dencoders/tmp_data/voids_highres'
voids_lowres = '/data02/MyArchive/aisteer_3Dencoders/tmp_data/voids_lowres'
raw_fname = '/data02/MyArchive/aisteer_3Dencoders/tmp_data/mosaic_raw/all_layers_%ibit.hdf5'%BIT_DEPTH


def read_raw_data(fname, multiplier):
    hf = h5py.File(fname, 'r')
    crop_wdb = int(hf["data"].shape[-1]%(multiplier))
    if crop_wdb:
        sw = slice(None,-crop_wdb)
    else:
        sw = slice(None,None)
    crop_z_wdb = int(hf["data"].shape[1]%(multiplier))
    if crop_z_wdb:
        sz = slice(None,-crop_z_wdb)
    else:
        sz = slice(None,None)
    projs = np.asarray(hf["data"][:, sz, sw])    
    theta = np.asarray(hf["theta"][:])
    center = float(np.asarray(hf["center"]))
    hf.close()
    return projs, theta, center

def downsample_projs(projs, theta, center, b, b_K):

    timer = TimerGPU()
    timer.tic()
    _, nz, n = projs.shape
    projs = projs[::b_K,::b,::b].copy()
    projs = projs.astype(np.float32)
    theta = theta[::b_K,...].copy()
    center = center/b
    print(f'\tSTAT: shape of raw projection data: {projs.shape}')
    _ = timer.toc("down-sampling projection data")
    return projs, theta, center

def bin_projs(projs, theta, center, b, b_K):

    timer = TimerGPU()
    timer.tic()
    _, nz, n = projs.shape
    projs = projs[::b_K].copy()
    projs = projs.reshape(projs.shape[0], nz//b, b, n//b, b).mean(axis=(2,4))
    projs = projs.astype(np.float32)
    theta = theta[::b_K,...].copy()
    center = center/b
    print(f'\tSTAT: shape of raw projection data: {projs.shape}')
    _ = timer.toc("binning projection data")
    return projs, theta, center



def crop_projs(projs, theta, center, p_sel):

    sino_st = p_sel.points[:,0].min()
    sino_end = p_sel.points[:,0].max() + wd
    sz = slice(sino_st, sino_end)
    projs = projs[:, sz, :].astype(np.float32)
    theta = theta
    center = center
    # make sure projection shapes are divisible by the patch width (both binning and full steps)
    print(f'\tSTAT: shape of raw projection data: {projs.shape}')
    return projs, theta, center, sino_st



if __name__ == "__main__":
    t_gpu = TimerGPU()
    # initialize segmenter fCNN
    fe = SurfaceSegmenter(model_initialization = 'load-model', \
                         model_names = model_names, \
                         model_path = model_path)    
    fe.test_speeds(128,n_reps = 5, input_size = (wd,wd,wd))    

    # read data and initialize output arrays
    print("BEGIN: Read projection data from disk")
    projs, theta, center = read_raw_data(raw_fname, wd*b)
    ##### BEGIN ALGORITHM ########
    # coarse mapping
    t_gpu.tic()
    raw_data = downsample_projs(projs, theta, center, b, b_K)
    V_bin = recon_coarse(*raw_data, blur_sigma = 0, median_kernel=3)
    V_bin = segment_otsu(V_bin)    
    voids_b = Voids().count_voids(V_bin, b)    
    _ = t_gpu.toc(f'Coarse Mapping at {b}X')
    exit()
    surf = voids_b.export_void_mesh_with_texture("sizes")
    surf.write_ply(ply_lowres)

    # select roi around a void
    void_id = np.argsort(voids_b["sizes"])[-void_rank]
    voids_b.select_around_void(void_id, radius_around_void_um, pixel_size_um = pixel_res)
    print(f"\nSTEP: visualize voids in the neighborhood of void id {void_id} at full detail")    

    cp.fft.config.clear_plan_cache()
    # reconstruct and segment voxel subsets
    p_voids, r_fac = voids_b.export_grid(wd)    
    projs, theta, center, z_shift = crop_projs(projs, theta, center, p_voids)
    p_voids.points[:,0] -= z_shift
    x_voids, p_voids = process_patches(projs, theta, center, fe, p_voids, rec_min_max)
    p_voids.points[:,0] += z_shift
    
    # export voids
    voids = Voids().import_from_grid(voids_b, x_voids, p_voids)
    voids_b.write_to_disk(voids_lowres)    
    voids.write_to_disk(voids_highres)
    surf = voids.export_void_mesh_with_texture("sizes")
    surf.write_ply(ply_highres)

    # # complete: save stuff    
    # from tomo_encoders.reconstruction.recon import recon_all
    # Vx = recon_all(projs, theta, center, 32)
    # ds_save = DataFile('/data02/MyArchive/aisteer_3Dencoders/tmp_data/mosaic_x_rec', tiff = True, d_shape = Vx.shape, d_type = Vx.dtype, VERBOSITY=0)
    # Vp = np.zeros(p_voids.vol_shape, dtype = np.uint8)
    # p_voids.fill_patches_in_volume(x_voids, Vp)
    # ds_save = DataFile('/data02/MyArchive/aisteer_3Dencoders/tmp_data/mosaic_y_pred', tiff = True, d_shape = Vp.shape, d_type = np.uint8, VERBOSITY=0)
    # ds_save.create_new(overwrite=True)
    # ds_save.write_full(Vp)
    # Vp_mask = np.zeros(p_voids.vol_shape, dtype = np.uint8) # Save for illustration purposes the guessed neighborhood of the surface
    # p_voids.fill_patches_in_volume(np.ones((len(p_voids),wd,wd,wd)), Vp_mask)
    # ds_save = DataFile('/data02/MyArchive/aisteer_3Dencoders/tmp_data/mosaic_y_surf', tiff = True, d_shape = Vp_mask.shape, d_type = np.uint8, VERBOSITY=0)
    # ds_save.create_new(overwrite=True)
    # ds_save.write_full(Vp_mask)
    