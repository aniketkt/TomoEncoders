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
from surface_determination import guess_surface, determine_surface, Voids
from tomo_encoders import DataFile
import cupy as cp
from tomo_encoders.neural_nets.surface_segmenter import SurfaceSegmenter
import vedo
from tomo_encoders.mesh_processing.vox2mesh import make_void_mesh
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
pixel_res = 1.17
size_thresh = 3.0
# handy code for timing stuff
# st_chkpt = cp.cuda.Event(); end_chkpt = cp.cuda.Event(); st_chkpt.record()    
# end_chkpt.record(); end_chkpt.synchronize(); t_chkpt = cp.cuda.get_elapsed_time(st_chkpt,end_chkpt)
# print(f"time checkpoint {t_chkpt/1000.0:.2f} secs")

## Output for vis
ply_lowres = '/home/atekawade/Dropbox/Arg/transfers/runtime_plots/mesh_lowres.ply'
ply_highres = '/home/atekawade/Dropbox/Arg/transfers/runtime_plots/mesh_highres.ply'

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
    voids_b = Voids().guess_voids(projs, theta, center, b, b_K)    
    voids_b.select_by_size(3.0)
    surf = voids_b.export_surface(rescale_fac = b)
    surf.write_ply(ply_lowres)

    # guess roi around a void
    void_id = np.argsort(voids_b["sizes"])[-2]
    pix_radius = int(np.cbrt(voids_b["sizes"][void_id])*2)
    voids_b.select_around_void(void_id, pix_radius)
    p_sel, r_fac = voids_b.export_grid(wd//b)
    p_sel = p_sel.rescale(b)
    


    cp.fft.config.clear_plan_cache()
    
    # determine surface
    # print("STEP: determine surface")
    start_determine = cp.cuda.Event(); end_determine = cp.cuda.Event(); start_determine.record()
    x_voids, p_voids = determine_surface(projs, theta, center, fe, p_sel)
    end_determine.record(); end_determine.synchronize(); t_determine = cp.cuda.get_elapsed_time(start_determine,end_determine)
    print(f'TIME: determining surface: {t_determine/1000.0:.2f} seconds')

    # export voids to vis
    st_chkpt = cp.cuda.Event(); end_chkpt = cp.cuda.Event(); st_chkpt.record()    
    voids = Voids().import_from_grid(voids_b, b, x_voids, p_voids)
    surf = voids.export_surface(decimate_fac = 0.1)
    surf.write_ply(ply_highres)
    end_chkpt.record(); end_chkpt.synchronize(); t_vis = cp.cuda.get_elapsed_time(st_chkpt,end_chkpt)
    print(f"TIME: export vis: {t_vis/1000.0:.2f} secs")


    # complete: save stuff    
    Vp = np.zeros(p_voids.vol_shape, dtype = np.uint8)
    p_voids.fill_patches_in_volume(x_voids, Vp)
    ds_save = DataFile('/data02/MyArchive/aisteer_3Dencoders/tmp_data/test_y_pred', tiff = True, d_shape = Vp.shape, d_type = np.uint8, VERBOSITY=0)
    ds_save.create_new(overwrite=True)
    ds_save.write_full(Vp)
    Vp_mask = np.zeros(p_voids.vol_shape, dtype = np.uint8) # Save for illustration purposes the guessed neighborhood of the surface
    p_voids.fill_patches_in_volume(np.ones((len(p_voids),wd,wd,wd)), Vp_mask)
    ds_save = DataFile('/data02/MyArchive/aisteer_3Dencoders/tmp_data/test_y_surf', tiff = True, d_shape = Vp_mask.shape, d_type = np.uint8, VERBOSITY=0)
    ds_save.create_new(overwrite=True)
    ds_save.write_full(Vp_mask)
    
