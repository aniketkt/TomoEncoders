#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import sys
from tomo_encoders import Patches, DataFile
import tensorflow as tf
import time, glob

sys.path.append('/data02/MyArchive/aisteer_3Dencoders/TomoEncoders/scratchpad/surface_determination/trainer')
from tomo_encoders.neural_nets.surface_segmenter import SurfaceSegmenter
from tomo_encoders.misc.feature_maps_vis import view_midplanes
from tomo_encoders.misc.voxel_processing import normalize_volume_gpu
from tomo_encoders.misc.viewer import view_midplanes

model_tags = ["M_a0%i"%i for i in range(1,7)]

default_path = '/data02/MyArchive/tomo_datasets/ZEISS_try2/Sample2/'
seg_path = '/data02/MyArchive/tomo_datasets/ZEISS_try2/Sample2_SEG/'
save_path = '/data02/MyArchive/tomo_datasets/ZEISS_try2/Sample2_CT/'
if not os.path.exists(seg_path):
    os.makedirs(seg_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

plot_path = '/home/atekawade/Dropbox/Arg/transfers/ZEISS_v2'
sz = slice(800, 1500, None)
sy = sx = slice(200, 1900, None)    
from params import model_path, get_model_params
TRAINING_INPUT_SIZE = (64,64,64)

if __name__ == "__main__":
    
    fpaths = glob.glob(default_path + "/*")
    ds_list = [DataFile(fpath, tiff = True, VERBOSITY = 0) for fpath in fpaths]
    
    for ds in ds_list:

        data_tag = os.path.split(ds.fname)[-1]
        print("#"*55)
        print(f'Working on data: {data_tag}')
        print("#"*55)
        
        vol = ds.read_full().astype(np.float32)
        img = vol[-1].copy()
        vol[vol == 0] = np.mean(img[500:1500,500:1500])
        vol = vol[sz, sy, sx].astype(np.float32)
        vol = normalize_volume_gpu(vol, normalize_sampling_factor = 4, chunk_size = 1).astype(np.float32)        

        # save tiff 
        save_fname = os.path.join(save_path, f'{data_tag}')
        ds_save = DataFile(save_fname, tiff = True, d_shape = vol.shape, d_type = np.float32) 
        ds_save.create_new(overwrite = True)
        ds_save.write_full(vol)
        
        for model_tag in model_tags:
            
            if model_tag == "M_a04":
                continue
            
            # load model
            print("#"*55, "\nWorking on model %s\n"%model_tag, "#"*55)
            model_names = {"segmenter" : "segmenter_Unet_%s"%model_tag}

            model_params = get_model_params(model_tag)
            fe = SurfaceSegmenter(model_initialization = 'load-model', \
                                  model_names = model_names, \
                                  model_path = model_path)            
            
            # Segmentation step
            t00 = time.time()
            p_grid = Patches(vol.shape, initialize_by = 'grid', patch_size = TRAINING_INPUT_SIZE)
            x = p_grid.extract(vol, TRAINING_INPUT_SIZE)
            y_pred = fe.predict_patches("segmenter", x[...,np.newaxis], 32, None)[...,0]
            vol_seg = np.zeros_like(vol)
            p_grid.fill_patches_in_volume(y_pred, vol_seg)
            t01 = time.time()            
            print(f'total segmentation time: {t01-t00:.2f} seconds')
            
            # save plot
            fig, ax = plt.subplots(1,3, figsize = (12,6))
            view_midplanes(vol_seg, ax = ax)   
            plt.savefig(os.path.join(plot_path, f'{data_tag}_{model_tag}.png'))
            plt.close()
            
            # save tiff 
            seg_fname = os.path.join(seg_path, f'{data_tag}_{model_tag}')
            ds_seg = DataFile(seg_fname, tiff = True, d_shape = vol_seg.shape, d_type = np.uint8) 
            ds_seg.create_new(overwrite = True)
            ds_seg.write_full(vol_seg)
            
        
        
        
        

    
    
    
    
    
    
    
