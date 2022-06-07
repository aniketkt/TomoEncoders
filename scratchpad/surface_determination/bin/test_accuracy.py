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

seg_path = '/data02/MyArchive/tomo_datasets/ZEISS_try2/Sample2_SEG/'
ct_path = '/data02/MyArchive/tomo_datasets/ZEISS_try2/Sample2_CT/'

plot_path = '/home/atekawade/Dropbox/Arg/transfers/ZEISS_v2'

if __name__ == "__main__":
    
    fpaths = glob.glob(ct_path + "/*")
    ds_list = [DataFile(fpath, tiff = True, VERBOSITY = 0) for fpath in fpaths]
    
    for ds in ds_list:

        data_tag = os.path.split(ds.fname)[-1]
        print("#"*55)
        print(f'Working on data: {data_tag}')
        print("#"*55)
        
        vol = ds.read_full().astype(np.float32)

        for model_tag in model_tags:
            
            if model_tag == "M_a04":
                continue
            
            # load model
            print("#"*55, "\nWorking on model %s\n"%model_tag, "#"*55)
            
            # Segmentation step
            t00 = time.time()
            
#             p_grid = Patches(vol.shape, initialize_by = 'grid', patch_size = TRAINING_INPUT_SIZE)
#             x = p_grid.extract(vol, TRAINING_INPUT_SIZE)
#             y_pred = fe.predict_patches("segmenter", x[...,np.newaxis], 32, None)[...,0]
#             vol_seg = np.zeros_like(vol)
#             p_grid.fill_patches_in_volume(y_pred, vol_seg)
            
            
    
            t01 = time.time()            
            print(f'total segmentation time: {t01-t00:.2f} seconds')
            
            
        
        
        
        

    
    
    
    
    
    
    
