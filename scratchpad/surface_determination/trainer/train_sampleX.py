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

from tomo_encoders.neural_nets.surface_segmenter import SurfaceSegmenter
from tomo_encoders.misc.feature_maps_vis import view_midplanes
from tomo_encoders.misc.voxel_processing import normalize_volume_gpu

######## START GPU SETTINGS ############
########## SET MEMORY GROWTH to True ############
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass        
######### END GPU SETTINGS ############

# INPUT SIZE CHANGE
from params import *
model_tags = ["M_a02", "M_a04", "M_a05", "M_a01", "M_a03", "M_a06"]
test_binning = 1
TRAINING_INPUT_SIZE = (64,64,64)

############## PATHS ##################
default_path = '/data02/MyArchive/tomo_datasets/ZEISS_try2/SampleX'
seg_path = '/data02/MyArchive/tomo_datasets/ZEISS_try2/SampleX_GT/GT_ANL_1X_FDK'
sz = slice(750, 1250, None)
sy = sx = slice(400, 1600, None)    


def fit(fe, Xs, Y):
    
    t0_train = time.time()
    
    fe.train(Xs, Y, \
             training_params["training_input_size"], \
             training_params["max_stride"], \
             training_params["batch_size"], \
             training_params["cutoff"], \
             training_params["random_rotate"], \
             training_params["add_noise"], \
             training_params["n_epochs"])
    
    fe.save_models(model_path)
    t1_train = time.time()
    tot_training_time = (t1_train - t0_train)/60.0
    print("\nTRAINING TIME: %.2f minutes"%tot_training_time)
    return fe

if __name__ == "__main__":

    fpaths = glob.glob(default_path + "/*")
    ds_list = [DataFile(fpath, tiff = True, VERBOSITY = 0) for fpath in fpaths]
    
    
    
    Xs = []
    for i, ds in enumerate(ds_list):
        X = ds.read_full()[sz, sy, sx].astype(np.float32)
        X = normalize_volume_gpu(X, normalize_sampling_factor = 4, chunk_size = 1).astype(np.float32)
        Xs.append(X)
        print("Done: %s"%os.path.split(ds.fname)[-1])
        
    Y = [DataFile(seg_path, tiff = True, VERBOSITY = 0).read_full()[sz, sy, sx]]
    
    
    
    training_params = get_training_params(TRAINING_INPUT_SIZE)
    for model_tag in model_tags:
        
        print("#"*55, "\nWorking on model %s\n"%model_tag, "#"*55)
        model_params = get_model_params(model_tag)
        fe = SurfaceSegmenter(model_initialization = 'define-new', \
                                 descriptor_tag = model_tag, \
                                 **model_params)
        
        fit(fe, Xs, Y)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
