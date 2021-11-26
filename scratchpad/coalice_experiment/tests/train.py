#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import sys 
import tensorflow as tf

# ######## START GPU SETTINGS ############
# to-do: find a way to limit gpu memory usage during training
# mem_limit_tf_gpu = 48.0
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#     try:
#         tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=mem_limit_tf_gpu)])
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)

# ######### END GPU SETTINGS ############

######## START GPU SETTINGS ############
########## SET MEMORY GROWTH to True ############
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass        
######### END GPU SETTINGS ############

import matplotlib.pyplot as plt 
import numpy as np 
import cupy as cp 
import time 
import h5py 
sys.path.append('../')

from tomo_encoders import DataFile, Patches
import os 
import tqdm
import pandas as pd
from tomo_encoders.neural_nets.autoencoders import SelfSupervisedCAE

# import matplotlib as mpl
# mpl.use('Agg')
from params import *

#### THIS EXPERIMENT ####
model_size = (64,64,64)
chunk_size = 32
model_tag = "M_a01"
from recon4D import SomeProjectionStream
from config import *

def fit(model_params):
    training_params = get_training_params()

    dget = SomeProjectionStream(*fnames, NTHETA_180, EXPOSURE_TIME_PER_PROJ)
    print("Time range: %.2f to %.2f seconds"%(dget.time_exposed_all.min(), dget.time_exposed_all.max()))
    print("Time per 180: 3.61 seconds")    

    time_elapsed_list = [0.0, 10.0, 50.0] #[0, 10.0, 20.0, 30.0, 60.0, 150.0]
    vols = []
    for time_elapsed in time_elapsed_list:
        vols.append(dget.reconstruct_window(time_elapsed, **recon_params))
    
    fe = SelfSupervisedCAE(model_initialization = 'define-new', \
                           model_size = model_size, \
                           descriptor_tag = model_tag, \
                           **model_params)    
    
    t0 = time.time()
    fe.train(vols, **training_params)
    t1 = time.time()
    training_time = (t1 - t0)
    print("training time per epoch = %.2f seconds"%(training_time/training_params["n_epochs"]))
    
    fe.save_models(model_path)
    sys.exit()
    
    return

def infer(model_params):
    
    dget = SomeProjectionStream(*fnames, NTHETA_180, EXPOSURE_TIME_PER_PROJ)
    print("Time range: %.2f to %.2f seconds"%(dget.time_exposed_all.min(), dget.time_exposed_all.max()))
    print("Time per 180: 3.61 seconds")    

    fe = SelfSupervisedCAE(model_initialization = 'load-model', \
                           model_size = model_size, \
                           model_tag = model_tag, model_path = model_path)    
    
    time_elapsed_list = [0.0, 10.0, 50.0]
    vols = []
    for time_elapsed in time_elapsed_list:
        vol = dget.reconstruct_window(time_elapsed, **recon_params)
    
        p = Patches(vol.shape, initialize_by = 'regular-grid', patch_size = model_size)
        x = p.extract(vol, model_size)[...,np.newaxis]
        min_max = vol[::2,::2,::2].min(), vol[::2,::2,::2].max()
        z, t_unit = fe.predict_embeddings(x, chunk_size, min_max = min_max, TIMEIT = True)

        print("z shape: ", z.shape)
        
    return


def encode_decode(model_params):
    
    dget = SomeProjectionStream(*fnames, NTHETA_180, EXPOSURE_TIME_PER_PROJ)
    print("Time range: %.2f to %.2f seconds"%(dget.time_exposed_all.min(), dget.time_exposed_all.max()))
    print("Time per 180: 3.61 seconds")    

    fe = SelfSupervisedCAE(model_initialization = 'load-model', \
                           model_size = model_size, \
                           model_tag = model_tag, model_path = model_path)    
    
    time_elapsed_list = [0.0, 10.0, 50.0]
    vols = []
    for time_elapsed in time_elapsed_list:
        vol = dget.reconstruct_window(time_elapsed, **recon_params)
        min_max = vol[::2,::2,::2].min(), vol[::2,::2,::2].max()
        
        p = Patches(vol.shape, initialize_by = 'regular-grid', patch_size = model_size)
        x = p.extract(vol, model_size)[...,np.newaxis]
        
        x, _ = fe.predict_patches("autoencoder", x, chunk_size, None, min_max = min_max, TIMEIT = True)
        x = x[...,0]
        
        import pdb; pdb.set_trace()
        vol_out = vol.copy()
        p.fill_patches_in_volume(x, vol_out)
#         show_planes(vol)
#         show_planes(vol_out)
    
    return

def show_planes(vol):
    
    fig, ax = plt.subplots(1,3, figsize = (14,6))
    ax[0].imshow(vol[int(vol.shape[0]*0.2)], cmap = 'gray')
    ax[1].imshow(vol[int(vol.shape[0]*0.5)], cmap = 'gray')
    ax[2].imshow(vol[int(vol.shape[0]*0.8)], cmap = 'gray')                
    plt.show()
    plt.close()
    

if __name__ == "__main__":

    print("EXPERIMENT WITH MODEL %s"%model_tag)
    model_params = get_model_params(model_tag)
    
    print("EXPERIMENT WITH INPUT_SIZE = ", model_size)
    if len(sys.argv) > 1:
        if sys.argv[1] == "infer":
            infer(model_params)
        elif sys.argv[1] == "fit":
            fit(model_params)
        elif sys.argv[1] == "encode-decode":
            encode_decode(model_params)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
