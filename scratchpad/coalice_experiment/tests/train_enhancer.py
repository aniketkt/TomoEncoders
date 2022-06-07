#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import sys 
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
from tomo_encoders.neural_nets.enhancers import Enhancer_fCNN

# import matplotlib as mpl
# mpl.use('Agg')
from enhancer_params import *
from vis_utils import show_planes
#### THIS EXPERIMENT ####
model_size = (32,32,32)
chunk_size = 64
model_tag = "M_a01"
from recon4D import SomeProjectionStream
from config import *

def fit(model_params):
    training_params = get_training_params(model_size)
    dget = SomeProjectionStream(*fnames, NTHETA_180, EXPOSURE_TIME_PER_PROJ)
    print("Time range: %.2f to %.2f seconds"%(dget.time_exposed_all.min(), dget.time_exposed_all.max()))
    print("Time per 180: 3.61 seconds")    

    time_elapsed_list = [0.0, 10.0, 50.0]
    vols = []
    for time_elapsed in time_elapsed_list:
        vols.append(dget.reconstruct_window(time_elapsed, **recon_params))
    
    fe = Enhancer_fCNN(model_initialization = 'define-new', \
                           descriptor_tag = model_tag, \
                           **model_params)    
    t0 = time.time()
    fe.train(vols, vols, **training_params)
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

    model_names = {"enhancer" : "enhancer_Unet_%s"%model_tag}
    fe = Enhancer_fCNN(model_initialization = 'load-model', \
                       model_names = model_names,\
                       model_path = model_path)    
    
    time_elapsed = 10.0
    vols = []
    
    vol = dget.reconstruct_window(time_elapsed, **recon_params)
    show_planes(vol, filetag = "original")
        
    itercount = 0
    while True:
        itercount += 1
        print("ITERATION %i"%itercount)
        
#         import pdb; pdb.set_trace()
        p = Patches(vol.shape, initialize_by = 'regular-grid', patch_size = model_size)
        x = p.extract(vol, model_size)[...,np.newaxis]
        min_max = vol[::2,::2,::2].min(), vol[::2,::2,::2].max()
        x, _ = fe.predict_patches("enhancer", x, chunk_size, None, min_max = min_max, TIMEIT = True)
        x = x[...,0]
        p.fill_patches_in_volume(x, vol)
        show_planes(vol, filetag = "enhanced_iter%02d"%itercount)
    
    return

if __name__ == "__main__":

    print("EXPERIMENT WITH MODEL %s"%model_tag)
    model_params = get_model_params(model_tag)
    
    print("EXPERIMENT WITH INPUT_SIZE = ", model_size)
    if len(sys.argv) > 1:
        if sys.argv[1] == "infer":
            infer(model_params)
        elif sys.argv[1] == "fit":
            fit(model_params)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
