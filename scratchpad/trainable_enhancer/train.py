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

##### FILE PATHS ############
from tifffile import imread
fname_X = '/data02/MyArchive/solder_imaging/25.tif'
fname_Y = '/data02/MyArchive/solder_imaging/25_enhanced.tif'
fname_X_test = '/data02/MyArchive/solder_imaging/35.tif'
from tomo_encoders.misc.viewer import view_midplanes
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
from params import *
from vis_utils import show_planes
#### THIS EXPERIMENT ####
INPUT_SIZE = (128,128,128)
INFERENCE_BATCH_SIZE = 4

N_EPOCHS = 10 
N_STEPS_PER_EPOCH = 20 
TRAINING_BATCH_SIZE = 8
model_tag = "M_a06"

from config import *

def load_volume(fname):
    vol = imread(fname).astype(np.float32)
    
    min_val = vol.min()
    max_val = vol.max()
    
    assert max_val - min_val > 1.0e-12, "max - min < 1.0e-12. Is this volume empty?"
    vol = (vol - min_val) / (max_val - min_val)
    return vol

def fit(model_params):
    training_params = get_training_params(INPUT_SIZE, \
                                          N_EPOCHS = N_EPOCHS, \
                                          N_STEPS_PER_EPOCH = N_STEPS_PER_EPOCH, \
                                          BATCH_SIZE = TRAINING_BATCH_SIZE)

    Xs = [load_volume(fname_X)]
    Ys = [load_volume(fname_Y)]
    
    show_planes(Xs[0], filetag = "training_input")
    show_planes(Ys[0], filetag = "training_target")
    
    assert all([Xs[i].shape == Ys[i].shape for i in range(len(Xs))])
    
    
    fe = Enhancer_fCNN(model_initialization = 'define-new', \
                           descriptor_tag = model_tag, \
                           **model_params)    
    
    fe.print_layers('enhancer')
    
    
    fe.train(Xs, Ys, **training_params)
    fe.save_models(model_path)
    sys.exit()
    return

def infer(model_params):
    
    model_names = {"enhancer" : "enhancer_Unet_%s"%model_tag}
    fe = Enhancer_fCNN(model_initialization = 'load-model', \
                       model_names = model_names,\
                       model_path = model_path)    
    
    time_elapsed = 10.0
    vol = load_volume(fname_X_test)
    show_planes(vol, filetag = "original")
    
    itercount = 0
    while True:
        itercount += 1
        print("ITERATION %i"%itercount)
        
        p = Patches(vol.shape, initialize_by = 'grid', patch_size = INPUT_SIZE)
        x = p.extract(vol, INPUT_SIZE)[...,np.newaxis]
        min_max = vol[::2,::2,::2].min(), vol[::2,::2,::2].max()
        x, _ = fe.predict_patches("enhancer", x, INFERENCE_BATCH_SIZE, None, min_max = min_max, TIMEIT = True)
        x = x[...,0]
        
        crop_val = 5
        sc = slice(crop_val, -crop_val, None)
        x = x[:,sc,sc,sc]
        p.widths = p.widths - 2*crop_val
        p.points = p.points + crop_val
        
        vol_out = np.zeros(vol.shape, dtype = np.float32) # ERASE OLD VOLUME !!!!        
        p.fill_patches_in_volume(x, vol_out)
        show_planes(vol_out, filetag = "enhanced_iter%02d"%itercount)
        
        break
        
        
    return

if __name__ == "__main__":

    print("EXPERIMENT WITH MODEL %s"%model_tag)
    model_params = get_model_params(model_tag)
    
    print("EXPERIMENT WITH INPUT_SIZE = ", INPUT_SIZE)
    if len(sys.argv) > 1:
        if sys.argv[1] == "infer":
            infer(model_params)
        elif sys.argv[1] == "fit":
            fit(model_params)
    
    
    
    
    
    
    
    
    
    
    
