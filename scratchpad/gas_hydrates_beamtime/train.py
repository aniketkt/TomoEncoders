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
from feature_extractor import FeatureExtraction_fCNN

# import matplotlib as mpl
# mpl.use('Agg')
from params import *
from vis_utils import show_planes
from config import *

def fit(model_params):
    training_params = get_training_params(TRAINING_MIN_INPUT_SIZE, \
                                          N_EPOCHS = N_EPOCHS, \
                                          N_STEPS_PER_EPOCH = N_STEPS_PER_EPOCH, \
                                          BATCH_SIZE = TRAINING_BATCH_SIZE)

    all_tsteps = get_tsteps(fpath)
    sel_tsteps = all_tsteps[::40]
    Xs = load_datasets(fpath, tsteps = sel_tsteps)
    Ys = [X.copy() for X in Xs] # apply some contrast filtering or segmentation here?
    
    show_planes(Xs[0], filetag = "training_input")
    show_planes(Ys[0], filetag = "training_target")
    
    assert all([Xs[i].shape == Ys[i].shape for i in range(len(Xs))])
    
    fe = FeatureExtraction_fCNN(model_initialization = 'define-new', \
                           descriptor_tag = model_tag, \
                           **model_params)    
    
    fe.print_layers('enhancer')
    
    fe.train(Xs, Ys, **training_params)
    fe.save_models(model_path)
    sys.exit()
    return

def infer(model_params):
    
    model_names = {"enhancer" : "enhancer_Unet_%s"%model_tag}
    fe = FeatureExtraction_fCNN(model_initialization = 'load-model', \
                       model_names = model_names,\
                       model_path = model_path)    
    
    all_tsteps = get_tsteps(fpath)
    
    vol = load_datasets(fpath, tsteps = all_tsteps[:1])[0]
    p = Patches(vol.shape, initialize_by = 'regular-grid', patch_size = INFERENCE_INPUT_SIZE)
    p = p.filter_by_cylindrical_mask()
    
    emb_vecs = []
    for ii, tstep in enumerate(tqdm.tqdm(all_tsteps)):

        vol = load_datasets(fpath, tsteps = [tstep])[0]
        x = p.extract(vol, INFERENCE_INPUT_SIZE)[...,np.newaxis]
        
        emb_vec = fe.predict_embeddings(x[...,np.newaxis], 'leaky_re_lu_3', 32, TIMEIT = True)  
        
        
        # do more stuff with this - visualize, send to epics, etc.
        emb_vecs.append(emb_vec)
        
    return

if __name__ == "__main__":

    print("EXPERIMENT WITH MODEL %s"%model_tag)
    model_params = get_model_params(model_tag)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "infer":
            infer(model_params)
        elif sys.argv[1] == "fit":
            fit(model_params)
    
    
    
    
    
    
    
    
    
    
    
