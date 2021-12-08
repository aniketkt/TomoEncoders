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


if __name__ == "__main__":

    print("EXPERIMENT WITH MODEL %s"%model_tag)
    model_params = get_model_params(model_tag)
    
    training_params = get_training_params(TRAINING_MIN_INPUT_SIZE, \
                                          N_EPOCHS = N_EPOCHS, \
                                          N_STEPS_PER_EPOCH = N_STEPS_PER_EPOCH, \
                                          BATCH_SIZE = TRAINING_BATCH_SIZE)

    all_tsteps = get_tsteps(fpath)
    sel_tsteps = all_tsteps[:1]
    vol = load_datasets(fpath, tsteps = sel_tsteps)[0]
    
    fe = FeatureExtraction_fCNN(model_initialization = 'define-new', \
                           descriptor_tag = model_tag, \
                           **model_params)    

    p = Patches(vol.shape, initialize_by = 'regular-grid', patch_size = INFERENCE_INPUT_SIZE)
    p = p.filter_by_cylindrical_mask()
#     emb_vec = fe.predict_embeddings(x, n_features)
    
#     show_planes(Xs[0], filetag = "training_input")
#     fe.print_layers('enhancer')
    
    
    
    
    
    
    
    
    
    
    
