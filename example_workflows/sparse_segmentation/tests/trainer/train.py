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
import time
from tomo_encoders.tasks import SparseSegmenter, load_dataset_pairs
from tomo_encoders.misc_utils.feature_maps_vis import view_midplanes

# descriptor_tag = 'tmp'#'test_noblanks_pt2cutoff_nostd'
from datasets import get_datasets, dataset_names

# INPUT SIZE CHANGE
from params import *
# model_tags = ["M_a02", "M_a04", "M_a05", "M_a01", "M_a03"]
model_tags = ["M_a01"]
test_binning = 1
TRAINING_INPUT_SIZE = (64,64,64)

def fit(fe, Xs, Ys):
    
    t0_train = time.time()
    fe.train(Xs, Ys, training_params["batch_size"], \
             training_params["sampling_method"], \
             training_params["n_epochs"], \
             max_stride = training_params["max_stride"], \
             random_rotate = training_params["random_rotate"], \
             add_noise = training_params["add_noise"], \
             cutoff = training_params["cutoff"])
    fe.save_models(model_path)
    t1_train = time.time()
    tot_training_time = (t1_train - t0_train)/60.0
    fe.tf_session.close()
    print("\nTRAINING TIME: %.2f minutes"%tot_training_time)
    return fe

def infer(fe, Xs, Ys):
    return

if __name__ == "__main__":

    
    datasets = get_datasets(dataset_names[:], test_binning = test_binning)
    print(datasets.keys())
    Xs, Ys = load_dataset_pairs(datasets)
    
    training_params = get_training_params(TRAINING_INPUT_SIZE)
    for model_tag in model_tags:
        
        print("#"*55, "\nWorking on model %s\n"%model_tag, "#"*55)
        model_params = get_model_params(model_tag)
        fe = SparseSegmenter(model_initialization = 'define-new', \
                                 input_size = training_params["training_input_size"], \
                                 descriptor_tag = model_tag, \
                                 **model_params)
        
        fit(fe, Xs, Ys)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
