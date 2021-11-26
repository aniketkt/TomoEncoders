#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import sys 
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np 
import cupy as cp 
import time 
import h5py 
from tomo_encoders import DataFile, Patches

import os 
import tqdm
import pandas as pd
from tomo_encoders.neural_nets.autoencoders import SelfSupervisedCAE
import matplotlib as mpl
mpl.use('Agg')
from params import *


#### THIS EXPERIMENT ####
model_size = (64,64,64)
chunk_size = 32
model_tag = "M_a01"

def fit(fe):
    training_params = get_training_params()
    batch_size = training_params["batch_size"]
    
    # to-do: load data and normalize it
    dg = fe.random_data_generator(batch_size)
    vols = np.random.normal(0, 1, (3, 600, 960, 960))
    
    t0 = time.time()
    fe.train(vols, **training_params)
    t1 = time.time()
    training_time = (t1 - t0)
    print("training time per epoch = %.2f seconds"%(training_time/n_epochs))        
    return
    
if __name__ == "__main__":

    print("EXPERIMENT WITH MODEL %s"%model_tag)
    print("EXPERIMENT WITH INPUT_SIZE = ", model_size)
    model_params = get_model_params(model_tag)
    
    fe = SelfSupervisedCAE(model_initialization = 'define-new', \
                           model_size = model_size, \
                           descriptor_tag = model_tag, \
                           **model_params)    
    fit(fe)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
