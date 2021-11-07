#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import sys
from tomo_encoders import Patches
from tomo_encoders import DataFile
import tensorflow as tf
import time
from tomo_encoders.tasks import SparseSegmenter
from tomo_encoders.misc_utils.feature_maps_vis import view_midplanes

# to-do: get these inputs from command line or config file

######### RUNTIME GPU USAGE ###########
GPU_mem_limit = 48.0
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU_mem_limit*1000.0)])
    except RuntimeError as e:
        print(e)        


########## CROPPING AND BINNING ##########        
#### to-do: apply this as filter on patches ####
data_path = '/data02/MyArchive/AM_part_Xuan' #ensure this path matches where your data is located.

test_binning = 2
# load vols here and quick look
dict_scrops = {'mli_L206_HT_650_L3' : (slice(100,-100, test_binning), \
                                    slice(None,None, test_binning), \
                                    slice(None,None, test_binning)), \
            'AM316_L205_fs_tomo_L5' : (slice(50,-50, test_binning), \
                                       slice(None,None, test_binning), \
                                       slice(None,None, test_binning))}

# create datasets input for train method
datasets = {}
for filename, s_crops in dict_scrops.items():
    ct_fpath = os.path.join(data_path, 'data', \
                            filename + '_rec_1x1_uint16.hdf5')
    seg_fpath = os.path.join(data_path, 'seg_data', \
                             filename, filename + '_GT.hdf5')
    
    datasets.update({filename : {'fpath_X' : ct_fpath, \
                                 'fpath_Y' : seg_fpath, \
                                 'data_tag_X' : 'data', \
                                 'data_tag_Y' : 'SEG', \
                                 's_crops' : s_crops}})



############ MODEL PARAMETERS ############
model_path = '/data02/MyArchive/aisteer_3Dencoders/models/AM_part_segmenter'
descriptor_tag = 'tmp'#'test_noblanks_pt2cutoff_nostd'

model_size = (64,64,64)
model_params = {"n_filters" : [32, 64],\
                "n_blocks" : 2,\
                "activation" : 'lrelu',\
                "batch_norm" : True,\
                "isconcat" : [True, True],\
                "pool_size" : [2,4],\
                "stdinput" : False}

training_params = {"sampling_method" : "random", \
                   "batch_size" : 24, \
                   "n_epochs" : 30,\
                   "random_rotate" : True, \
                   "add_noise" : 0.05, \
                   "max_stride" : 4, \
                   "cutoff" : 0.2}

    
if __name__ == "__main__":

    fe = SparseSegmenter(model_initialization = 'define-new', \
                             model_size = model_size, \
                             descriptor_tag = descriptor_tag, \
                             **model_params)

    Xs, Ys = fe.load_datasets(datasets)

    fe.train(Xs, Ys, training_params["batch_size"], \
             training_params["sampling_method"], \
             training_params["n_epochs"], \
             max_stride = training_params["max_stride"], \
             random_rotate = training_params["random_rotate"], \
             add_noise = training_params["add_noise"], \
             cutoff = training_params["cutoff"])
    fe.save_models(model_path)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
