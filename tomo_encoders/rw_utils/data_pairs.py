#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class implementations for real-time 3D feature extraction


"""
from abc import abstractmethod
import pandas as pd
import os
import glob
import numpy as np


from tomo_encoders import Patches
from tomo_encoders import DataFile
import cupy as cp
import h5py
import abc
import time
from tomo_encoders.misc.voxel_processing import normalize_volume_gpu


# Parameters for weighted cross-entropy and focal loss - alpha is higher than 0.5 to emphasize loss in "ones" or metal pixels.



def _msg_exec_time(func, t_exec):
    print("TIME: %s: %.2f seconds"%(func.__name__, t_exec))
    return

def read_data_pair(ds_X, ds_Y, s_crops, normalize_sampling_factor):

    print("loading data...")

#     X = ds_X.read_data(slice_3D = s_crops).astype(np.float32)
#     Y = ds_Y.read_data(slice_3D = s_crops).astype(np.uint8)
    
    X = ds_X.read_full().astype(np.float32)
    Y = ds_Y.read_full().astype(np.uint8)
    X = X[s_crops].copy()
    Y = Y[s_crops].copy()

    # normalize volume, check if shape is compatible.  
    X = normalize_volume_gpu(X, normalize_sampling_factor = normalize_sampling_factor, chunk_size = 1).astype(np.float16)

    print("done")
    print("Shape X %s, shape Y %s"%(str(X.shape), str(Y.shape)))
    return X, Y

def load_dataset_pairs(datasets, normalize_sampling_factor = 4, TIMEIT = False):

    
    '''
    load datasets using DataFile objects for X and Y. Multiple dataset pairs can be loaded.  
    '''
    TIMEIT = True
    t0 = time.time()
    n_vols = len(datasets)

    Xs = [0]*n_vols
    Ys = [0]*n_vols
    ii = 0
    for filename, dataset in datasets.items():

        ds_X = DataFile(dataset['fpath_X'], autodetect_format = True, \
                        data_tag = dataset['data_tag_X'], VERBOSITY = 0)

        ds_Y = DataFile(dataset['fpath_Y'], autodetect_format = True, \
                        data_tag = dataset['data_tag_Y'], VERBOSITY = 0)

        Xs[ii], Ys[ii] = read_data_pair(ds_X, ds_Y, dataset['s_crops'], normalize_sampling_factor)
        ii += 1
    del ii
    if TIMEIT:
        t_exec = float(time.time() - t0)
        _msg_exec_time(load_dataset_pairs, t_exec)
    
    
    print("DATASET SHAPES: \n")
    for ip in range(len(Xs)):
        print("dataset %i: "%(ip+1), " -- ", Xs[ip].shape)
                                
    return Xs, Ys


if __name__ == "__main__":
    
    print('just a bunch of functions')
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
