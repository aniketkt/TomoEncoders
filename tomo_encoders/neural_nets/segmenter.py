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

from tensorflow.keras import layers as L
from tensorflow import keras
from tomo_encoders.neural_nets.porosity_encoders import analysis_block, synthesis_block, custom_Conv3D
from tomo_encoders import Patches
from tomo_encoders import DataFile
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import UpSampling3D
from multiprocessing import Pool, cpu_count
import functools
import cupy as cp
import h5py
import abc
import time
from tomo_encoders.misc.voxel_processing import _rescale_data, _find_min_max, modified_autocontrast, normalize_volume_gpu, _edge_map
from tomo_encoders.neural_nets.keras_processor import Vox2VoxProcessor_fCNN



# Parameters for weighted cross-entropy and focal loss - alpha is higher than 0.5 to emphasize loss in "ones" or metal pixels.
eps = 1e-12
alpha = 0.75
gamma = 2.0
DEFAULT_INPUT_SIZE = (64,64,64)

def _binary_lossmap(y_true, y_pred):
    # y_true, y_pred are tensors of shape (batch_size, img_h, img_w, n_channels)
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return pt_1, pt_0

def focal_loss(y_true, y_pred):
    """
    :return: loss value
    
    Focal loss is defined here: https://arxiv.org/abs/1708.02002
    Using this provides improved fidelity in unbalanced datasets: 
    Tekawade et al. https://doi.org/10.1117/12.2540442
    
    
    Parameters
    ----------
    
    y_true  : tensor
            Ground truth tensor of shape (batch_size, n_rows, n_cols, n_channels)
    y_pred  : tensor
            Predicted tensor of shape (batch_size, n_rows, n_cols, n_channels)
    
    """

    pt_1, pt_0 = _binary_lossmap(y_true, y_pred)
    loss_map = -alpha*tf.math.log(pt_1 + eps)*tf.math.pow(1. - pt_1,gamma) - (1-alpha)*tf.math.log(1. - pt_0 + eps)*tf.math.pow(pt_0,gamma)
    return tf.reduce_mean(loss_map)

class Segmenter_fCNN(Vox2VoxProcessor_fCNN):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
        # could be "data" or "label"
        self.input_type = "data"
        self.output_type = "labels"
        return

    def random_data_generator(self, batch_size, input_size = (64,64,64)):

        while True:

            x_shape = tuple([batch_size] + list(input_size) + [1])
            x = np.random.uniform(0, 1, x_shape)#.astype(np.float32)
            y = np.random.randint(0, 2, x_shape)#.astype(np.uint8)
            x[x == 0] = 1.0e-12
            yield x, y
    
    
def build_Unet_3D(n_filters = [16,32,64], \
                  n_blocks = 3, activation = 'lrelu',\
                  batch_norm = True, kern_size = 3, kern_size_upconv = 2,\
                  isconcat = None, pool_size = 2):
    """
    Define a 3D convolutional Unet, based on the arguments provided. Output image size is the same as input image size.  
    
    Returns
    -------
    tf.Keras.model
        keras model(s) for a 3D autoencoder-decoder architecture.  
        
    Parameters
    ----------
    vol_shape  : tuple
            input volume shape (nz,ny,nx,1)  
            
    n_filters : list
            a list of the number of filters in the convolutional layers for each block. Length must equal number of number of blocks.  
            
    n_blocks  : int
            Number of repeating blocks in the convolutional part  
            
    activation : str or tf.Keras.layers.Activation
            name of custom activation or Keras activation layer  
            
    batch_norm : bool
            True to insert BN layer after the convolutional layers  
            
    kern_size  : tuple
            kernel size for conv. layers in downsampling block, e.g. (3,3,3).  
            
    kern_size_upconv  : tuple
            kernel size for conv. layers in upsampling block, e.g. (2,2,2).  
            
    isconcat : bool or list
            Selectively concatenate layers (skip connections)  
    
    pool_size : int or list
            if list, list length must be equal to number of blocks.  
            
    """
    
    inp = L.Input((None,None,None,1))
    
    if isconcat is None:
        isconcat = [False]*n_blocks
    
    if type(pool_size) is int:
        pool_size = [pool_size]*n_blocks
    elif len(pool_size) != n_blocks:
        raise ValueError("list length must be equal to number of blocks")
        
    concats = []
    # downsampling path. e.g. n_blocks = 3, n_filters = [16,32,64], input volume is 64^3
    for ii in range(n_blocks): # 3 iterations
        
        if ii == 0:
            code = inp
            
        code, concat_tensor = analysis_block(code, \
                                             n_filters[ii], \
                                             pool_size[ii], \
                                             kern_size = kern_size, \
                                             activation = activation, \
                                             batch_norm = batch_norm)
        concats.append(concat_tensor)

    nf = code.shape[-1]
    code = custom_Conv3D(code, nf, kern_size, \
                         activation = activation, batch_norm = batch_norm)
    decoded = custom_Conv3D(code, 2*nf, kern_size, \
                         activation = activation, batch_norm = batch_norm)    
    
    # upsampling path. e.g. n_blocks = 3
    for ii in range(n_blocks-1, -1, -1):
        # ii is 2, 1, 0
#         print("############# ii = %i"%ii)
        
        decoded = synthesis_block(decoded, \
                                  2*n_filters[ii], \
                                  pool_size[ii], \
                                  concat_tensor = concats[ii], \
                                  activation = activation, \
                                  kern_size = kern_size, \
                                  kern_size_upconv = kern_size_upconv, \
                                  batch_norm = batch_norm, \
                                  concat_flag = isconcat[ii])
        
    decoded = L.Conv3D(1, (1,1,1), activation = 'sigmoid', padding = "same")(decoded)
    
    segmenter = keras.models.Model(inp, decoded, name = "segmenter")
    
    return segmenter
    
    
    
    
    
    
    
if __name__ == "__main__":
    
    print('just a bunch of functions')
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
