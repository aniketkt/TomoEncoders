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
from tomo_encoders.neural_nets.Unet3D import build_Unet3D
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
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    
    print('just a bunch of functions')
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
