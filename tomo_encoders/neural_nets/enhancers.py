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

from tensorflow import keras
from tomo_encoders import Patches
from tomo_encoders import DataFile
import tensorflow as tf
from tensorflow.keras.models import load_model
from multiprocessing import Pool, cpu_count
import functools
import cupy as cp
import h5py
import abc
import time
from tomo_encoders.misc.voxel_processing import _rescale_data, _find_min_max, modified_autocontrast, normalize_volume_gpu, _edge_map
from tomo_encoders.neural_nets.keras_processor import Vox2VoxProcessor_fCNN

DEFAULT_INPUT_SIZE = None
MAX_ITERS = 1000
class Enhancer_fCNN(Vox2VoxProcessor_fCNN):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        # could be "data" or "label"
        self.input_type = "data"
        self.output_type = "data"
        return

    def random_data_generator(self, batch_size, input_size = (64,64,64)):

        while True:
            x_shape = tuple([batch_size] + list(input_size) + [1])
            x = np.random.uniform(0, 1, x_shape)#.astype(np.float32)
            y = np.random.randint(0, 2, x_shape)#.astype(np.uint8)
            x[x == 0] = 1.0e-12
            yield x, y
    
    def train(self, Xs, Ys, batch_size, \
              sampling_method, n_epochs,\
              random_rotate = False, \
              add_noise = 0.1,\
              max_stride = 1, \
              mask_ratio = 0.95, \
              training_input_size = None, steps_per_epoch = None):
        
        '''
        Parameters  
        ----------  
        '''
        n_vols = len(Xs)
        
        # instantiate data generator for use in training.  
        dg = self.data_generator(Xs, Ys, batch_size, sampling_method, \
                                 max_stride = max_stride, \
                                 random_rotate = random_rotate, \
                                 add_noise = add_noise, \
                                 mask_ratio = mask_ratio, \
                                 input_size = training_input_size)

        validation_steps_per_epoch = int(0.25*steps_per_epoch)

        
        t0 = time.time()
        self.models["enhancer"].fit(x = dg, epochs = n_epochs,\
                  steps_per_epoch=steps_per_epoch,\
                  validation_steps=validation_steps_per_epoch, verbose = 1)    
        t1 = time.time()
        training_time = (t1 - t0)
        print("training time = %.2f seconds"%training_time)        
        print("training time per epoch = %.2f seconds"%(training_time/n_epochs))
        
        return
        
        
        
    def _build_models(self, descriptor_tag = "misc", **model_params):
        '''
        
        Implementation of Segmenter_fCNN that removes blank volumes during training.  
        Parameters
        ----------
        model_keys : list  
            list of strings describing the model, e.g., ["segmenter"], etc.
        model_params : dict
            for passing any number of model hyperparameters necessary to define the model(s).
            
        '''
        if model_params is None:
            raise ValueError("Need model hyperparameters or instance of model. Neither were provided")
        else:
            self.models = {}

        # insert your model building code here. The models variable must be a dictionary of models with str descriptors as keys
            
        self.model_tag = "Unet_%s"%(descriptor_tag)

        model_key = "enhancer"
        self.models.update({model_key : None})
        # input_size here is redundant if the network is fully convolutional
        self.models[model_key] = build_Unet_3D(**model_params)
        self.models[model_key].compile(optimizer=tf.keras.optimizers.Adam(),\
                                         loss= tf.keras.losses.MeanSquaredError())
        return
        
    def get_patches(self, vol_shape,\
                    sampling_method, \
                    batch_size, \
                    max_stride = None, \
                    mask_ratio = 0.95, \
                    input_size = None):
        ip = 0
        tot_len = 0
        patches = None
        while tot_len < batch_size:
            
            # count iterations
            assert ip <= MAX_ITERS, "stuck in loop while finding patches to train on"
            ip+= 1
            
            if sampling_method in ["grid", 'regular-grid', "random-fixed-width"]:
                p_tmp = Patches(vol_shape, initialize_by = sampling_method, \
                                  patch_size = input_size, \
                                  n_points = batch_size)    
                
            elif sampling_method in ["random"]:
                p_tmp = Patches(vol_shape, initialize_by = sampling_method, \
                                  min_patch_size = input_size, \
                                  max_stride = max_stride, \
                                  n_points = batch_size)    
            else:
                raise ValueError("sampling method not supported")
            
            # to-do: insert cylindrical crop mask to avoid unreconstructed areas
            mask_ratio = 1.00
            if mask_ratio < 1.00:
                cond_list = self._find_within_cylindrical_crop(p_tmp, cutoff)
            else:
                cond_list = np.asarray([True]*len(p_tmp)).astype(bool)
            
            if np.sum(cond_list) > 0:
                # do stuff
                p_tmp = p_tmp.filter_by_condition(cond_list)

                if patches is None:
                    patches = p_tmp.copy()
                else:
                    patches.append(p_tmp)
                    tot_len = len(patches)
            else:
                continue
        
        
        assert patches is not None, "get_patches() failed to return any patches with selected conditions"
        patches = patches.select_random_sample(batch_size)
        return patches
        
    def _find_within_cylindrical_crop(self, p_tmp, mask_ratio):
        
        ystd = np.std(y_tmp, axis = (1,2,3))
        cond_list = ystd > np.max(ystd)*cutoff
        raise NotImplementedError("do this")
        return cond_list.astype(bool)
        
    def data_generator(self, Xs, Ys, batch_size, sampling_method, \
                       max_stride = 1, random_rotate = False, \
                       add_noise = 0.1, mask_ratio = 0.95, \
                       input_size = None):
        '''
        
        Parameters  
        ----------  
        vol : np.array  
            Volume from which patches are extracted.  
        batch_size : int  
            Size of the batch generated at every iteration.  
        sampling_method : str  
            Possible methods include "random", "random-fixed-width", "grid"  
        max_stride : int  
            If method is "random" or "multiple-grids", then max_stride is required.  
        
        '''
        
        while True:
            n_vols = len(Xs)
            # sample volumes
            # use _get_xy
            idx_vols = np.repeat(np.arange(0, n_vols), int(np.ceil(batch_size/n_vols)))
            idx_vols = idx_vols[:batch_size]
            
            xy = []
            for ivol in range(n_vols):
                patches = self.get_patches(Xs[ivol].shape, \
                                           sampling_method, \
                                           np.sum(idx_vols == ivol),\
                                           max_stride = max_stride, \
                                           mask_ratio = mask_ratio, \
                                           input_size = input_size)
                xy.append(self.extract_training_patch_pairs(Xs[ivol], Ys[ivol], patches, add_noise, random_rotate, input_size))
            yield np.concatenate([xy[ivol][0] for ivol in range(n_vols)], axis = 0, dtype = 'float32'), np.concatenate([xy[ivol][1] for ivol in range(n_vols)], axis = 0, dtype = 'float32')

if __name__ == "__main__":
    
    print('just a bunch of functions')
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
