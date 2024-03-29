#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class implementations for real-time 3D feature extraction


"""

import pandas as pd
import os
import glob
import numpy as np


# from skimage.feature import match_template
# from tomopy import normalize, minus_log, angles, recon, circ_mask
# from scipy.ndimage.filters import median_filter

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

MAX_ITERS = 2000 # iteration max for find_patches(). Will raise warnings if count is exceeded.
# Parameters for weighted cross-entropy and focal loss - alpha is higher than 0.5 to emphasize loss in "ones" or metal pixels.

from tomo_encoders.neural_nets.keras_processor import Vox2VoxProcessor_fCNN
from tomo_encoders.neural_nets.Unet3D import build_Unet_3D


from tomo_encoders.rw_utils.data_pairs import read_data_pair, load_dataset_pairs
from tomo_encoders.misc.voxel_processing import _rescale_data, _find_min_max, modified_autocontrast, normalize_volume_gpu, _edge_map



class SparseSegmenter(Vox2VoxProcessor_fCNN):
    
    def __init__(self,**kwargs):
        
        super().__init__(**kwargs)
        return
    
    def train(self, Xs, Ys, batch_size, \
              sampling_method, n_epochs,\
              random_rotate = False, \
              add_noise = 0.1,\
              max_stride = 1, \
              cutoff = 0.0):
        
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
                                 cutoff = cutoff)
        tot_steps = 1000
        val_split = 0.2
        steps_per_epoch = int((1-val_split)*tot_steps//batch_size)
        validation_steps = int(val_split*tot_steps//batch_size)

        t0 = time.time()
        self.models["segmenter"].fit(x = dg, epochs = n_epochs,\
                  steps_per_epoch=steps_per_epoch,\
                  validation_steps=validation_steps, verbose = 1)    
        t1 = time.time()
        training_time = (t1 - t0)
        print("training time = %.2f seconds"%training_time)        
        
        return
    
    def _find_blanks(self, patches, cutoff, Y_gt, input_size = (64,64,64)):
        
        assert Y_gt.shape == patches.vol_shape, "volume of Y_gt does not match vol_shape"
        y_tmp = patches.extract(Y_gt, input_size)[...,np.newaxis]
        ystd = np.std(y_tmp, axis = (1,2,3))

        cond_list = ystd > np.max(ystd)*cutoff
        return cond_list.astype(bool)
    
    def get_patches(self, vol_shape,\
                    sampling_method, \
                    batch_size, \
                    max_stride = None, \
                    cutoff = 0.0, Y_gt = None, input_size = (64,64,64)):
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
            
            # blank removal - avoid if not requested.
            # to-do: how fast is blank removal?
            if cutoff > 0.0:
                cond_list = self._find_blanks(p_tmp, cutoff, Y_gt)
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

    def save_models(self, model_path):
        
        model_key = "segmenter"
        model = self.models[model_key]
        filepath = os.path.join(model_path, "%s_%s.hdf5"%(model_key, self.model_tag))
        tf.keras.models.save_model(model, filepath, include_optimizer=False)        
        return
    
    def _load_models(self, model_names = None, model_path = 'some/path'):
        
        '''
        Parameters
        ----------
        model_names : dict
            example {"segmenter" : "Unet"}
        model_path : str  
            example "some/path"
        custom_objects_dict : dict  
            dictionary of custom objects (usually pickled with the keras models)
            
        '''
        self.models = {} # clears any existing models linked to this class!!!!
        for model_key, model_name in model_names.items():
            self.models.update({model_key : \
                                load_model(os.path.join(model_path, \
                                                        model_name + '.hdf5'))})
        model_key = "segmenter"
        self.model_tag = "_".join(model_names[model_key].split("_")[1:])
        return
    

    def data_generator(self, Xs, Ys, batch_size, sampling_method, max_stride = 1, random_rotate = False, add_noise = 0.1, cutoff = 0.0, return_patches = False, TIMEIT = False):

        
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
                                           cutoff = cutoff,\
                                           Y_gt = Ys[ivol])
                xy.append(self.extract_training_patch_pairs(Xs[ivol], \
                                                            Ys[ivol], \
                                                            patches, \
                                                            add_noise, \
                                                            random_rotate))
                
            
            yield np.concatenate([xy[ivol][0] for ivol in range(n_vols)], axis = 0, dtype = 'float32'), np.concatenate([xy[ivol][1] for ivol in range(n_vols)], axis = 0, dtype = 'uint8')

    
    def predict_patches(self, model_key, x, chunk_size, out_arr, \
                         min_max = None, \
                         TIMEIT = False):

        '''
        Predicts sub_vols. This is a wrapper around keras.model.predict() that speeds up inference on inputs lengths that are not factors of 2. Use this function to do multiprocessing if necessary.  
        
        '''
        assert x.ndim == 5, "x must be 5-dimensional (batch_size, nz, ny, nx, 1)."
        
        t0 = time.time()
#         print("call to predict_patches, len(x) = %i, shape = %s, chunk_size = %i"%(len(x), str(x.shape[1:-1]), chunk_size))
        nb = len(x)
        nchunks = int(np.ceil(nb/chunk_size))
        nb_padded = nchunks*chunk_size
        padding = nb_padded - nb

        if out_arr is None:
            out_arr = np.empty_like(x) # use numpy since return from predict is numpy
        else:
            # to-do: check dims
            assert out_arr.shape == x.shape, "x and out_arr shapes must be equal and 4-dimensional (batch_size, nz, ny, nx, 1)"

        for k in range(nchunks):

            sb = slice(k*chunk_size , min((k+1)*chunk_size, nb))
            x_in = x[sb,...]

            if min_max is not None:
                min_val, max_val = min_max
                x_in = _rescale_data(x_in, float(min_val), float(max_val))
            
            if padding != 0:
                if k == nchunks - 1:
                    x_in = np.pad(x_in, \
                                  ((0,padding), (0,0), \
                                   (0,0), (0,0), (0,0)), mode = 'edge')
                
                x_out = self.models[model_key].predict(x_in)

                if k == nchunks -1:
                    x_out = x_out[:-padding,...]
            else:
                x_out = self.models[model_key].predict(x_in)
            out_arr[sb,...] = x_out
        
        out_arr = np.round(out_arr).astype(np.uint8)
        t_unit = (time.time() - t0)*1000.0/nb
        
        if TIMEIT:
            print("inf. time p. input patch size %s = %.2f ms, nb = %i"%(str(x[0,...,0].shape), t_unit, nb))
            print("\n")
            return out_arr, t_unit
        else:
            return out_arr
    
    
    
    def load_datasets(self, datasets, normalize_sampling_factor = 4, TIMEIT = False):
    
        '''
        Parameters  
        ----------  
        
        '''
        Xs, Ys = load_dataset_pairs(datasets, \
                               normalize_sampling_factor = normalize_sampling_factor, \
                               TIMEIT = TIMEIT)
        return Xs, Ys
    
            
            
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

        model_key = "segmenter"
        self.models.update({model_key : None})
        # input_size here is redundant if the network is fully convolutional
        self.models[model_key] = build_Unet_3D(**model_params)
        self.models[model_key].compile(optimizer=tf.keras.optimizers.Adam(),\
                                         loss= tf.keras.losses.BinaryCrossentropy())
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
