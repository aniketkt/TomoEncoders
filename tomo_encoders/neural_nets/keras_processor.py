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
import tensorflow as tf
from tensorflow.keras.models import load_model
from multiprocessing import Pool, cpu_count
import functools
import cupy as cp
import h5py
import abc
import time
from tomo_encoders.misc.voxel_processing import _rescale_data, _find_min_max, modified_autocontrast, normalize_volume_gpu, _edge_map

from tomo_encoders.neural_nets.Unet3D import build_Unet_3D



class GenericKerasProcessor():
    
    def __init__(self,\
                 model_initialization = "define-new", \
                 descriptor_tag = "misc", **kwargs):
        
        '''
        
        Parameters
        ----------
        model_initialization : str
            either "define-new" or "load-model"
        
        descriptor_tag : str
            some description used while saving the models

        model_params : dict
            If "define-new", this contains the hyperparameters that define the architecture of the neural network model. If "load-model", it contains the paths to the models.
            
        models : dict of tf.keras.Models.model 
            example dict contains {"segmenter" : segmenter}

        '''

        # could be "data" or "labels" or "embeddings"
        self.input_type = "data"
        self.output_type = "data"
        
        model_getters = {"load-model" : self._load_models, \
                         "define-new" : self._build_models}

        # any function chosen must assign self.models, self.model_tag
        
        if model_initialization == "define-new":
            model_getters[model_initialization](descriptor_tag = descriptor_tag, \
                                                **kwargs)
        elif model_initialization == "load-model":
            model_getters[model_initialization](**kwargs)
        else:
            raise NotImplementedError("method is not implemented")
            
        return

    def print_layers(self, modelkey):
        
        txt_out = []
        for ii in range(len(self.models[modelkey].layers)):
            lshape = str(self.models[modelkey].layers[ii].output_shape)
            lname = str(self.models[modelkey].layers[ii].name)
            txt_out.append(lshape + "    ::    "  + lname)
        print('\n'.join(txt_out))
        return
    
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
                
                if model_key == "autoencoder":
                    z = self.models["encoder"].predict(x_in)
                    x_out = self.models["decoder"].predict(z)
                else:
                    x_out = self.models[model_key].predict(x_in)

                if k == nchunks -1:
                    x_out = x_out[:-padding,...]
            else:
                if self.output_type == "autoencoder":
                    z = self.models["encoder"].predict(x_in)
                    x_out = self.models["decoder"].predict(z)
                else:
                    x_out = self.models[model_key].predict(x_in)
            out_arr[sb,...] = x_out
        
        if self.output_type == "labels":
            out_arr = np.round(out_arr).astype(np.uint8)
        elif self.output_type == "embeddings":
            out_arr = np.round(out_arr).astype(x.dtype)
            
        t_unit = (time.time() - t0)*1000.0/nb
        
        if TIMEIT:
            print("inf. time p. input patch size %s = %.2f ms, nb = %i"%(str(x[0,...,0].shape), t_unit, nb))
            print("\n")
            return out_arr, t_unit
        else:
            return out_arr
    
    def calc_voxel_min_max(self, vol, sampling_factor, TIMEIT = False):

        '''
        returns min and max values for a big volume sampled at some factor
        '''

        return _find_min_max(vol, sampling_factor, TIMEIT = TIMEIT)
    
    
    def rescale_data(self, data, min_val, max_val):
        '''
        Recales data to values into range [min_val, max_val]. Data can be any numpy or cupy array of any shape.  

        '''
        xp = cp.get_array_module(data)  # 'xp' is a standard usage in the community
        eps = 1e-12
        data = (data - min_val) / (max_val - min_val + eps)
        return data
    
    
    @abstractmethod
    def test_speeds(self):
        pass
    
    @abstractmethod
    def save_models(self):
        pass
    
    @abstractmethod
    def _build_models(self):
        pass
        
    @abstractmethod
    def _load_models(self):
        pass

    def _msg_exec_time(self, func, t_exec):
        print("TIME: %s: %.2f seconds"%(func.__name__, t_exec))
        return
        
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def get_patches(self):
        pass
    
    @abstractmethod
    def data_generator(self):
        pass

class EmbeddingLearner(GenericKerasProcessor):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        return
    
    def data_generator(self, Xs, batch_size, sampling_method, \
                       max_stride = 1, \
                       random_rotate = False, add_noise = 0.1):
        
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
            
            x = []
            for ivol in range(n_vols):
                patches = self.get_patches(Xs[ivol].shape, sampling_method, np.sum(idx_vols == ivol), max_stride = max_stride)
                x.append(self.extract_training_sub_volumes(Xs[ivol], patches, add_noise, random_rotate))
            
            yield np.concatenate(x, axis = 0, dtype = 'float32')
    
    def get_patches(self, vol_shape, sampling_method, batch_size, max_stride = None):

        if sampling_method in ["grid", 'regular-grid', "random-fixed-width"]:
            patches = Patches(vol_shape, initialize_by = sampling_method, \
                              patch_size = self.model_size, \
                              n_points = batch_size)    

        elif sampling_method in ["random"]:
            patches = Patches(vol_shape, initialize_by = sampling_method, \
                              min_patch_size = self.model_size, \
                              max_stride = max_stride, \
                              n_points = batch_size)    
        else:
            raise ValueError("sampling method not supported")

        return patches    

    def extract_training_sub_volumes(self, X, patches, add_noise, random_rotate):
        '''
        Extract training pairs x and y from a given volume X, Y pair
        '''
        
        batch_size = len(patches)
        x = patches.extract(X, self.model_size)[...,np.newaxis]

        if random_rotate:
            nrots = np.random.randint(0, 4, batch_size)
            for ii in range(batch_size):
                axes = tuple(np.random.choice([0, 1, 2], size=2, replace=False))
                x[ii, ..., 0] = np.rot90(x[ii, ..., 0], k=nrots[ii], axes=axes)
        
        return x    
    
    def random_data_generator(self, batch_size):

        while True:

            x_shape = tuple([batch_size] + list(self.input_size) + [1])
            x = np.random.uniform(0, 1, x_shape)#.astype(np.float32)
            x[x == 0] = 1.0e-12
            yield x
    
    def predict_embeddings(self, x, chunk_size, min_max = None, TIMEIT = False):

        '''
        Predicts on sub_vols. This is a wrapper around keras.model.predict() that speeds up inference on inputs lengths that are not factors of 2. Use this function to do multiprocessing if necessary.  
        
        '''
        assert x.ndim == 5, "x must be 5-dimensional (batch_size, nz, ny, nx, 1)."
        
        t0 = time.time()
        print("call to keras predict, len(x) = %i, shape = %s, chunk_size = %i"%(len(x), str(x.shape[1:-1]), chunk_size))
        nb = len(x)
        nchunks = int(np.ceil(nb/chunk_size))
        nb_padded = nchunks*chunk_size
        padding = nb_padded - nb

        out_arr = np.zeros((nb, self.models["encoder"].output_shape[-1]), dtype = np.float32) # use numpy since return from predict is numpy

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
                x_out = self.models["encoder"].predict(x_in)

                if k == nchunks -1:
                    x_out = x_out[:-padding,...]
            else:
                x_out = self.models["encoder"].predict(x_in)
                
            out_arr[sb,...] = x_out
        
        if self.output_type == "embeddings":
            print("shape of output array: ", out_arr.shape)
        t_unit = (time.time() - t0)*1000.0/nb
        
        if TIMEIT:
            print("inf. time p. input patch size %s = %.2f ms, nb = %i"%(str(x[0,...,0].shape), t_unit, nb))
            print("\n")
            return out_arr, t_unit
        else:
            return out_arr
    
class Vox2VoxProcessor_fCNN(GenericKerasProcessor):
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
    def test_speeds(self, chunk_size, n_reps = 3, input_size = None, model_key = "segmenter"):
        
        if input_size is None:
            input_size = (64,64,64)
        for jj in range(n_reps):
            x = np.random.uniform(0, 1, tuple([chunk_size] + list(input_size) + [1])).astype(np.float32)
            y_pred = self.predict_patches(model_key, x, chunk_size, None, min_max = (-1,1), TIMEIT = True)
        return

    def extract_training_patch_pairs(self, X, Y, patches, add_noise, random_rotate, input_size):
        '''
        Extract training pairs x and y from a given volume X, Y pair
        '''
        
        batch_size = len(patches)
        y = patches.extract(Y, input_size)[...,np.newaxis]            
        x = patches.extract(X, input_size)[...,np.newaxis]
        std_batch = np.random.uniform(0, add_noise, batch_size)
        x = x + np.asarray([np.random.normal(0, std_batch[ii], x.shape[1:]) for ii in range(batch_size)])

        if random_rotate:
            nrots = np.random.randint(0, 4, batch_size)
            for ii in range(batch_size):
                axes = tuple(np.random.choice([0, 1, 2], size=2, replace=False))
                x[ii, ..., 0] = np.rot90(x[ii, ..., 0], k=nrots[ii], axes=axes)
                y[ii, ..., 0] = np.rot90(y[ii, ..., 0], k=nrots[ii], axes=axes)
#         print("DEBUG: shape x %s, shape y %s"%(str(x.shape), str(y.shape)))
        
        return x, y
    
    def random_data_generator(self, batch_size, input_size = (64,64,64)):

        while True:

            x_shape = tuple([batch_size] + list(input_size) + [1])
            x = np.random.uniform(0, 1, x_shape)#.astype(np.float32)
            y = np.random.uniform(0, 1, x_shape)#.astype(np.float32)
            x[x == 0] = 1.0e-12
            y[y == 0] = 1.0e-12
            yield x, y

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

        if self.output_type == "data":
            model_key = "enhancer"
        elif self.output_type == "labels":
            model_key = "segmenter"
        self.models.update({model_key : None})
        # input_size here is redundant if the network is fully convolutional
        self.models[model_key] = build_Unet_3D(**model_params)
        self.models[model_key].compile(optimizer=tf.keras.optimizers.Adam(),\
                                         loss= tf.keras.losses.MeanSquaredError())
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
        if self.output_type == "data":
            model_key = "enhancer"
        elif self.output_type == "labels":
            model_key = "segmenter"
        
        self.model_tag = "_".join(model_names[model_key].split("_")[1:])
        return
    
    def save_models(self, model_path):
        
        if self.output_type == "data":
            model_key = "enhancer"
        elif self.output_type == "labels":
            model_key = "segmenter"
        
        model = self.models[model_key]
        filepath = os.path.join(model_path, "%s_%s.hdf5"%(model_key, self.model_tag))
        tf.keras.models.save_model(model, filepath, include_optimizer=False)        
        return
            
            
            
            
            
if __name__ == "__main__":
    
    print('just a bunch of functions')
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
