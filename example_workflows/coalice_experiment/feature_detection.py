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

from tomo_encoders.neural_nets.porosity_encoders import build_CAE_3D, custom_objects_dict
from tomo_encoders.patches import Patches
import tensorflow as tf
from tensorflow.keras.models import load_model

from multiprocessing import Pool, cpu_count
import functools

import h5py
import abc
import time


# complete once you write one or two feature extractors
# class AnyFeatureExtractor(metaclass = abc.ABCMeta):
    
#     @abc.abstractmethod
#     def do_something(self):
#         pass
#     @abc.abstractmethod
#     def do_something_else(self):
#         pass


class SelfSupervisedCAE():
    def __init__(self, vol_shape, \
                 model_initialization = "define-new", \
                 model_size = (64,64,64), \
                 descriptor_tag = "misc", **model_params):
        '''
        
        Parameters
        ----------
        vol_shape : tuple
            Shape of the object volume  
        
        model_initialization : str
            either "define-new" or "load-model"
        
        model_size : tuple
            shape of the input patch required by model

        descriptor_tag : str
            some description used while saving the models

        model_params : dict
            If "define-new", this contains the hyperparameters that define the architecture of the neural network model. If "load-model", it contains the paths to the models.
            
        models : dict of tf.keras.Models.model 
            dict contains {"latent_embedder" : encoder_model, "CAE" : denoiser_model}. Here, both models are keras 3D models with input shape = model_size

        '''
        model_getters = {"load-model" : self._load_models, \
                         "define-new" : self._build_models, \
                         "load-weights" : self._load_weights}
        
        self.vol_shape = vol_shape
        # any function chosen must assign self.models, self.model_tag and self.model_size
        
        if model_initialization == "define-new":
            model_getters[model_initialization](model_size = model_size, \
                                                descriptor_tag = descriptor_tag, \
                                                **model_params)
        elif model_initialization == "load-model":
            model_getters[model_initialization](**model_params)
        else:
            raise NotImplementedError("method is not implemented")
            
        return
            
    def save_models(self, model_path):
        
        for model_key in self.models.keys():
            self.models[model_key].save(os.path.join(model_path, "%s_%s.hdf5"%(model_key, self.model_tag)))
        return
    
    def _build_models(self, model_size = (64,64,64), descriptor_tag = "misc", **model_params):
        '''
        
        Parameters
        ----------
        model_keys : list  
            list of strings describing the model, e.g., ["CAE", "embedder"], etc.
        model_params : dict
            for passing any number of model hyperparameters necessary to define the model(s).
            
        '''
        if model_params is None:
            raise ValueError("Need model hyperparameters or instance of model. Neither were provided")
        else:
            self.models = {}

        # insert your model building code here. The models variable must be a dictionary of models with str descriptors as keys
        self.model_size = model_size
        isconcat = model_params["isconcat"]
        latent_dim = model_params["hidden_units"][-1]
        self.model_tag = "%i%i%id%s_%s"%(isconcat[0], \
                                    isconcat[1], \
                                    isconcat[2], \
                                    latent_dim, \
                                    descriptor_tag)

        model_keys = ["latent_embedder", "CAE"]
        for key in model_keys:
            self.models.update({key : None})
        self.models["latent_embedder"], self.models["CAE"] = build_CAE_3D(self.model_size + (1,), **model_params)
        self.models["CAE"].compile(optimizer=tf.keras.optimizers.Adam(),\
                      loss=tf.keras.losses.MeanSquaredError())
        self.models["latent_embedder"].compile()        
        return
    
    def _load_weights(self, **kwargs):
        raise NotImplementedError("Not implemented yet")
        
    def _load_models(self, model_names = None, model_path = 'some/path'):
        
        '''
        Parameters
        ----------
        model_names : dict
            example {"CAE" : "CAE_name", "embedder" : "embedder_name"}
        model_path : str  
            example "some/path"
        custom_objects_dict : dict  
            dictionary of custom objects (usually pickled with the keras models)
            
        '''
        self.models = {} # clears any existing models linked to this class!!!!
        for model_key, model_name in model_names.items():
            self.models.update({model_key : load_model(os.path.join(model_path, model_name + '.hdf5'), \
                                                      custom_objects = custom_objects_dict)})
        # insert assignment of model_size here
        self.model_size = self.models["CAE"].input_shape[1:-1]
        self.model_tag = "_".join(model_names["CAE"].split("_")[1:])
        return

    def _normalize_volume(self, vol):
        '''
        Normalizes volume to values into range [0,1]  

        '''
        if vol.shape != self.vol_shape:
            raise ValueError("vol shape does not match")
        else:
            eps = 1e-12
            max_val = np.max(vol)
            min_val = np.min(vol)
            vol = (vol - min_val) / (max_val - min_val + eps)
            return vol
        
    def data_generator(self, vol, batch_size, sampling_method, max_stride = 1, random_rotate = False, add_noise = 0.1):

        
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
            
            if sampling_method in ["grid", "random-fixed-width"]:
                patches = Patches(vol.shape, initialize_by = sampling_method, \
                                  patch_size = self.model_size, \
                                  stride = max_stride, \
                                  n_points = batch_size)    
                
            elif sampling_method in ["random"]:
                patches = Patches(vol.shape, initialize_by = sampling_method, \
                                  min_patch_size = self.model_size, \
                                  max_stride = max_stride, \
                                  n_points = batch_size)    

            y = patches.extract(vol, self.model_size)[...,np.newaxis]
            x = y.copy() + np.random.normal(0, add_noise, y.shape)    

            if random_rotate:
                nrots = np.random.randint(0, 4, batch_size)
                for ii in range(batch_size):
                    axes = tuple(np.random.choice([0, 1, 2], size=2, replace=False))
                    x[ii, ..., 0] = np.rot90(x[ii, ..., 0], k=nrots[ii], axes=axes)
                    y[ii, ..., 0] = np.rot90(y[ii, ..., 0], k=nrots[ii], axes=axes)

            yield x, y           
        
    def train(self, vol, batch_size, sampling_method, n_epochs,\
                                 random_rotate = False,\
                                 add_noise = 0.1,\
                                 max_stride = 1):
        
        '''
        
        '''
        # normalize volume, check if shape is compatible.  
        vol = self._normalize_volume(vol)
        
        # instantiate data generator for use in training.  
        dg = self.data_generator(vol, batch_size, sampling_method, \
                                 max_stride = max_stride, \
                                 random_rotate = random_rotate, \
                                 add_noise = add_noise)
        
        
        tot_steps = 500
        val_split = 0.2
        steps_per_epoch = int((1-val_split)*tot_steps//batch_size)
        validation_steps = int(val_split*tot_steps//batch_size)

        t0 = time.time()
        self.models["CAE"].fit(x = dg, epochs = n_epochs,\
                  steps_per_epoch=steps_per_epoch,\
                  validation_steps=validation_steps, verbose = 1)    
        t1 = time.time()
        training_time = (t1 - t0)
        print("training time = %.2f seconds"%training_time)        
        
        return
        
    def detect_changes(self, vol_prev, vol_curr, patches):
        
        '''
        '''
        t0 = time.time()
        sub_vols_prev = patches.extract(self._normalize_volume(vol_prev), self.model_size)
        sub_vols_curr = patches.extract(self._normalize_volume(vol_curr), self.model_size)

        h_prev = self.models["latent_embedder"].predict(sub_vols_prev[...,np.newaxis])
        h_curr = self.models["latent_embedder"].predict(sub_vols_curr[...,np.newaxis])
        h_delta = (h_curr - h_prev)**2
        h_delta = np.mean(h_delta, axis = 1)
        h_delta = np.sqrt(h_delta)
        patches.add_features(h_delta.reshape(-1,1), names = ["h_delta"])
        t1 = time.time()
        tot_time_fe = t1 - t0
        print("total time for change detector = %.2f seconds"%tot_time_fe)
        mse = np.mean(np.power(sub_vols_curr - sub_vols_prev, 2), axis = (1,2,3))
        patches.add_features(mse.reshape(-1,1), names = ["mse"])
        
        return patches


if __name__ == "__main__":
    
    print('just a bunch of functions')
    
