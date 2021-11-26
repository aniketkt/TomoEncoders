#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class implementations for real-time 3D feature extraction

"""
import pandas as pd
import os
import glob
import numpy as np



from tomo_encoders import *
from tensorflow import keras
from tomo_encoders import Patches

import tensorflow as tf
from tensorflow.keras.models import load_model
from multiprocessing import Pool, cpu_count
import functools

import h5py
import abc
import time

# from tensorflow import RunOptions
from tensorflow.keras.backend import random_normal
from tensorflow import map_fn, constant, reduce_max, reduce_min
from tensorflow.keras import layers as L

# tensorflow configs
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


def insert_activation(tensor_in, activation):
    """
    Returns
    -------
    tensor
        of rank 2 (FC layer), 4 (image) or 5 (volume) (batch_size, nz, ny, nx, n_channels)
    
    Parameters
    ----------
    tensor_in : tensor
            input tensor
    activation : str or tf.Keras.layers.Activation
            name of custom activation or Keras activation layer
            
    """
    if activation is None:
        return tensor_in
    if activation == 'lrelu':
        tensor_out = L.LeakyReLU(alpha = 0.2)(tensor_in)
    else:
        tensor_out = L.Activation(activation)(tensor_in)
    return tensor_out


def hidden_layer(tensor_in, n_hidden, activation = None, batch_norm = False):
    """
    Define a fully-connected layer with batch normalization, dropout and custom activations.
    
    
    Returns
    -------
    tensor
        of rank 2 (batch_size, n_hidden)
    
    Parameters
    ----------
    tensor_in  : tensor
            input tensor
    n_hidden  : int
            number of units in the dense layer (this is the output shape)
    activation : str or tf.Keras.layers.Activation
            name of custom activation or Keras activation layer
    batch_norm : bool
            True to insert a BN layer
            
    """
    
    tensor_out = L.Dense(n_hidden, activation = None)(tensor_in)
    
    if batch_norm:
        tensor_out = L.BatchNormalization(momentum = 0.9, epsilon = 1e-5)(tensor_out)
        
    tensor_out = insert_activation(tensor_out, activation)
    
    
    return tensor_out

def stdize_vol(vol):
    
    eps = constant(1e-12, dtype = 'float32')
    max_ = reduce_max(vol)
    min_ = reduce_min(vol)
    vol = (vol - min_ )  / (max_ - min_ + eps)
    return vol

def standardize(vols):
    return map_fn(stdize_vol, vols)

def custom_Conv3D(tensor_in, n_filters, kern_size, activation = None, batch_norm = False):
    
    """
    Define a custom 3D convolutional layer with batch normalization and custom activation function (includes lrelu)  

    This is the order chosen in our implementation:  
    
    -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->  
    
    See supmat in: https://dmitryulyanov.github.io/deep_image_prior

    Returns
    -------
    tensor
        of rank 5 (batch_size, nz, ny, nx, n_channels)
    
    Parameters
    ----------
    tensor_in  : tensor
            input tensor
    n_filters  : int
            number of filters in the first convolutional layer
    kern_size  : tuple
            kernel size, e.g. (3,3,3)
    activation : str or tf.Keras.layers.Activation
            name of custom activation or Keras activation layer
    batch_norm : bool
            True to insert a BN layer
            
    """
    
    tensor_out = L.Conv3D(n_filters, kern_size, activation = None, padding = "same")(tensor_in)
    
    if batch_norm:
        tensor_out = L.BatchNormalization(momentum = 0.9, epsilon = 1e-5)(tensor_out)
    
    tensor_out = insert_activation(tensor_out, activation)
    return tensor_out
    
##############
# Contributed by Audrey Bartlett (Berkeley)

def analysis_block_small(tensor_in, n_filters, pool_size, \
                   kern_size = None, \
                   activation = None, \
                   batch_norm = False):

    """
    Define a block of 2 3D convolutional layers followed by a 3D max-pooling layer


    Returns
    -------
    output tensor of rank 5 (batch_size, nz, ny, nx, n_channels)
    
    Parameters
    ----------
    tensor_in  : tensor
            input tensor
    n_filters  : int
            number of filters in the first convolutional layer
    pool_size  : tuple
            max pooling e.g. (2,2,2)
    kern_size  : tuple
            kernel size, e.g. (3,3,3)
    activation : str or tf.Keras.layers.Activation
            name of custom activation or Keras activation layer
    kern_init  : str
            kernel initialization method
    batch_norm : bool
            True to insert a BN layer
            
    """
    
    # layer # 1
    tensor_out = custom_Conv3D(tensor_in, n_filters, kern_size, \
                               activation = activation, \
                               batch_norm = batch_norm)
    
    # layer # 2; 2x filters
    tensor_out = custom_Conv3D(tensor_out, 2*n_filters, kern_size, \
                               activation = activation, \
                               batch_norm = batch_norm)

    # MaxPool3D
    return L.MaxPool3D(pool_size = pool_size, padding = "same")(tensor_out)



def synthesis_block_small(tensor_in, n_filters, pool_size, \
                    activation = None, \
                    kern_size = 3, \
                    kern_size_upconv = 2, \
                    batch_norm = False):
    """
    Define a 3D upsample block (with no concatenation/skip connections)
    
    Returns
    -------
    tensor
        of rank 5 (batch_size, nz, ny, nx, n_channels)

    Parameters  
    ----------  
    tensor_in     : tensor  
            input tensor  
    concat_tensor : tensor  
            this will be concatenated to the output of the upconvolutional layer  
    n_filters  : int  
            number of filters in each convolutional layer after the transpose conv.  
    pool_size  : tuple  
            reverse the max pooling e.g. (2,2,2) with these many strides for transpose conv.  
    kern_size  : int  
            kernel size for conv, e.g. 3  
    kern_size_upconv  : int  
            kernel size for upconv, e.g. 2  
    activation : str or tf.Keras.layers.Activation
            name of custom activation or Keras activation layer
    batch_norm : bool
            True to insert a BN layer
    concat_flag : bool
            True to concatenate layers (add skip connections)
    """
    
    # transpose convolution
    n_filters_upconv = tensor_in.shape[-1]
    tensor_out = L.Conv3DTranspose(n_filters_upconv, kern_size_upconv, padding = "same", activation = None, strides = pool_size) (tensor_in)
    tensor_out = insert_activation(tensor_out, activation)

    # layer # 1
    tensor_out = custom_Conv3D(tensor_out, n_filters, kern_size, \
                               activation = activation, \
                               batch_norm = batch_norm)
    
    # layer # 2
    tensor_out = custom_Conv3D(tensor_out, n_filters, kern_size, \
                               activation = activation, \
                               batch_norm = batch_norm)
    
    return tensor_out    


def build_encoder_r(input_shape, n_filters = [32, 64], \
                 n_blocks = 2, activation = 'lrelu',\
                 batch_norm = True, kern_size = 3, kern_size_upconv = 2,\
                 hidden_units = [128,32,2], pool_size = 2, POOL_FLAG = True):
    
    """
    @arshadzahangirchowdhury
    Define the encoder of a 3D convolutional autoencoder, based on the arguments provided. 
    
    
    Returns
    -------
    tf.Keras.model
        keras model(s) for the encoder of a 3D autoencoder-decoder architecture. 
    flatten_shape
    
    preflatten_shape
            
    Parameters
    ----------
    input_shape  : tuple
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
            
    hidden_units: list
            list of number of hidden layer units. last value is the code.  
            
    pool_size : int or list
            if list, list length must be equal to number of blocks.  
    
    
    """
      
    inp = L.Input(input_shape)
    
    if type(pool_size) is int:
        pool_size = [pool_size]*n_blocks
    elif len(pool_size) != n_blocks:
        raise ValueError("list length must be equal to number of blocks")
        
    # downsampling path. e.g. n_blocks = 3, n_filters = [16,32,64], input volume is 64^3
    for ii in range(n_blocks): #iterations
        
        if ii == 0:
            code = inp
            
        code = analysis_block_small(code, \
                              n_filters[ii], \
                              pool_size[ii], \
                              kern_size = kern_size, \
                              activation = activation, \
                              batch_norm = batch_norm)

    if POOL_FLAG:
        # pool a second time before flattening
        code = L.MaxPool3D(pool_size = 2, padding = "same")(code)

    for ic, n_hidden in enumerate(hidden_units):
        if ic == len(hidden_units) - 1:  # ic = 2 (last unit is the code)
            break
        elif ic == 0:
            # ic = 0 --> n_hidden = 128;
            # first hidden layer takes flattened vector as input
            preflatten_shape = tuple(code.shape[1:])
            code = L.Flatten()(code)
            flatten_shape = code.shape[-1]
            code = hidden_layer(code, n_hidden, \
                                activation = activation, \
                                batch_norm = batch_norm)
        else:
            # ic = 1 --> n_hidden = 32;
            code = hidden_layer(code, n_hidden, \
                                activation = activation, \
                                batch_norm = batch_norm)
        
    z = hidden_layer(code, hidden_units[-1], \
                     activation = activation, \
                     batch_norm = True)
    
    print('inp:',inp)
    encoder = keras.models.Model(inp, z, name = "encoder")
    print('encoder:',encoder)
    
    return encoder, flatten_shape, preflatten_shape


def build_decoder_r(flatten_shape, preflatten_shape, n_filters = [32, 64], \
                 n_blocks = 2, activation = 'lrelu',\
                 batch_norm = True, kern_size = 3, kern_size_upconv = 2,\
                 hidden_units = [128,32,2], pool_size = 2, POOL_FLAG = True):
    
    """
    @arshadzahangirchowdhury
    Define the decoder of a 3D convolutional autoencoder, based on the arguments provided. 
    2 layers and no skip connections
    
    Version of _2 with no skip connections
    
    NOTE: borrowed from build_CAE_3D_4()
    
    to-do: Integrate build_CAE_3D_3 from change_encoders and build_CAE_3D_r
    
    Returns
    -------
    tf.Keras.model
        keras model(s) for the encoder of a 3D autoencoder-decoder architecture.  
        
    Parameters
    ----------
    flatten_shape  : tuple
            input volume shape (nz,ny,nx,1)  
    
    preflatten_shape  : tuple
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
            
    hidden_units: list
            list of number of hidden layer units. last value is the code.  
            
    pool_size : int or list
            if list, list length must be equal to number of blocks.  
    
    
    """
    decoder_input=L.Input((hidden_units[-1],), name = "decoder_input")
    for ic, n_hidden in enumerate(hidden_units[::-1]): # iterate as e.g. [16,32,128]
        if ic == 0:
            # skip n_hidden = 16 as we already implemented that in the previous loop
            decoded = decoder_input
        else:
            # ic = 1 --> n_hidden = 32
            # ic = 2 --> n_hidden = 128
            decoded = hidden_layer(decoded, n_hidden, activation = activation, batch_norm = batch_norm)

    # n_hidden = flattened shape
    decoded = hidden_layer(decoded, flatten_shape, activation = activation, batch_norm = batch_norm)

    # reshape to convolutional feature maps
    decoded = L.Reshape(preflatten_shape)(decoded)

    if POOL_FLAG:
        # upsample once before synthesis block
        n_filters_upconv = decoded.shape[-1]
        decoded = L.Conv3DTranspose(n_filters_upconv, \
                                    kern_size_upconv, \
                                    padding = "same", \
                                    activation = None, \
                                    strides = 2) (decoded)
        decoded = insert_activation(decoded, activation)
    
    # upsampling path. e.g. n_blocks = 3
    for ii in range(n_blocks-1, -1, -1):
        
        decoded = synthesis_block_small(decoded, \
                                  n_filters[ii], \
                                  pool_size[ii], \
                                  activation = activation, \
                                  kern_size = kern_size, \
                                  kern_size_upconv = kern_size_upconv, \
                                  batch_norm = batch_norm)
        
    decoded = L.Conv3D(1, (1,1,1), activation = 'sigmoid', padding = "same")(decoded)

    decoder = keras.models.Model(decoder_input, decoded, name = "decoder")
    decoder.summary()
    return decoder

class RegularizedAutoencoder(keras.Model):
    
    """
    Modifies the keras.Model to implement custom loss functions and train step
    
    Parameters
    ----------
    encoder : tf.keras.Model
        the encoder model.
    
    decoder : tf.keras.Model
        the decoder model.
    
    weight: float
        strength of the regularization loss (L1 or KL).
    
    regularization_type: str 
        Type of regularization of model loss.'kl': Kullback-Leibler divergence loss. 'L1': L1 loss.
    
    
    """

    def __init__(self, encoder, decoder, weight=1/250.0,regularization_type='kl', **kwargs):
        super(RegularizedAutoencoder, self).__init__(**kwargs)
        
        if len(encoder.output_shape[1:]) !=1:
            print('WARNING: Encoder output is not a vector.')
            
        assert encoder.input_shape == decoder.output_shape, 'Encoder input shape and decoder output shape must match.'
        
        self.encoder = encoder
        self.decoder = decoder
        self.weight=float(weight)
        self.regularization_type=regularization_type
        
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.pixel_mse_loss_tracker = keras.metrics.Mean(
            name="pixel_mse_loss"
        )
        self.regularization_loss_tracker = keras.metrics.Mean(name="regularization_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.pixel_mse_loss_tracker,
            self.regularization_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            
            z = self.encoder(data)
            decoded = self.decoder(z)
            pixel_mse_loss = tf.reduce_mean(keras.losses.mean_squared_error(data, decoded))
            #to-do: Try lambda function or tensorflow map
            
            if self.regularization_type=='L1':
                regularization_loss=tf.reduce_mean(tf.abs(z))
                
            elif self.regularization_type=='kl':
                regularization_loss = tf.reduce_mean(keras.losses.kl_divergence(data, decoded))
            else:
                raise ValueError("Regularization loss must be either 'L1' or 'kl' " )

            total_loss = pixel_mse_loss + self.weight*regularization_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.pixel_mse_loss_tracker.update_state(pixel_mse_loss)
        self.regularization_loss_tracker.update_state(regularization_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "pixel_mse_loss": self.pixel_mse_loss_tracker.result(),
            "regularization_loss": self.regularization_loss_tracker.result()
        }

from tomo_encoders.neural_nets.keras_processor import EmbeddingLearner
class SelfSupervisedCAE(EmbeddingLearner):
    def __init__(self, **kwargs):
        
        '''
        models : dict of tf.keras.Models.model 
            dict contains some models with model keys as string descriptors of them.

        '''
        self.model_keys = ["encoder", "decoder", "autoencoder"]
        self.output_type = "embeddings"
        super().__init__(**kwargs)
        
        return
            
    def save_models(self, model_path):
        
        for model_key in self.models.keys():
            if model_key == "autoencoder":
                continue
            filepath = os.path.join(model_path, "%s_%s.hdf5"%(model_key, self.model_tag))
            self.models[model_key].save(filepath, include_optimizer = False)
        return
    
    def _build_models(self, model_size = (64,64,64), descriptor_tag = "misc", **model_params):
        '''
        
        Parameters
        ----------
        model_keys : list  
            list of strings describing the model, e.g., ["encoder", "decoder"], etc.
        model_params : dict
            for passing any number of model hyperparameters necessary to define the model(s).
            
        '''
        if model_params is None:
            raise ValueError("Need model hyperparameters or instance of model. Neither were provided")
        else:
            self.models = {}

        # insert your model building code here. The models variable must be a dictionary of models with str descriptors as keys
        self.model_size = model_size
        self.model_tag = "%s"%descriptor_tag

        for key in self.model_keys:
            self.models.update({key : None})
        self.models["encoder"], _flatten_shape, _preflatten_shape = build_encoder_r(self.model_size + (1,), **model_params)
        self.models["decoder"] = build_decoder_r(_flatten_shape, _preflatten_shape, **model_params)
        self.models["autoencoder"] = None
      
        return
    
    def _load_models(self, model_tag = None, model_size = (64,64,64), model_path = 'some-path'):
        
        '''
        Parameters
        ----------
        model_names : dict
            example {"model_key" : tf.keras.Model, ...}
        model_path : str  
            example "some/path"
            
        '''
        assert model_tag is not None, "need model_tag"
        
        self.models = {} # clears any existing models linked to this class!!!!
        for model_key in self.model_keys:
            if model_key == "autoencoder":
                self.models.update({model_key : None})
            else:
                filepath = os.path.join(model_path, "%s_%s.hdf5"%(model_key, model_tag))
                self.models.update({model_key : load_model(filepath)})
        # insert assignment of model_size here
        self.model_size = self.models["encoder"].input_shape[1:-1]
        self.model_tag = model_tag
        return
    
    def train(self, vols, batch_size = 10, \
              sampling_method = 'random-fixed-width', \
              n_epochs = 10, \
              random_rotate = True, \
              add_noise = 0.1,\
              max_stride = 1, \
              normalize_sampling_factor = 2):
        
        '''
        
        '''
        
        # to-do: IMPORTANT! Go make data_loader.py, make sure normalize volume is done there.
        # instantiate data generator for use in training.  
        dg = self.data_generator(vols, batch_size, sampling_method, \
                                 max_stride = max_stride, \
                                 random_rotate = random_rotate, \
                                 add_noise = add_noise)
        
        tot_steps = 500
        val_split = 0.2
        steps_per_epoch = int((1-val_split)*tot_steps//batch_size)
        validation_steps = int(val_split*tot_steps//batch_size)

        t0 = time.time()
        
        self.models["autoencoder"] = RegularizedAutoencoder(self.models['encoder'],\
                                                            self.models['decoder'],\
                                                            weight=1/250.0,\
                                                            regularization_type='kl')
        
        
        self.models["autoencoder"].compile(optimizer='adam')
        self.models["autoencoder"].fit(x = dg, epochs = n_epochs , \
                                       steps_per_epoch=steps_per_epoch, \
                                       validation_steps=validation_steps, verbose = 1)    

        self.models["encoder"] = self.models["autoencoder"].encoder
        self.models["decoder"] = self.models["autoencoder"].decoder
        
        t1 = time.time()
        training_time = (t1 - t0)
        print("training time = %.2f seconds"%training_time)        
        
        return
        
#     def detect_changes(self, vol_prev, vol_curr, patches):
        
#         '''
#         '''
#         t0 = time.time()
#         sub_vols_prev = patches.extract(self._normalize_volume(vol_prev), self.model_size)
#         sub_vols_curr = patches.extract(self._normalize_volume(vol_curr), self.model_size)

#         h_prev = self.models["encoder"].predict(sub_vols_prev[...,np.newaxis])
#         h_curr = self.models["encoder"].predict(sub_vols_curr[...,np.newaxis])
#         h_delta = (h_curr - h_prev)**2
#         h_delta = np.mean(h_delta, axis = 1)
#         h_delta = np.sqrt(h_delta)
#         patches.add_features(h_delta.reshape(-1,1), names = ["h_delta"])
#         t1 = time.time()
#         tot_time_fe = t1 - t0
#         print("total time for change detector = %.2f seconds"%tot_time_fe)
#         mse = np.mean(np.power(sub_vols_curr - sub_vols_prev, 2), axis = (1,2,3))
#         patches.add_features(mse.reshape(-1,1), names = ["mse"])
        
#         return patches
    
    
    
    
    
if __name__ == "__main__":
    
    print('just a bunch of functions')
    
