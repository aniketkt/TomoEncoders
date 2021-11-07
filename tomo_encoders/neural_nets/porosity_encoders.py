#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Easily define U-net-like architectures using Keras layers

"""

import numpy as np
# from tensorflow import RunOptions
from tensorflow import keras
from tensorflow.keras.backend import random_normal
import tensorflow as tf
from tensorflow import map_fn, constant, reduce_max, reduce_min
from tensorflow.keras import layers as L

# tensorflow configs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


####################### VAE #############################

# Some code is borrowed from:
# https://keras.io/examples/generative/vae/
    
#########################################################

# from models3D import insert_activation, hidden_layer
# from ct_segnet.model_utils.losses import focal_loss, my_binary_crossentropy, weighted_crossentropy, IoU, acc_zeros, acc_ones


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
    
    
def analysis_block(tensor_in, n_filters, pool_size, \
                   kern_size = None, \
                   activation = None, \
                   batch_norm = False):

    """
    Define a block of 2 3D convolutional layers followed by a 3D max-pooling layer


    Returns
    -------
    tuple of two tensors (output, tensor to concatenate in synthesis path)
        of rank 5 (batch_size, nz, ny, nx, n_channels)
    
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
    return L.MaxPool3D(pool_size = pool_size, padding = "same")(tensor_out), tensor_out


def synthesis_block(tensor_in, n_filters, pool_size, \
                    concat_tensor = None, \
                    activation = None, \
                    kern_size = 3, \
                    kern_size_upconv = 2, \
                    batch_norm = False, \
                    concat_flag = True):
    """
    Define a 3D upsample block and concatenate the output of downsample block to it (skip connection)
    
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

    if concat_flag:
        tensor_out = L.concatenate([tensor_out, concat_tensor])
    
    
    # layer # 1
    tensor_out = custom_Conv3D(tensor_out, n_filters, kern_size, \
                               activation = activation, \
                               batch_norm = batch_norm)
    
    # layer # 2
    tensor_out = custom_Conv3D(tensor_out, n_filters, kern_size, \
                               activation = activation, \
                               batch_norm = batch_norm)
    
    return tensor_out

    
def build_CAE_3D(vol_shape, n_filters = [16,32,64], \
                 n_blocks = 3, activation = 'lrelu',\
                 batch_norm = True, kern_size = 3, kern_size_upconv = 2,\
                 stdinput = False, hidden_units = [128,32,2],\
                 isconcat = None, pool_size = 2):
    """
    Define a 3D convolutional autoencoder, based on the arguments provided. Output image size is the same as input image size.  
    
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
            
    stdinput     : bool
            If True, the input image will be normalized into [0,1]  
            
    isconcat : bool or list
            Selectively concatenate layers (skip connections)  
    
    hidden_units: list
            list of number of hidden layer units. last value is the code.  
            
    pool_size : int or list
            if list, list length must be equal to number of blocks.  
            
    """

    inp = L.Input(vol_shape)
    
    if stdinput:
        standardizer = L.Lambda(standardize)
        stdinp = standardizer(inp)
    else:
        stdinp = inp
    
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
            code = stdinp
            
        code, concat_tensor = analysis_block(code, \
                                             n_filters[ii], \
                                             pool_size[ii], \
                                             kern_size = kern_size, \
                                             activation = activation, \
                                             batch_norm = batch_norm)
        concats.append(concat_tensor)

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
    encoder = keras.models.Model(inp, z, name = "encoder")
    
    for ic, n_hidden in enumerate(hidden_units[::-1]): # iterate as e.g. [16,32,128]
        if ic == 0:
            # skip n_hidden = 16 as we already implemented that in the previous loop
            decoded = z
        else:
            # ic = 1 --> n_hidden = 32
            # ic = 2 --> n_hidden = 128
            decoded = hidden_layer(decoded, n_hidden, activation = activation, batch_norm = batch_norm)

    # n_hidden = flattened shape
    decoded = hidden_layer(decoded, flatten_shape, activation = activation, batch_norm = batch_norm)

    # reshape to convolutional feature maps
    decoded = L.Reshape(preflatten_shape)(decoded)
    
    # upsampling path. e.g. n_blocks = 3
    for ii in range(n_blocks-1, -1, -1):
        # ii iterates as 2, 1, 0
    #         print("############# ii = %i"%ii)

    # this piece of code was accidentally left in from a previous version of the function and led to unknowingly removing the first skip connection.    
    #         if ii == 0:
    #             concat_flag = False
    #         else:
    #             concat_flag = isconcat[ii]
        
        decoded = synthesis_block(decoded, \
                                  n_filters[ii], \
                                  pool_size[ii], \
                                  concat_tensor = concats[ii], \
                                  activation = activation, \
                                  kern_size = kern_size, \
                                  kern_size_upconv = kern_size_upconv, \
                                  batch_norm = batch_norm, \
                                  concat_flag = isconcat[ii])
        
    decoded = L.Conv3D(1, (1,1,1), activation = 'sigmoid', padding = "same")(decoded)
    
    segmenter = keras.models.Model(inp, decoded, name = "segmenter")
    
    return encoder, segmenter

def build_CAE_3D_2(vol_shape, n_filters = [32, 64], \
                 n_blocks = 2, activation = 'lrelu',\
                 batch_norm = True, kern_size = 3, kern_size_upconv = 2,\
                 stdinput = False, hidden_units = [128,32,2],\
                 isconcat = None, pool_size = 2):
    
    
    """
    Two layers with two max pool steps.
    
    Define a 3D convolutional autoencoder, based on the arguments provided. Output image size is the same as input image size. 
   
    
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
            
    stdinput     : bool
            If True, the input image will be normalized into [0,1]  
            
    isconcat : bool or list
            Selectively concatenate layers (skip connections)  
    
    hidden_units: list
            list of number of hidden layer units. last value is the code.  
            
    pool_size : int or list
            if list, list length must be equal to number of blocks.  
            
    """
      
    
    inp = L.Input(vol_shape)
    
    if stdinput:
        standardizer = L.Lambda(standardize)
        stdinp = standardizer(inp)
    else:
        stdinp = inp
    
    if isconcat is None:
        isconcat = [False]*n_blocks
    
    if type(pool_size) is int:
        pool_size = [pool_size]*n_blocks
    elif len(pool_size) != n_blocks:
        raise ValueError("list length must be equal to number of blocks")
        
    concats = []
    # downsampling path. e.g. n_blocks = 3, n_filters = [16,32,64], input volume is 64^3
    for ii in range(n_blocks): #iterations
        
        if ii == 0:
            code = stdinp
            
        code, concat_tensor = analysis_block(code, \
                                             n_filters[ii], \
                                             pool_size[ii], \
                                             kern_size = kern_size, \
                                             activation = activation, \
                                             batch_norm = batch_norm)
        
        concats.append(concat_tensor)
        
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
    
    encoder = keras.models.Model(inp, z, name = "encoder")
    
    for ic, n_hidden in enumerate(hidden_units[::-1]): # iterate as e.g. [16,32,128]
        if ic == 0:
            # skip n_hidden = 16 as we already implemented that in the previous loop
            decoded = z
        else:
            # ic = 1 --> n_hidden = 32
            # ic = 2 --> n_hidden = 128
            decoded = hidden_layer(decoded, n_hidden, activation = activation, batch_norm = batch_norm)

    # n_hidden = flattened shape
    decoded = hidden_layer(decoded, flatten_shape, activation = activation, batch_norm = batch_norm)

    # reshape to convolutional feature maps
    decoded = L.Reshape(preflatten_shape)(decoded)

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
        
        decoded = synthesis_block(decoded, \
                                  n_filters[ii], \
                                  pool_size[ii], \
                                  concat_tensor = concats[ii], \
                                  activation = activation, \
                                  kern_size = kern_size, \
                                  kern_size_upconv = kern_size_upconv, \
                                  batch_norm = batch_norm, \
                                  concat_flag = isconcat[ii])
        
    decoded = L.Conv3D(1, (1,1,1), activation = 'sigmoid', padding = "same")(decoded)
    
    segmenter = keras.models.Model(inp, decoded, name = "segmenter")
    
    return encoder, segmenter
    


def build_Unet_3D(vol_shape, n_filters = [16,32,64], \
                 n_blocks = 3, activation = 'lrelu',\
                 batch_norm = True, kern_size = 3, kern_size_upconv = 2,\
                 stdinput = False, isconcat = None, pool_size = 2):
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
            
    stdinput     : bool
            If True, the input image will be normalized into [0,1]  
            
    isconcat : bool or list
            Selectively concatenate layers (skip connections)  
    
    pool_size : int or list
            if list, list length must be equal to number of blocks.  
            
    """

    inp = L.Input(vol_shape)
    
    if stdinput:
        standardizer = L.Lambda(standardize)
        stdinp = standardizer(inp)
    else:
        stdinp = inp
    
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
            code = stdinp
            
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

objects = [standardize]

custom_objects_dict = {'tf': tf}
for item in objects:
    custom_objects_dict[item.__name__] = item








    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

