#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Easily visualize U-net intermediate activations

"""
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Lambda
import tensorflow as tf
import numpy as np


def get_interim_layer(model, layer_id, flatten = False, binning = 1, layer_only = False, filt_binning = 1):

    if type(layer_id) == str:
        layer_names = [layer.name for layer in model.layers]
        layer_idx = layer_names.index(layer_id)
    
    inp = model.input
    out = model.layers[layer_idx].output
    print(out.name)

    
    if binning > 1:
        newsize = out.shape[1]//binning
        out = Lambda(lambda image: tf.image.resize(image, (newsize,newsize)), name = "lambda_99")(out)

    if filt_binning > 1:
        out = Lambda(lambda image: image[:,:,:,::filt_binning], name = "lambda_98")(out)
        
    if flatten:
        out = Flatten()(out)
    
    if layer_only:
        return out
    else:
        return keras.models.Model(inputs = inp, outputs = out)

def plot_feature_maps(input_img, model, layer_name, ax = None, ncols = 16, gap = 2, axis3D = None):
    '''
    Plot feature maps on a matplotlib axes  
    
    Parameters
    ----------
    input_img : np.array
        Input image for which feature maps are to be computed. Can be 2D or 3D shape (ny,nx) or (nz,ny,nx).  
    model : tf.keras.model  
        Keras U-net-like model (encoder-decoder). Input and output share must match.  
    layer_name : str  
        Name of layer whose activations are to be visualized.  
    ncols : int  
        number of columns in the tiled image showing the feature maps  
    axis3D : None or int  
        if 3D image, an axis (0, 1 or 2) must be provided to draw slices for a 2D plot  
    gap : int  
        gap in pixel between adjacent maps in the tiled image  
        
    '''
    
    layer_out = get_interim_layer(model, layer_name)
    fmaps = layer_out.predict(input_img[np.newaxis,...,np.newaxis])[0]
    
    plot_img = tile_feature_maps(fmaps, ncols = ncols, gap = gap, axis3D = axis3D)
    if ax is None:
        figsize = (8, 8*float(plot_img.shape[0])/plot_img.shape[1])
        fig, ax = plt.subplots(1,1, figsize = figsize)
    ax.imshow(plot_img, cmap = 'gray')
    ax.axis('off')
    
    return
    
    
def tile_feature_maps(fmaps, ncols = 16, gap = 2, axis3D = None):
    
    '''
    Tile feature maps into a 2D numpy array (image)  
    
    Parameters
    ----------
    fmaps : np.array
        feature maps for a given input (2D/3D) image (nz, ny, nx, n_channels)  
    ncols : int  
        number of columns in the tiled image showing the feature maps  
    axis3D : None or int  
        if 3D image, an axis (0, 1 or 2) must be provided to draw slices for a 2D plot  
    gap : int  
        gap in pixel between adjacent maps in the tiled image  
    '''
    # if feature maps are 3D images, draw a slice along axis3D (0, 1 or 2)
    if axis3D is not None:
        fmaps = fmaps.take(int(fmaps.shape[axis3D]//2), axis = axis3D)
    
    fmaps = np.moveaxis(fmaps, 2, 0)
    
    flen = fmaps.shape[0]
    fsize = fmaps.shape[1] # assume a square-shaped feature map
    nrows = flen//ncols
    if nrows*ncols != flen:
        raise ValueError("shape is not valid")
        
    img_shape = (nrows*(fsize+gap), ncols*(fsize+gap))
    img = np.ones(img_shape)
    
    
    idx_map = 0
    for ii in range(nrows):
        for jj in range(ncols):
            
            fmap = fmaps[idx_map].copy()
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min())
            slice_y = slice(ii*(gap+fsize), ii*(gap+fsize) + fsize)
            slice_x = slice(jj*(gap+fsize), jj*(gap+fsize) + fsize)
            img[slice_y, slice_x] = fmap
            idx_map += 1
    
    return img



def view_midplanes(vol = None, ax = None, cmap = 'gray', alpha = None, idxs = None, axis_off = False, label_planes = True):
    """View 3 images drawn from planes through axis 0, 1, and 2 at indices listed (idx). Do this for a DataFile or numpy.array
    
    Parameters
    ----------
    
    ax : matplotlib.axes  
        three axes  
    
    vol : numpy.array  
        3D numpy array  
    
    """

    if ax is None:
        fig, ax = plt.subplots(1,3)
    imgs = get_orthoplanes(vol = vol, idxs = idxs)
    for i in range(3):
        ax[i].imshow(imgs[i], cmap = cmap, alpha = alpha)
    
    if label_planes:
        h = ax[0].set_title("XY mid-plane")
        h = ax[1].set_title("XZ mid-plane")
        h = ax[2].set_title("YZ mid-plane")    
    
    if axis_off:
        for ii in range(3):
            ax[ii].axis('off')
    
    return ax

def get_orthoplanes(vol = None, idxs = None):
    """Return 3 images drawn from planes through axis 0, 1, and 2 at indices listed (idx). Do this for a DataFile or numpy.array
    
    Returns
    -------
    list  
        images at three midplanes  
    
    Parameters
    ----------
    ax : matplotlib.axes  
        three axes  
    
    vol : numpy.array  
        3D numpy array  
    
    """
    
    if vol is not None:
        if idxs is None: idxs = [vol.shape[i]//2 for i in range(3)]
        imgs = [vol.take(idxs[i], axis = i) for i in range(3)]
    else:
        raise NotImplementedError("must be only volume data object")
        
    return imgs    
