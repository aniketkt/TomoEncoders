#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Util functions for:  
1. data (patched volume) generator from any given volume data pair  
2. Handle inference on 3D auto-encoder  
3. Porespy data generator  

https://porespy.readthedocs.io/en/master/getting_started.html#generating-an-image  

"""
import h5py
import numpy as np
import os
import pandas as pd
from tomo_encoders.misc.img_stats import calc_SNR
import time


def read_data_pairs(fpaths, normalize = False, data_tags =  ("recon", "gt_labels"), group_tags = [], downres = 1):
    '''
    Read data pair
    Returns
    -------
    tuple
        X, Y where each is a 3D numpy array.
    Parameters
    ----------
    fpaths : str or list 
        Path to .hdf5 file. Must contain "data4D" and "mask4D" datasets with shape (nt, nz, ny, nx). If list is provided, multiple data pairs are read.  
    data_tags : tuple  
        hdf5 group path of input volume, ground truth volume respectively  
        
    normalize : bool  
        If True, X data will be normalized between [0,1]  
    
    '''
    
    # this needs more work - not RAM-efficient  
    
    if type(fpaths) is str:
        fpaths = [fpaths]
        
    
    Xs = []
    Ys = []
    
    for ii, fpath in enumerate(fpaths):
        
        hf = h5py.File(fpath, 'r')
        X = np.asarray(hf[os.path.join(*group_tags, data_tags[0])][:], dtype = np.float32)
        
        if data_tags[1] is not None:
            Y = np.asarray(hf[os.path.join(*group_tags, data_tags[1])][:], \
                           dtype = np.uint8)
        else:
            Y = None
            
        if downres > 1:
            X = X[::downres, ::downres, ::downres]
            if Y is not None:
                Y = Y[::downres, ::downres, ::downres]
            
        hf.close()
        data_shape = X.shape

        if normalize:
            X = _normalize_volume(X)

        if Y is not None:
            Y = (Y > 0).astype(np.uint8) # ensure there are only two labels 0 and 1
        
        Xs.append(X.copy())
        if Y is not None:
            Ys.append(Y.copy())
        
        if ii == 0:
            check_shape = X.shape
        else:
            if X.shape != check_shape:
                raise ValueError("All volumes must have same shape")

        del X
        del Y
        
        
    Xs = np.asarray(Xs)
    
    if data_tags[1] is not None:
        Ys = np.asarray(Ys)
    else:
        Ys = None
    
    return Xs, Ys
    
def get_data_from_flist(csv_path, **kwargs):
    fpaths = pd.read_csv(csv_path)
#     fpaths = fpaths.sort_values(by = "name")
    fpaths = list(fpaths["name"])
    fpaths = [os.path.join(os.path.split(csv_path)[0], fpath) for fpath in fpaths]
    data_labels = [os.path.split(path)[-1].split(".hdf5")[0] for path in fpaths]
    Xs, Ys = read_data_pairs(fpaths, **kwargs)
#     print(data_labels)
    return Xs, Ys, data_labels

def _normalize_volume(vol):
    '''
    Normalizes volume to values into range [0,1]  
    
    '''
    eps = 1e-12
    max_val = np.max(vol)
    min_val = np.min(vol)
    vol = (vol - min_val) / (max_val - min_val + eps)
    return vol
    
def get_patch(vol, coordinates, patch_size = (64,64,64)):
    
    """
    Get a volumetric patch at given coordinates
    vol : np.array
        Input volume (arbitrarily sized)
    coordinates : tuple
        tuple of coordinates in terms of the 3D array indices
        
    """
    
    iz, iy, ix = coordinates
    pz, py, px = patch_size
    
#     if ((iz < pz) | (iy < py) | (ix < px)):
    if ((iz - pz//2 < 0) | (iy - py//2 < 0) | (ix - px//2 < 0)):
        raise ValueError("coordinates must be greater than minimum index")
    elif ((iz + pz//2 > vol.shape[0]) | (iy + py//2 > vol.shape[1]) | (ix + px//2 > vol.shape[2])):
        raise ValueError("coordinates must be smaller than maximum index")
    
    return vol[int(iz-pz//2): int(iz+pz//2), \
               int(iy-py//2): int(iy+py//2), \
               int(ix-px//2): int(ix+px//2)]

def make_ellipsoid(patch_size, rad, ea):
    pts = np.arange(-patch_size[0]//2, patch_size[0]//2)
    zz, yy, xx = np.meshgrid(pts, pts, pts, indexing = 'ij')
    dist = np.sqrt(zz**2/ea[0]**2 + yy**2/ea[1]**2 + xx**2/ea[2]**2)
    sph = dist > rad
    
    return sph.astype(np.uint8)
    
def ellipse_or_sphere_generator(patch_size, batch_size, \
                      ellipse_range = (1.5, 1.8), \
                      rad_range = (10,16), add_noise = 0.1, scan_idx = False):
    '''
    Generator that yields randomly sampled data pairs of number = batch_size, ellipsoids with ellipse parameters uniformly sampled from a range.  
    
    Parameters  
    ----------   
    patch_size : tuple  
        size of the 3D patch as input volume  
        
    batch_size : int  
        size of the batch (number of patches to be extracted per batch)  
    
    Returns  
    -------  
    tuple  
        x, y numpy arrays each of shape (batch_size, ny, ny, 1)   
    
    '''
    
    while True:
        ea_labels = np.random.randint(0,2, batch_size)
        
        ea_vals = np.random.uniform(*ellipse_range, (batch_size, 3))
        rad_vals = np.random.uniform(*rad_range, batch_size)
        ea_vals[ea_labels == 1,:] = 1.0
        
        y = np.asarray([make_ellipsoid(patch_size, rad_vals[ii], ea_vals[ii,...]) for ii in range(len(rad_vals))])
        std_batch = np.random.uniform(0, add_noise, batch_size)
        
        x = y + np.asarray([np.random.normal(0, std_batch[ii], y.shape[1:]) for ii in range(batch_size)])
        
        x = x[...,np.newaxis]
        y = y[...,np.newaxis]
        
        if scan_idx:
            yield (x, y, ea_labels) 
        else:
            yield (x, y)

def get_sub_vols_elliptical_voids(patch_size, batch_size, \
                                  ellipse_range = (1.5, 1.8), \
                                  rad_range = (10,16)):

        ea_labels = np.random.randint(0,2, batch_size)
        
        ea_vals = np.random.uniform(*ellipse_range, (batch_size, 3))
        rad_vals = np.random.uniform(*rad_range, batch_size)
        ea_vals[ea_labels == 1,:] = 1.0
        
        return np.asarray([make_ellipsoid(patch_size, rad_vals[ii], ea_vals[ii,...]) for ii in range(len(rad_vals))])

            
def ellipse_generator(patch_size, batch_size, \
                      ellipse_range = (0.3, 1.8), \
                      rad_range = (10,16), add_noise = 0.1, \
                      scan_idx = False, discrete_only = True):
    '''
    Generator that yields randomly sampled data pairs of number = batch_size, ellipsoids with ellipse parameters uniformly sampled from a range.  
    
    Parameters  
    ----------   
    patch_size : tuple  
        size of the 3D patch as input volume  
        
    batch_size : int  
        size of the batch (number of patches to be extracted per batch)  
    
    Returns  
    -------  
    tuple  
        x, y numpy arrays each of shape (batch_size, ny, ny, 1)   
    
    '''
    
    while True:
        ea_vals = np.random.uniform(*ellipse_range, (batch_size, 3))
        rad_vals = np.random.uniform(*rad_range, batch_size)
        
        if discrete_only:
            ea_labels = np.random.randint(0, 2, batch_size)
            ea_vals[ea_labels == 1,:] = 1.0
        else:
            ea_labels = np.std(ea_vals, axis = 1)
        
        y = np.asarray([make_ellipsoid(patch_size, rad_vals[ii], ea_vals[ii,...]) for ii in range(len(rad_vals))])
        std_batch = np.random.uniform(0, add_noise, batch_size)
        
        x = y + np.asarray([np.random.normal(0, std_batch[ii], y.shape[1:]) for ii in range(batch_size)])
        
        x = x[...,np.newaxis]
        y = y[...,np.newaxis]
        
        if scan_idx:
            yield (x, y, ea_labels) 
        else:
            yield (x, y)
    
    
    
    

