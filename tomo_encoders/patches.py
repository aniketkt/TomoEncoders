#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
implementation of the patches data structure  


"""

import pandas as pd
import os
import glob
import numpy as np


from skimage.feature import match_template
import h5py

from tomopy import normalize, minus_log, angles, recon, circ_mask
from scipy.ndimage.filters import median_filter

# from patch_maker_3D import *

from multiprocessing import Pool, cpu_count
import functools

import numpy as np
import h5py
import abc


class Patches():
    
    def __init__(self, vol_shape, initialize_by = "data", \
                 features = None, names = [], **kwargs):
        '''
        
        '''
        
        self.vol_shape = vol_shape
        
        initializers = {"data" : self._check_data, \
                        "grid" : self._set_grid}

        self.points, self.widths, self.check_valid = initializers[initialize_by](**kwargs)
        self._check_valid_points()

        # append features if passed
        self.features = None
        self.feature_names = []
        self.append_features(features, names)
        return
    
    
    def append_features(self, features, names = []):
        '''
        Store features corresponding to patch coordinates.
        
        Parameters
        ----------
        features : np.array  
            array of features, must be same length and as corresponding patch coordinates.  
        '''

        if features is None:
            return
        
        # handle feature array here
        if len(self.points) != len(features): # check length
            raise ValueError("numbers of anchor points and corresponding features must match")
        if features.ndim != 2: # check shape
            raise ValueError("shape of features array not valid")
        else:
            npts, nfe = features.shape

        if self.features is not None:
            self.features = np.concatenate([self.features, features], axis = 1)
        else:
            self.features = features
            
        # handle feature names here
        cond1 = len(self.feature_names) != 0
        cond2 = len(names) != features.shape[-1]
        if cond1 and cond2:
            raise ValueError("feature array and corresponding names input are not compatible")
        else:
            self.feature_names += names
        
        return
    
    
    
    
    def _check_data(self, points = None, widths = None, check_valid = None):
        
        if len(points) != len(widths):
            raise ValueError("number of anchor points and corresponding widths must match")
        if np.shape(points)[-1] != np.shape(widths)[-1]:
            raise ValueError("dimension mismatch for points and corresponding widths")
            
        points = np.asarray(points)
        widths = np.asarray(widths)
            
        return points, widths, check_valid
    
    def features_to_numpy(self, names):
        '''
        Parameters
        ----------
        names : list of strings with names of features  
        
        '''
        
        if self.feature_names is None: raise ValueError("feature names must be defined first.")
        out_list = []
        for name in names:
            out_list.append(self.features[:,self.feature_names.index(name)])
        return np.asarray(out_list).T
    
    def _check_stride(self, patch_size, stride):
        
        '''
        Check if stride is valid by finding the maximum stride possible. Then set the width accordingly.  
        This is calculated as the largest multiple of the original patch size that can fit along the respective axis.  
        '''
        
        # TO-DO: Increasing stride size also increase overlap. At maximum stride, the overlap is nearly the width of the patch, which is weird. Handle this. Perhaps show a warning.  
        
        if stride is None: return patch_size
        
        max_stride = min([self.vol_shape[ii]//patch_size[ii] for ii in range(len(self.vol_shape))])
        if stride > max_stride:
            raise ValueError("Cannot preserve aspect ratio with given value of zoom_out. Pick lower value.")

        return tuple([patch_size[ii]*stride for ii in range(3)])
        
    def _set_grid(self, patch_size = None, stride = None):

        '''
        Initialize (n,3) points on the corner of volume patches placed on a grid. Some overlap is introduced to prevent edge effects while stitching.  
        
        Parameters  
        ----------  
        patch_size : tuple  
            A tuple of widths (or the smallest possible values thereof) of the patch volume  
            
        stride : int  
            Effectively multiplies the patch size by factor of stride.    
        
        '''
        
        patch_size = self._check_stride(patch_size, stride)
#         import pdb; pdb.set_trace()
        
        # Find optimum number of patches to cover full image
        mz, my, mx = self.vol_shape
        pz, py, px = patch_size
        nx, ny, nz = int(np.ceil(mx/px)), int(np.ceil(my/py)), int(np.ceil(mz/pz))
        stepx = (mx-px) // (nx-1) if mx != px else 0
        stepy = (my-py) // (ny-1) if my != py else 0
        stepz = (mz-pz) // (nz-1) if mz != pz else 0
        
        stepsize  = (stepz, stepy, stepx)
        nsteps = (nz, ny, nx)
        
        points = []
        for ii in range(nsteps[0]):
            for jj in range(nsteps[1]):
                for kk in range(nsteps[2]):
                    points.append([ii*stepsize[0], jj*stepsize[1], kk*stepsize[2]])
        widths = [list(patch_size)]*len(points)
        
        return np.asarray(points), np.asarray(widths), False

    def _check_valid_points(self):
        is_valid = True
        for ii in range(len(self.points)):
            for ic in range(self.points.shape[-1]):
                cond1 = self.points[ii,ic] < 0
                cond2 = self.points[ii,ic] + self.widths[ii,ic] > self.vol_shape[ic]
                if any([cond1, cond2]):
                    print("Patch %i, %s, %s is invalid"%(ii, str(self.points[ii]), str(self.widths[ii])))
                    is_valid = False
        
        if not is_valid:
            raise ValueError("Some points are invalid")
        return
    
    def slices(self):
        '''  
        Get python slice objects from the list of coordinates  
        
        Returns  
        -------  
        np.ndarray (n_pts, 3)    
            each element of the array is a slice object  
        
        '''  
        
        s = [[slice(self.points[ii,jj], self.points[ii,jj] + self.widths[ii,jj]) for jj in range(3)] for ii in range(len(self.points))]
        return np.asarray(s)
    
    def centers(self):
        '''  
        Get centers of the patch volumes.    
        
        Returns  
        -------  
        np.ndarray (n_pts, 3)    
            each element of the array is the z, y, x coordinate of the center of the patch volume.    
        
        '''  
        
        s = [[int(self.points[ii,jj] + self.widths[ii,jj]//2) for jj in range(3)] for ii in range(len(self.points))]
        return np.asarray(s)
    
        
            
    def filter_by_condition(self, cond_list):
        '''  
        Select coordinates based on condition list. Here we use numpy.compress. The input cond_list can be from a number of classifiers.  
        
        Parameters  
        ----------  
        cond_list : np.ndarray  
            array with shape (n_pts, n_conditions). Selection will be done based on ALL conditions being met for the given patch.  
        '''  
        
        if cond_list.shape[0] != len(self.points):
            raise ValueError("length of condition list must same as the current number of stored points")
        
        if cond_list.ndim == 2:
            cond_list = np.prod(cond_list, axis = 1) # AND operator on all conditions
        elif cond_list.ndim > 2:
            raise ValueError("condition list must have 1 or 2 dimensions like so (n_pts,) or (n_pts, n_conditions)")
            
        return Patches(self.vol_shape, initialize_by = "data", \
                       points = np.compress(cond_list, self.points, axis = 0),\
                       widths = np.compress(cond_list, self.widths, axis = 0),\
                       features = np.compress(cond_list, self.features, axis = 0))
    
        
    def sort_by_feature(self, feature = None, ife = None):
        '''  
        Sort patches list in ascending order of the value of a feature.    
        
        Parameters  
        ----------  
        ife : int  
            index of feature to be used for sorting. The features are accessed from  the current instance of patches  
        feature : np.ndarray  
            array with shape (n_pts,). If provided separately, ife will be ignored.  
        
        '''  
        
        if feature is None:
            feature = self.features[:,ife]
        else:
            if feature.ndim != 1: raise ValueError("feature must be 1D array")
            if len(feature) != len(self.points): raise ValueError("length of feature array must match number of patch points")
        
        idxs = np.argsort(feature)
        return Patches(self.vol_shape, initialize_by = "data", \
                       points = self.points[idxs],\
                       widths = self.widths[idxs],\
                       features = self.features[idxs])
    
    def select_patches(self, n_selections,\
                       feature = None, ife = None,\
                       selection_by = "highest"):
        '''  
        Select "n_selections" patches having the "highest" or "lowest" feature value. The values are sorted (starting with highest or lowest), then the first n_selections values are chosen. For e.g., if feature contains values 0.1, 0.9, 1.5, 2.0, n_selections = 2 and selection_by = highest, then the patches having feature value 2.0 and 1.5 will be selected.  
        
        Parameters  
        ----------  
        ife : int  
            index of feature to be used for sorting. The features are accessed from  the current instance of patches  
        feature : np.ndarray  
            array with shape (n_pts,). If provided separately, ife will be ignored.  
        
        selection_by : str  
            highest or lowest; if highest, the highest-valued n_selections will be selected.  
        
        '''  
        
        if feature is None:
            feature = self.features[:,ife]
        else:
            if feature.ndim != 1: raise ValueError("feature must be 1D array")
            if len(feature) != len(self.points): raise ValueError("length of feature array must match number of patch points")
        
        idxs = np.argsort(feature)
        n_selections = min(n_selections, len(self.points))
            
        if selection_by == "highest":
            idxs = idxs[::-1]
        idxs = idxs[:n_selections]

        return Patches(self.vol_shape, initialize_by = "data", \
                       points = self.points[idxs],\
                       widths = self.widths[idxs],\
                       features = self.features[idxs])

    def extract(self, vol):

        '''  
        Returns a list of volume patches at the active list of coordinates by drawing from the given big volume 'vol'  
        
        Returns
        -------
        np.ndarray  
            shape is (n_pts, patch_z, patch_y, patch_x)  
        
        '''  
        
        if vol.shape != self.vol_shape:
            raise ValueError("Shape of big volume does not match vol_shape attribute of patches data")
        
        # make a list of slices
        s = self.slices()
        # make a list of patches
        p = [np.asarray(vol[s[ii,0], s[ii,1], s[ii,2]]) for ii in range(len(self.points))]
        
        return p
    
    def reconstruct(self, p):
        '''  
        Reconstructs the big volume from a list of volume patches provided.  
        
        Returns
        -------
        np.ndarray  
            shape is (n_pts, patch_z, patch_y, patch_x)  
        
        '''  
        
        if p.shape[0] != len(self.points):
            raise ValueError("number of patch points and length of input list of patches must match")
            
        vol = np.zeros(self.vol_shape, dtype = p.dtype)
        
        s = self.slices()
        for ii in range(len(self.points)):
            vol[s[ii,0],s[ii,1],s[ii,2]] = p[ii]
        return vol
        
if __name__ == "__main__":
    
    print('just a bunch of functions')
    
