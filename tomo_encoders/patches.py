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
from numpy.random import default_rng
import h5py
import abc


class Patches():
    
    def __init__(self, vol_shape, initialize_by = "data", \
                 features = None, names = [], **kwargs):
        '''
        A patch is the set of all pixels in a rectangle / cuboid sampled from a (big) image / volume. The Patches data structure allows the following. Think of this as a pandas DataFrame. Each row stores coordinates and features corresponding to a new patch constrained within a big volume of shape vol_shape.  
        
        1. stores coordinates and widths of the patches as arrays of shape (n_pts, z, y, x,) and (n_pts, pz, py, px) respectively.
        2. extracts patches from a big volume and reconstructs a big volume from patches
        3. stores feature vectors evaluated on the patches as an array of shape (n_pts, n_features)
        4. filters, sorts and selects patches based on a feature or features        
        '''
        
        self.vol_shape = vol_shape
        
        initializers = {"data" : self._check_data, \
                        "grid" : self._set_grid, \
                        "multiple-grids" : self._set_multiple_grids, \
                        "random-fixed-width" : self._get_random_fixed_width, \
                        "random" : self._get_random}

        if initialize_by == "file":
            self._load_from_disk(**kwargs)
            return
        else:
            self.points, self.widths, self.check_valid = initializers[initialize_by](**kwargs)
            self._check_valid_points()
            # append features if passed
            self.features = None
            self.feature_names = []
            self.add_features(features, names)
            return

    def dump(self, fpath):
        # create df from points, widths, features
        
        with h5py.File(fpath, 'w') as hf:
            hf.create_dataset("vol_shape", data = self.vol_shape)
            hf.create_dataset("points", data = self.points)
            hf.create_dataset("widths", data = self.widths)
            if self.features is not None:
                hf.create_dataset("features", data = self.features)
            if len(self.feature_names) > 0:
                hf.create_dataset("feature_names", data = np.asarray(self.feature_names, dtype = 'S'))
        return
    
    # use this when initialize_by = "file"
    def _load_from_disk(self, fpath = None):
        
        with h5py.File(fpath, 'r') as hf:
            vol_shape = tuple(np.asarray(hf["vol_shape"]))
            if np.any(vol_shape != self.vol_shape):
                raise ValueError("Volume shape of patches requested does not match the attribute read from the file")
                
            self.points = np.asarray(hf["points"])
            self.widths = np.asarray(hf["widths"])
            if "features" in hf:
                self.features = np.asarray(hf["features"])
            else:
                self.features = None
                
            if "feature_names" in hf:
                out_list = list(hf["feature_names"])
                self.feature_names = [name.decode('UTF-8') for name in out_list]
            else:
                self.feature_names = []

    
    def add_features(self, features, names = []):
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
    
    def append(self, more_patches):
        
        '''
        Append the input patches to self in place.  
        
        Parameters  
        ----------  
        more_patches : Patches
            additional rows of patches to be appended.  
            
        Returns
        -------
        None
            Append in place so nothing is returned.  
        '''
        
        if self.vol_shape != more_patches.vol_shape:
            raise ValueError("patches data is not compatible. Ensure that big volume shapes match")
            
        self.points = np.concatenate([self.points, more_patches.points], axis = 0)
        self.widths = np.concatenate([self.widths, more_patches.widths], axis = 0)
        
        # if features are not stored already, do nothing more
        if self.features is None:
            return
        
        # if feature vector shapes mismatch, numpy will throw an error for concatenate  
        self.features = np.concatenate([self.features, more_patches.features], axis = 0)
        
        # if feature name vectors don't match, raise an error
        if self.feature_names != more_patches.feature_names:
            raise ValueError("feature names in self do not match input")
        return
    
    
    def _check_data(self, points = None, widths = None, check_valid = None):
        
        if len(points) != len(widths):
            raise ValueError("number of anchor points and corresponding widths must match")
        if np.shape(points)[-1] != np.shape(widths)[-1]:
            raise ValueError("dimension mismatch for points and corresponding widths")
            
        points = np.asarray(points)
        widths = np.asarray(widths)
            
        return points, widths, check_valid
    
    def _check_stride(self, patch_size, stride):
        
        '''
        Check if stride is valid by finding the maximum stride possible. Then set the width accordingly.  
        This is calculated as the largest multiple of the original patch size that can fit along the respective axis.  
        '''
        
        # TO-DO: Increasing stride size also increase overlap. At maximum stride, the overlap is nearly the width of the patch, which is weird. Handle this. Perhaps show a warning.  
        
        if stride is None: return patch_size
        
        max_possible_stride = min([self.vol_shape[ii]//patch_size[ii] for ii in range(len(self.vol_shape))])
        if stride > max_possible_stride:
            raise ValueError("Cannot preserve aspect ratio with given value of zoom_out. Pick lower value.")

        return tuple([patch_size[ii]*stride for ii in range(3)])
        

    def _set_multiple_grids(self, min_patch_size = None, \
                          max_stride = None, n_points = None):
        '''
        Sets multiple grids starting from the minimum patch_size up to the maximum using stride as a multiplier. if n_points is passed, returns only that many randomly sampled patches.    
        
        '''
        all_strides = [i+1 for i in range(max_stride)]
        
        points = []
        widths = []
        for stride in all_strides:
            p, w, _ = self._set_grid(patch_size = min_patch_size, stride = stride)
            points.append(p)
            widths.append(w)
        points = np.concatenate(points, axis = 0)
        widths = np.concatenate(widths, axis = 0)
            
        if n_points is not None:
            # sample randomly
            rng = default_rng()
            idxs = rng.choice(points.shape[0], n_points, replace = False)
            points = points[idxs,...].copy()
            widths = widths[idxs,...].copy()
        
        return np.asarray(points), np.asarray(widths), False
                          
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

    def _get_random_fixed_width(self, patch_size = None, n_points = None, stride = None):
        """
        Generator that yields randomly sampled data pairs of number = batch_size.

        Parameters:
        ----------
        patch_size: tuple  
            size of the 3D patch as input volume  
            
        stride : int  
            (optional) width is defined as stride value multiplied by min_patch_size. Default is None (or = 1)    

        n_points: int
            size of the batch (number of patches to be extracted per batch)

        """
        patch_size = self._check_stride(patch_size, stride)
        
        points = np.asarray([np.random.randint(0, self.vol_shape[ii] - patch_size[ii], n_points) \
                           for ii in range(3)]).T
        widths = np.asarray([list(patch_size)]*n_points)
        return np.asarray(points), np.asarray(widths), False
    
    def _get_random(self, min_patch_size = None, max_stride = None, n_points = None):
        """
        Generator that yields randomly sampled data pairs of number = batch_size.

        Parameters:
        ----------
        min_patch_size: tuple
            size of the 3D patch as input volume

        max_stride : int  
            width is defined as stride value multiplied by min_patch_size.    
            
        n_points: int
            size of the batch (number of patches to be extracted per batch)  

        """
        _ = self._check_stride(min_patch_size, max_stride) # check max stride before going into the loop
        random_strides = np.random.randint(1, max_stride, n_points)
        points = []
        widths = []
        for stride in random_strides:
            curr_patch_size = self._check_stride(min_patch_size, stride)
            points.append([np.random.randint(0, self.vol_shape[ii] - curr_patch_size[ii]) for ii in range(3)])    
            widths.append(list(curr_patch_size))

        points = np.asarray(points)
        widths = np.asarray(widths)        
        
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
    
    def _points_to_slices(self, a, w, b):
        
        # b is binning, a is the array of start values and w = stop - start (width)
        return [[slice(a[ii,jj], a[ii,jj] + w[ii,jj], b[ii]) for jj in range(3)] for ii in range(len(a))]
    
    def slices(self, binning = None):
        '''  
        Get python slice objects from the list of coordinates  
        
        Returns  
        -------  
        np.ndarray (n_pts, 3)    
            each element of the array is a slice object  
        
        '''  
        
        if binning is None:
            binning = [1]*len(self.points)
        elif isinstance(binning, int):
            binning = [binning]*len(self.points)
            
        s = self._points_to_slices(self.points, self.widths, binning)
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
                       features = np.compress(cond_list, self.features, axis = 0), \
                       names = self.feature_names)
    
    def select_by_indices(self, idxs):

        '''
        Select patches corresponding to the input list of indices.  
        Parameters
        ----------
        idxs : list  
            list of integers as indices.  
        '''
        
        return Patches(self.vol_shape, initialize_by = "data", \
                       points = self.points[idxs],\
                       widths = self.widths[idxs],\
                       features = self.features[idxs], \
                       names = self.feature_names)
        
    def select_random_sample(self, n_points):
        
        '''
        Select a given number of patches randomly without replacement.  
        
        Parameters
        ----------
        n_points : list  
            list of integers as indices.  
        '''
        rng = default_rng()
        idxs = rng.choice(self.points.shape[0], n_points, replace = False)
        return self.select_by_indices(idxs)
    
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
        
        return self.select_by_indices(idxs)
    
    def select_by_feature(self, n_selections,\
                       feature = None, ife = None,\
                       selection_by = "highest"):
        '''  
        Select highest (or lowest) n_selections patches based on a feature value. The values are sorted (starting with highest or lowest), then the first n_selections values are chosen. For e.g., if feature contains values 0.1, 0.9, 1.5, 2.0, n_selections = 2 and selection_by = highest, then the patches having feature value 2.0 and 1.5 will be selected.  
        
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

        return self.select_by_indices(idxs)
        
    def _calc_binning(self, patch_size):        
        bin_vals = self.widths//np.asarray(patch_size)
        cond1 = np.sum(np.max(bin_vals, axis = 1) != np.min(bin_vals, axis = 1)) > 0
        cond2 = np.any(bin_vals == 0)
        cond3 = np.any(self.widths%np.asarray(patch_size))
        
        if cond1: # binning is assumed to be isotropic so aspect ratio must be preserved
            raise ValueError("aspect ratios of some patches don't match!! Cannot bin to patch_size")
        if cond2: # avoid the need to upsample patches, can use a smaller model instead
            raise ValueError("patch_size cannot be larger than any given patch in the list")
        if cond3: # constraint for ensuring stitch() works
            raise ValueError("stitch only works when binning values are even numbers")

        return bin_vals[:,0]
    
    def extract(self, vol, patch_size):

        '''  
        Returns a list of volume patches at the active list of coordinates by drawing from the given big volume 'vol'  
        
        Returns
        -------
        np.ndarray  
            shape is (n_pts, patch_z, patch_y, patch_x)  
        
        '''  
        if vol.shape != self.vol_shape:
            raise ValueError("Shape of big volume does not match vol_shape attribute of patches data")

        # calculate binning
        bin_vals = self._calc_binning(patch_size)
        # make a list of slices
        s = self.slices(binning = bin_vals)
        # make a list of patches
        sub_vols = [np.asarray(vol[s[ii,0], s[ii,1], s[ii,2]]) for ii in range(len(self.points))]
        
        return np.asarray(sub_vols, dtype = vol.dtype)
    
    def stitch(self, sub_vols, patch_size, upsample = False):
        '''  
        Stitches the big volume from a list of volume patches (with upsampling).    
        
        Returns
        -------
        np.ndarray  
            shape is (n_pts, patch_z, patch_y, patch_x)  
        
        '''  
        
        if sub_vols.shape[0] != len(self.points):
            raise ValueError("number of patch points and length of input list of patches must match")
        vol = np.zeros(self.vol_shape, dtype = sub_vols.dtype)
        
        # calculate binning
        bin_vals = self._calc_binning(patch_size)
        # make a list of slices
        s = self.slices(binning = bin_vals)
        for ii in range(len(self.points)):
            if upsample:
                for ibin in range(bin_vals[ii]):
                    vol[self._slice_shift(s[ii,0], ibin),\
                        self._slice_shift(s[ii,1], ibin),\
                        self._slice_shift(s[ii,2], ibin)] = sub_vols[ii]
            else:
                vol[s[ii,0],s[ii,1],s[ii,2]] = sub_vols[ii]
        return vol

    def _slice_shift(self, s, shift):
        return slice(s.start + shift, s.stop + shift, s.step)
    
    
    
if __name__ == "__main__":
    
    print('just a bunch of functions')
    
