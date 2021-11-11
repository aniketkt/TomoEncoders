#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
implementation of the patches data structure  


"""

import pandas as pd
import os
import glob
import numpy as np


import h5py
import cupy as cp

from multiprocessing import Pool, cpu_count
import functools
import time

from numpy.random import default_rng
import abc
import tensorflow as tf
from tensorflow.keras.layers import UpSampling3D


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
                        "slices" : self._from_slices,\
                        "grid" : self._set_grid, \
                        "regular-grid" : self._set_regular_grid, \
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

        
    def __len__(self):
        return len(self.points)
    
    def dump(self, fpath):
        # create df from points, widths, features
        
        with h5py.File(fpath, 'w') as hf:
            hf.create_dataset("vol_shape", data = self.vol_shape)
            hf.create_dataset("points", data = self.points)
            hf.create_dataset("widths", data = self.widths)
            if self.features is not None:
                hf.create_dataset("features", data = self.features)
            if any(self.feature_names):
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
        cond1 = any(self.feature_names)
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

    def _from_slices(self, s = None, check_valid = None):
        
        plen = len(s)
        _ndim = len(s[0])
        
        points = np.empty((plen, 3))
        widths = np.empty((plen, 3))
        for ii in range(plen):
            points[ii,...] = np.asarray([s[ii][ia].start for ia in range(_ndim)])
            widths[ii,...] = np.asarray([s[ii][ia].stop - s[ii][ia].start for ia in range(_ndim)])
            
        points = np.asarray(points).astype(int)
        widths = np.asarray(widths).astype(int)
            
        return points, widths, check_valid
    
    def _check_stride(self, patch_size, stride):
        
        '''
        Check if stride is valid by finding the maximum stride possible. Then set the width accordingly.  
        This is calculated as the largest multiple of the original patch size that can fit along the respective axis.  
        '''
        
        # to-do: When using "grid", increasing stride size also increase overlap. At maximum stride, the overlap is nearly the width of the patch, which is weird. Handle this. Perhaps show a warning.  
        
        _ndims = len(self.vol_shape)
        if stride is None: return patch_size
        
        max_possible_stride = min([self.vol_shape[ii]//patch_size[ii] for ii in range(_ndims)])
        if stride > max_possible_stride:
            raise ValueError("Cannot preserve aspect ratio with given combination of patch_size and stride. Try lower values.")

        return tuple([patch_size[ii]*stride for ii in range(_ndims)])
        
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
            n_points = min(n_points, len(points))
            # sample randomly
            rng = default_rng()
            idxs = rng.choice(points.shape[0], n_points, replace = False)
            points = points[idxs,...].copy()
            widths = widths[idxs,...].copy()
        
        return np.asarray(points), np.asarray(widths), False
                          
    def _set_grid(self, patch_size = None, stride = 1, n_points = None):

        '''
        Initialize (n,3) points on the corner of volume patches placed on a grid. Some overlap is introduced to cover the full volume.  
        
        Parameters  
        ----------  
        patch_size : tuple  
            A tuple of widths (or the smallest possible values thereof) of the patch volume  
            
        stride : int  
            Effectively multiplies the patch size by factor of stride.    
        
        '''
        patch_size = self._check_stride(patch_size, stride)
        # Find optimum number of patches to cover full image
        # this was rewritten to accommodate both 2D and 3D patches.
        # old code commented out below
        m = list(self.vol_shape)
        p = list(patch_size)
        
        nsteps = [int(np.ceil(m[i]/p[i])) for i in range(len(m))]
        
        stepsize = []
        for i in range(len(nsteps)):
            _s = (m[i]-p[i]) // (nsteps[i]-1) if m[i] != p[i] else 0
            stepsize.append(_s)

        points = []            
        if len(nsteps) == 3:
            for ii in range(nsteps[0]):
                for jj in range(nsteps[1]):
                    for kk in range(nsteps[2]):
                        points.append([ii*stepsize[0], jj*stepsize[1], kk*stepsize[2]])
        elif len(nsteps) == 2:
            for ii in range(nsteps[0]):
                for jj in range(nsteps[1]):
                    points.append([ii*stepsize[0], jj*stepsize[1]])
        
        widths = [list(patch_size)]*len(points)
        
        
        if n_points is not None:
            n_points = min(n_points, len(points))
            points = np.asarray(points)
            widths = np.asarray(widths)
            # sample randomly
            rng = default_rng()
            idxs = rng.choice(points.shape[0], n_points, replace = False)
            points = points[idxs,...].copy()
            widths = widths[idxs,...].copy()
        
        return np.asarray(points), np.asarray(widths), False
            
    def _set_regular_grid(self, patch_size = None, n_points = None):

        '''
        Initialize (n,3) points on the corner of volume patches placed on a grid. No overlap is used. Instead, the volume is cropped such that it is divisible by the patch_size in that dimension.  
        
        Parameters  
        ----------  
        patch_size : tuple  
            A tuple of widths (or the smallest possible values thereof) of the patch volume  
        
        '''
        
        patch_size = self._check_stride(patch_size, 1)

        
        # Find optimum number of patches to cover full image
        m = list(self.vol_shape)
        p = list(patch_size)
        
        nsteps = [int(m[i]//p[i]) for i in range(len(m))]
        stepsize = patch_size
        
        points = []
        if len(m) == 3:
            for ii in range(nsteps[0]):
                for jj in range(nsteps[1]):
                    for kk in range(nsteps[2]):
                        points.append([ii*stepsize[0], jj*stepsize[1], kk*stepsize[2]])
        elif len(m) == 2:
            for ii in range(nsteps[0]):
                for jj in range(nsteps[1]):
                    points.append([ii*stepsize[0], jj*stepsize[1]])
        
        widths = [list(patch_size)]*len(points)
        
        if n_points is not None:
            n_points = min(n_points, len(points))
            points = np.asarray(points)
            widths = np.asarray(widths)
            # sample randomly
            rng = default_rng()
            idxs = rng.choice(points.shape[0], n_points, replace = False)
            points = points[idxs,...].copy()
            widths = widths[idxs,...].copy()
        
        return np.asarray(points), np.asarray(widths), False
    
    
    def _get_random_fixed_width(self, patch_size = None, n_points = None):
        """
        Generator that yields randomly sampled data pairs of number = batch_size.

        Parameters:
        ----------
        patch_size: tuple  
            size of the 3D patch as input volume  
            
        n_points: int
            size of the batch (number of patches to be extracted per batch)

        """
        _ndim = len(self.vol_shape)
        patch_size = self._check_stride(patch_size, 1)
        
        points = np.asarray([np.random.randint(0, self.vol_shape[ii] - patch_size[ii], n_points) \
                           for ii in range(_ndim)]).T
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
        _ndim = len(self.vol_shape)
        _ = self._check_stride(min_patch_size, max_stride) # check max stride before going into the loop
        random_strides = np.random.randint(1, max_stride, n_points)
        points = []
        widths = []
        for stride in random_strides:
            curr_patch_size = self._check_stride(min_patch_size, stride)
            points.append([np.random.randint(0, self.vol_shape[ii] - curr_patch_size[ii]) for ii in range(_ndim)])    
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
        _ndim = len(self.vol_shape)
        return [[slice(a[ii,jj], a[ii,jj] + w[ii,jj], b[ii]) for jj in range(_ndim)] for ii in range(len(a))]
    
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
        _ndim = len(self.vol_shape)
        s = [[int(self.points[ii,jj] + self.widths[ii,jj]//2) for jj in range(_ndim)] for ii in range(len(self.points))]
        return np.asarray(s)
    
    def features_to_numpy(self, names):
        '''
        Parameters
        ----------
        names : list of strings with names of features  
        
        '''
        
        if not any(self.feature_names): raise ValueError("feature names must be defined first.")
        out_list = []
        for name in names:
            out_list.append(self.features[:,self.feature_names.index(name)])
        return np.asarray(out_list).T
    
    
    def filter_by_cylindrical_mask(self, mask_ratio = 0.9, height_ratio = 1.0):
        '''
        Selects patches whose centers lie inside a cylindrical volume of radius = mask_ratio*nx/2. This assumes that the volume shape is a tomogram where ny = nx. The patches are filtered along the vertical (or z) axis if height_ratio < 1.0.  
        '''
        
        max_radius = int(mask_ratio*self.vol_shape[-1]/2.0)
        max_z_from_center = int(height_ratio*self.vol_shape[0]/2.0)
        
        p_centers = self.centers()
        return
    
    
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
                       features = None if self.features is None else np.compress(cond_list, self.features, axis = 0), \
                       names = self.feature_names if any(self.feature_names) else [])
    
    def copy(self):
        
        _fcopy = None if self.features is None else self.features.copy()
        _names = self.feature_names.copy() if any(self.feature_names) else []
        
        return Patches(self.vol_shape, initialize_by = "data", \
                       points = self.points.copy(),\
                       widths = self.widths.copy(),\
                       features = _fcopy, \
                       names = _names)
        

    def rescale(self, fac, new_vol_shape):
        '''
        '''
        
        _fcopy = None if self.features is None else self.features.copy()
        _names = self.feature_names.copy() if any(self.feature_names) else []
        
        fac = int(fac)
        px_max = np.max(self.points, axis = 0)*fac
        extra_pix = np.asarray(new_vol_shape) - px_max
        
#         cond0 = extra_pix < 0
#         cond1 = extra_pix > 1
#         cond_fin = cond0 | cond1
        cond_fin = extra_pix < 0
        
        if np.any(cond_fin):
            raise ValueError("new volume shape is inappropriate")
        else:
            return Patches(new_vol_shape, initialize_by = "data", \
                           points = self.points.copy()*fac,\
                           widths = self.widths.copy()*fac,\
                           features = _fcopy, \
                           names = _names)
    
    def select_by_indices(self, idxs):

        '''
        Select patches corresponding to the input list of indices.  
        Parameters
        ----------
        idxs : list  
            list of integers as indices.  
        '''
        
        return Patches(self.vol_shape, initialize_by = "data", \
                       points = self.points[idxs].copy(),\
                       widths = self.widths[idxs].copy(),\
                       features = None if self.features is None else self.features[idxs].copy(), \
                       names =  self.feature_names if any(self.feature_names) else [])
        
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
    
    def select_by_plane(self, plane_axis, plane_idx):
        '''
        Select all patches that include a given plane as defined by plane_axis (0, 1 or 2) and plane_idx (index along axis dimension).  
        
        Parameters
        ----------
        plane_axis : int  
            plane along which axis  
        plane_idx : int  
            plane at which index (along given axis)  
        '''
    
        condlist = np.zeros((len(self.points), 2))
        condlist[:,0] = plane_idx > self.points[:, plane_axis] # plane_idx is greater than the min corner point
        condlist[:,1] = plane_idx < self.points[:, plane_axis] + self.widths[:, plane_axis] # plane_idx is smaller than the max corner pt
        
        return self.filter_by_condition(condlist)
    
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
        
        return bin_vals[:,0]
    
    def extract(self, vol, patch_size):

        '''  
        Returns a list of volume patches at the active list of coordinates by drawing from the given big volume 'vol'  
        
        Returns
        -------
        np.ndarray  
            shape is (n_pts, patch_z, patch_y, patch_x)  
        
        '''  
        xp = cp.get_array_module(vol)
        
        _ndim = len(self.vol_shape)
        assert vol.shape == self.vol_shape, "Shape of big volume does not match vol_shape attribute of patches data"

        
        if patch_size is not None:
            # calculate binning
            bin_vals = self._calc_binning(patch_size)
            # make a list of slices
            s = self.slices(binning = bin_vals)
        else:
            s = self.slices()
        
        # make a list of patches
        if _ndim == 3:
            sub_vols = [xp.asarray(vol[s[ii,0], s[ii,1], s[ii,2]]) for ii in range(len(self.points))]
        elif _ndim == 2:
            sub_vols = [xp.asarray(vol[s[ii,0], s[ii,1]]) for ii in range(len(self.points))]
        
        if patch_size is None:
            return sub_vols
        else:
            return xp.asarray(sub_vols, dtype = vol.dtype)
    
    
    def fill_patches_in_volume(self, sub_vols, vol_out, TIMEIT = False):
        
        '''
        to-do: Test cases: a volume full of ones may be assigned a list of sub_vols of zeroes 
        then extract patches back from the vol, and assert equal to sub_vols from before
        '''
        
        t0 = time.time()
        
        # to-do: check if these assertions slow things down?
        assert len(self) == len(sub_vols), "number of sub-volumes do not match the number of items in patches"
        
        assert self.vol_shape == vol_out.shape, "shape of volume enclosing the patches does not match that of the output volume"
        for ii in range(len(self)):
            assert tuple(self.widths[ii,...]) == tuple(sub_vols[ii].shape), "width mismatch between sub_vol and patch at ii = %i"%ii
            assert sub_vols[ii].ndim == 3, "sub_vols must be a 4-D array or a list of volumes of some shape. (batch_size, nz, ny, nx)"        
        s = self.slices()
        for idx in range(len(self)):
                vol_out[tuple(s[idx,...])] = sub_vols[idx]
        t1 = time.time()
        t_tot = (t1-t0)*1000.0/len(self)
        if TIMEIT:
            print("TIME PER UNIT PATCH fill_patches_in_volume: %.2f ms"%t_tot)
        return
        
        # should I return the volume?    
        
    
    def plot_3D_feature(self, ife, ax, plot_type = 'centers'):

        if len(self.vol_shape) != 3:
            raise NotImplementedError("implemented only for 3D patches")
        
        if plot_type == 'centers':
            ax.scatter(self.centers()[:,0], self.centers()[:,1], self.centers()[:,2], c = self.features[:,ife])
        elif plot_type == 'corners':
            ax.scatter(self.points[:,0], self.points[:,1], self.points[:,2], c = self.features[:,ife])

        ax.set_xlim3d(0, self.vol_shape[0])
        ax.set_ylim3d(0, self.vol_shape[1])
        ax.set_zlim3d(0, self.vol_shape[2])  
        if self.feature_names is not None:
            ax.set_title(self.feature_names[ife], fontsize = 16)
        return    
    
    def upsample_patches(self, sub_vols, upsampling_fac):
        raise NotImplementedError("not ")
        assert sub_vols.ndim == 4, "sub_vols array must have shape (batch_size, width_z, width_y, width_x) and ndim == 4 (no channel axis)"

        # can we use some cupy function here? to-do.
        sub_vols = UpSampling3D(upsampling_fac)(sub_vols)
        return sub_vols, self.rescale(upsampling_fac, new_vol_shape = new_vol_shape)

        
        
#     ######## CONSIDERING REMOVING THE CODE BELOW ###############
#     def stitch(self, sub_vols, patch_size, upsample = False):
#         '''  
#         Stitches the big volume from a list of volume patches (with upsampling).    
        
#         Returns
#         -------
#         np.ndarray  
#             shape is (n_pts, patch_z, patch_y, patch_x)  
        
#         '''  
#         _ndim = len(self.vol_shape)
#         if sub_vols.shape[0] != len(self.points):
#             raise ValueError("number of patch points and length of input list of patches must match")
#         vol = np.zeros(self.vol_shape, dtype = sub_vols.dtype)
        
#         # calculate binning
#         bin_vals = self._calc_binning(patch_size)
#         # make a list of slices
#         s = self.slices()
        
#         if _ndim == 3:
#             # set gpu for upsampling
#             gpus = tf.config.experimental.list_physical_devices('GPU')
#             if gpus:
#                 try:
#                     mem_limit = 4*np.prod(patch_size)*np.max(bin_vals)**3*1.5/1.0e6
#                     tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem_limit)])
#                 except RuntimeError as e:
#                     print(e)        
            
#             for ii in range(len(self.points)):
#                 vol_out = sub_vols[ii].copy()[np.newaxis,...,np.newaxis]
#                 vol_out = UpSampling3D(size = tuple([bin_vals[ii]]*3))(vol_out)
#                 vol_out = vol_out[0,...,0]
#                 vol[s[ii,0],s[ii,1],s[ii,2]] = vol_out

#         elif _ndim == 2:
#             for ii in range(len(self.points)):
#                 vol_out = sub_vols[ii].copy()[np.newaxis,...,np.newaxis]
#                 vol_out = cv2.resize(vol_out, (patch_size[1], patch_size[0]))
#                 vol_out = vol_out[0,...,0]
#                 vol[s[ii,0],s[ii,1]] = vol_out
            
#         return vol

        
        
if __name__ == "__main__":
    
    print('just a bunch of functions')
    
