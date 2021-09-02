#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class implementations for real-time 3D feature extraction


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

class AnyDataGetter(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def reconstruct_window(self):
        pass
    
class DataGetter(AnyDataGetter):
    def __init__(self, fname, fname_flat, fname_dark, nproj):
        '''
        Read, reconstruct and segment data from a 4D dataset.  
        
        Parameters
        ----------
        nproj : int
            Number of projections in 180 a degree spin.  
        fname : str
            Name of the file containing projection data.  
        fname_flat : str  
            Name of the file containing all flat fields (in exchange/data)  
        fname_dark : str  
            Name of the file containing all dark fields (in exchange/data)  
        
        '''
        self.nproj = nproj
        self.fname = fname
        self.fname_flat = fname_flat
        self.fname_dark = fname_dark
    
        self._read_flat_field()
        self._read_proj_stats()
        print("Shape of projection image: %s"%str(self.flat.shape))
        return

    def _read_proj_stats(self):
        with h5py.File(self.fname, 'r') as hf:        
            self.total_projs, self.nrows, self.ncols = hf['exchange/data'].shape
            self.theta_all = np.asarray(hf['exchange/theta'][:])
    
    def _read_flat_field(self):
        
        with h5py.File(self.fname_flat, 'r') as hf:
            self.flat = np.mean(hf['exchange/data_white'][:], axis = 0)
            self.flat = self.flat[np.newaxis,...]
        with h5py.File(self.fname_dark, 'r') as hf:
            self.dark = np.mean(hf['exchange/data_dark'][:], axis = 0)        
            self.dark = self.dark[np.newaxis,...]
        return
        
    def _get_projections_180deg(self, istart):
        s = slice(istart, istart + self.nproj)
        with h5py.File(self.fname, 'r') as hf:
            theta = np.asarray(hf['exchange/theta'][s,...])
            proj = np.asarray(hf['exchange/data'][s,...])        
            
        if theta.size != self.nproj:
            raise ValueError("full 180 not available for given istart")
        else:
            proj = normalize(proj, self.flat, self.dark)
            return proj, theta
        
    def _get_projections_opposing(self, istart, blur_kernel = 3):
        
        proj = [0]*2
        with h5py.File(self.fname, 'r') as hf:
            proj[0] = np.asarray(hf['exchange/data'][istart,...])        
            proj[1] = np.asarray(hf['exchange/data'][istart+self.nproj,...])
        
        proj = normalize(proj, self.flat, self.dark)
        proj = np.asarray([median_filter(p, size = (blur_kernel,blur_kernel)) for p in proj])
        return proj
        
    
    
    def find_center(self, istart, center_guess = None, search_width = 50, roi_width = 0.8):
        '''
        find the rotational center at a sliding window index 'istart'.  
        
        Parameters  
        ----------  
        istart : int  
            starting index  
        center_guess : int  
            starting guess, if None, then starts at center of column axis    
        search_width : int  
            width around each side of guess, e.g. guess = 100 and width 50, searches 50 to 150  
        roi_width : int  
            region of interest width (fraction of total width)  
        
        '''
        proj = self._get_projections_opposing(istart)
        
        if center_guess is None:
            center_guess = proj.shape[-1]//2
        
        search_range = (center_guess - search_width, center_guess + search_width)

        center_val = estimate_center(proj, search_range, roi_width = roi_width, metric = "NCC", procs = cpu_count()//4)
        print("center = %.2f"%(center_val))
        return center_val
    
    
    def reconstruct_window(self, istart, rot_center, mask_ratio = 0.95, contrast_s = 0.01):

        '''
        reconstruct over a sliding window starting at index istart and ending at the opposing angle (180 spin).  
        
        Parameters
        ----------
        istart : int  
            starting index  
        rot_center : int  
            rotation center;  
            
        Returns
        -------
        np.ndarray
            three dimensional array  
        
        '''
        
        proj, theta = self._get_projections_180deg(istart)
        
        if theta[-1] - theta[0] != 180.0:
            raise ValueError("theta values don't seem right")
        else:
            theta = angles(proj.shape[0]) # or theta = theta.size
            # TO-DO DO NOT GENERATE ANGLES, READ THEM FROM STREAM
        
        proj = minus_log(proj)
        
        
        # gridrec padding  (from tomopy-cli)  
        N = proj.shape[2]
        proj_pad = np.zeros([proj.shape[0],proj.shape[1],3*N//2],dtype = "float32")
        proj_pad[:,:,N//4:5*N//4] = proj
        proj_pad[:,:,0:N//4] = np.reshape(proj[:,:,0],[proj.shape[0],proj.shape[1],1])
        proj_pad[:,:,5*N//4:] = np.reshape(proj[:,:,-1],[proj.shape[0],proj.shape[1],1])
        proj = proj_pad
        rot_center = rot_center + N//4
        
        rec = recon(proj, theta = theta, \
                      center = rot_center, \
                      algorithm = 'gridrec', \
                      sinogram_order = False, \
                    filter_name = 'parzen')
        
        rec = rec[:,N//4:5*N//4,N//4:5*N//4]
        
        if mask_ratio is not None:
            rec = circ_mask(rec, 0, ratio = mask_ratio)
        
        if contrast_s > 0.0:
            h = modified_autocontrast(rec, s = contrast_s)
            rec = np.clip(rec, *h)
        
        return rec
    

    
# for center finding    
def match_opposing(center_guess, proj = None, roi_width = None, metric = 'NCC'):
    
    '''
    translates projection at theta = 0
    '''
    
    if roi_width is None:
        roi_width = 0.8
    center_guess = int(center_guess)
    
    sx = slice(center_guess - int(roi_width*proj.shape[-1]//2), center_guess + int(roi_width*proj.shape[-1]//2))
    
    if metric == "NCC":
        match_val = match_template(proj[0, :, sx], \
                                   np.fliplr(proj[-1, :, sx]))[0][0]
    elif metric == "MSE":
        match_val = np.linalg.norm(proj[0, : , sx] - np.fliplr(proj[-1, :, sx]), ord = 2)
    
    
    return match_val

def estimate_center(proj, search_range, roi_width = 0.8, metric = "NCC", procs = 12):
    
    center_guesses = np.arange(*search_range).tolist()
    match_vals = Parallelize(center_guesses, match_opposing, proj = proj, roi_width = roi_width, metric = metric, procs = procs)
    match_vals = np.asarray(match_vals)
    return center_guesses[np.argmax(match_vals)]

def Parallelize(ListIn, f, procs = -1, **kwargs):
    
    """This function packages the "starmap" function in multiprocessing, to allow multiple iterable inputs for the parallelized function.  
    
    Parameters
    ----------
    ListIn: list
        each item in the list is a tuple of non-keyworded arguments for f.  
    f : func
        function to be parallelized. Signature must not contain any other non-keyworded arguments other than those passed as iterables.  
    
    Example:  
    
    .. highlight:: python  
    .. code-block:: python  
    
        def multiply(x, y, factor = 1.0):
            return factor*x*y
    
        X = np.linspace(0,1,1000)  
        Y = np.linspace(1,2,1000)  
        XY = [ (x, Y[i]) for i, x in enumerate(X)] # List of tuples  
        Z = Parallelize_MultiIn(XY, multiply, factor = 3.0, procs = 8)  
    
    Create as many positional arguments as required, but all must be packed into a list of tuples.
    
    """
    if type(ListIn[0]) != tuple:
        ListIn = [(ListIn[i],) for i in range(len(ListIn))]
    
    reduced_argfunc = functools.partial(f, **kwargs)
    
    if procs == -1:
        opt_procs = int(np.interp(len(ListIn), [1,100,500,1000,3000,5000,10000] ,[1,2,4,8,12,36,48]))
        procs = min(opt_procs, cpu_count())

    if procs == 1:
        OutList = [reduced_argfunc(*ListIn[iS]) for iS in range(len(ListIn))]
    else:
        p = Pool(processes = procs)
        OutList = p.starmap(reduced_argfunc, ListIn)
        p.close()
        p.join()
    
    return OutList

def modified_autocontrast(vol, s = 0.01):
    
    '''
    Returns
    -------
    tuple
        alow, ahigh values to clamp data  
    
    Parameters
    ----------
    s : float
        quantile of image data to saturate. E.g. s = 0.01 means saturate the lowest 1% and highest 1% pixels
    
    '''
    
    data_type  = np.asarray(vol).dtype
    
    
    if type(s) == tuple and len(s) == 2:
        slow, shigh = s
    else:
        slow = s
        shigh = s

    h, bins = np.histogram(vol, bins = 500)
    c = np.cumsum(h)
    c_norm = c/np.max(c)
    
    ibin_low = np.argmin(np.abs(c_norm - slow))
    ibin_high = np.argmin(np.abs(c_norm - 1 + shigh))
    
    alow = bins[ibin_low]
    ahigh = bins[ibin_high]
    
    return alow, ahigh




if __name__ == "__main__":
    
    print('just a bunch of functions')
    
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-f', "--input-fname", required = True, type = str, help = "Path to tiff folder or hdf5 file")
#     parser.add_argument('-i', "--stats_only", required = False, action = "store_true", default = False, help = "show stats only")
#     parser.add_argument('-v', "--verbosity", required = False, type = int, default = 0, help = "read / write verbosity; 0 - silent, 1 - important stuff, 2 - print everything")
#     parser.add_argument('-g', "--center_guess", required = False, type = int, default = 0, help = "initial guess is horizontal center of projection if not provided")
#     parser.add_argument('--search_width', required = False, type = int, default = 50, help = "search width around guessed center, e.g. default width of 50 and guess of 600 will search on range 550 - 650")
#     parser.add_argument('--roi_width', required = False, type = float, default = 0.8, help = "fraction of horizontal roi to use for cross correlation match")
#     parser.add_argument('--metric', required = False, type = str, default = "NCC", help = "metric can be MSE or NCC. NCC is preferrred")
#     parser.add_argument('--procs', required = False, type = int, default = 12, help = "number of processes to spawn for multiprocessing")
#     args = parser.parse_args()

#     if args.stats_only:
#         print("not implemented")
#     else:
#         main(args)
 
