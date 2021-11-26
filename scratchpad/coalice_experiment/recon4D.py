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
from scipy.ndimage.filters import median_filter

from multiprocessing import Pool, cpu_count
import functools
import numpy as np
import h5py
import abc
from tomo_encoders.reconstruction.recon import darkflat_correction
from tomo_encoders.reconstruction.solver import solver


class AnyProjectionStream(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def reconstruct_window(self):
        pass
    
class SomeProjectionStream(AnyProjectionStream):
    def __init__(self, fname, fname_flat, fname_dark, NTHETA_180, EXPOSURE_TIME_PER_PROJ):
        '''
        Read, reconstruct and segment data from a 4D dataset.  
        
        Parameters
        ----------
        fname : str
            Name of the file containing projection data.  
        fname_flat : str  
            Name of the file containing all flat fields (in exchange/data)  
        fname_dark : str  
            Name of the file containing all dark fields (in exchange/data)  
        
        '''
        self.fname = fname
        self.fname_flat = fname_flat
        self.fname_dark = fname_dark
        self.NTHETA_180 = NTHETA_180
        self.EXPOSURE_TIME_PER_PROJ = EXPOSURE_TIME_PER_PROJ
    
        self._read_flat_field()
        self._read_proj_stats()
        print("Shape of projection image: %s"%str(self.flat.shape))
        return

    def _read_proj_stats(self):
        with h5py.File(self.fname, 'r') as hf:        
            self.total_projs, self.nrows, self.ncols = hf['exchange/data'].shape
            self.theta_all = np.radians(np.asarray(hf['exchange/theta'][:]))
            self.time_exposed_all = np.arange(self.total_projs)*self.EXPOSURE_TIME_PER_PROJ
        
        # find center
        self.center = 471.0
        return
            
    def _read_flat_field(self):
        with h5py.File(self.fname_flat, 'r') as hf:
            self.flat = np.mean(hf['exchange/data_white'][:], axis = 0)
        with h5py.File(self.fname_dark, 'r') as hf:
            self.dark = np.mean(hf['exchange/data_dark'][:], axis = 0)        
        return
        
    def _get_projections_180deg(self, istart, vert_slice = None):
        s = slice(istart, istart + self.NTHETA_180)
        if vert_slice is None:
            vert_slice = slice(None,None,None)
        
        with h5py.File(self.fname, 'r') as hf:
            theta = np.radians(np.asarray(hf['exchange/theta'][s,...]), dtype = np.float32)
            proj = np.asarray(hf['exchange/data'][s,vert_slice,...], dtype = np.float32)        
            
        if theta.size != self.NTHETA_180:
            raise ValueError("full 180 not available for given istart")
        else:
            return proj, theta
        
    def reconstruct_window(self, time_elapsed, mask_ratio = 0.95, contrast_s = 0.01, vert_slice = None):

        '''
        reconstruct over a sliding window starting at index istart and ending at the opposing angle (180 spin).  
        
        Parameters
        ----------
        time_elapsed : float  
            exposure time elapsed (seconds)  
        rot_center : int  
            rotation center;  
            
        Returns
        -------
        np.ndarray
            three dimensional array  
        
        '''
        
        istart = np.argmin(np.abs(self.time_exposed_all - time_elapsed))
        projs, theta = self._get_projections_180deg(istart, vert_slice = vert_slice)
        
        
        vol_rec = solver(projs, theta, self.center, self.dark, self.flat, \
                         contrast_adjust_factor = contrast_s, \
                         mask_ratio = mask_ratio, \
                         ## DEBUG ONLY \
                         apply_darkflat_correction = True, \
                         apply_fbp = True, \
                         apply_minus_log = True, TIMEIT = False)
        return vol_rec

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use('Agg')
    from config import *
    
    import pdb; pdb.set_trace()
    
    dget = DataGetter(*fnames)
    imgs = []
    t_vals = []
    
######### FOR MAKING A VIDEO ###############    
#     center_val = self.center
#     from tqdm import trange
#     for idx in trange(200):
#         true_idx = 90*idx
#         img_t = dget.reconstruct_window(true_idx,center_val, **recon_params)[0]
#         imgs.append(img_t)
#         t_vals.append(true_idx*delta_t)

#     save_plot_path = '/home/atekawade/Dropbox/Arg/transfers/plots_aisteer/fullfield_video'
    
    
#     for ii in trange(len(imgs)):
#         fig, ax = plt.subplots(1,1)
#         ax.imshow(imgs[ii], cmap = 'gray')
#         ax.axis('off')
#         ax.text(30,50,'t = %2.0f secs'%t_vals[ii])
#         plt.savefig(os.path.join(save_plot_path, 'plot%02d.png'%ii)) 
#         plt.close()

#     save_plot_path = '/home/atekawade/Dropbox/Arg/transfers/plots_aisteer/zoomed_video'
#     for ii in trange(len(imgs)):
#         scrop = slice(400,600,None)
#         fig, ax = plt.subplots(1,1)
#         ax.imshow(imgs[ii][scrop, scrop], cmap = 'gray')
#         ax.axis('off')
#         ax.text(15,25,'t = %2.0f secs'%t_vals[ii])
#         plt.savefig(os.path.join(save_plot_path, 'plot%02d.png'%ii)) 
#         plt.close()
        
        
        