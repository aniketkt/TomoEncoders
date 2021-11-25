#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import sys 
import matplotlib.pyplot as plt 
import numpy as np 
import cupy as cp 
import time 
import os 
import tensorflow as tf
import vedo


from tomo_encoders.reconstruction.recon import recon_patches_3d, recon_binning
from tomo_encoders.labeling.detect_voids import export_voids
from tomo_encoders.misc.voxel_processing import modified_autocontrast, cylindrical_mask
from tomo_encoders import Patches

class VoidMetrology():
    '''
    '''
    def __init__(self, THETA_BINNING, DET_BINNING, \
                 INF_INPUT_SIZE, INF_CHUNK_SIZE,\
                 N_MAX_DETECT, CIRC_MASK_FRAC,
                 TIMEIT_lev1 = True, TIMEIT_lev2 = False, \
                 NORM_SAMPLING_FACTOR = 4):
        '''
        '''
        self.THETA_BINNING = THETA_BINNING
        self.DET_BINNING = DET_BINNING
        self.INF_INPUT_SIZE = INF_INPUT_SIZE
        self.INF_CHUNK_SIZE = INF_CHUNK_SIZE
        self.N_MAX_DETECT = N_MAX_DETECT
        self.CIRC_MASK_FRAC = CIRC_MASK_FRAC
        
        self.TIMEIT_lev1 = TIMEIT_lev1
        self.TIMEIT_lev2 = TIMEIT_lev2
        self.NORM_SAMPLING_FACTOR = NORM_SAMPLING_FACTOR
        return
    
    def reconstruct(self, projs, theta, center):

        # BINNED RECONSTRUCTION
        vol_rec_b = recon_binning(projs, theta, center, \
                                  self.THETA_BINNING, \
                                  self.DET_BINNING, \
                                  self.DET_BINNING, \
                                  apply_fbp = True, \
                                  TIMEIT = self.TIMEIT_lev2)
#         print("vol_rec_b shape: ", vol_rec_b.shape)

        ##### CLIP WITH AUTOCONTRAST #####
        clip_vals = modified_autocontrast(vol_rec_b, s = 0.05, \
                                          normalize_sampling_factor = self.NORM_SAMPLING_FACTOR)
        vol_rec_b = np.clip(vol_rec_b, *clip_vals)

        return vol_rec_b
    
    def reconstruct_patches(self, projs, theta, center, \
                            p3d_grid):
        
        '''
        Selected patches reconstructed at full resolution.  
        '''
        
        sub_vols_grid, p3d_grid = recon_patches_3d(projs, theta, center, p3d_grid, \
                                                   apply_fbp = True, \
                                                   TIMEIT = self.TIMEIT_lev2)
        
#         print("length of sub_vols reconstructed %i"%len(sub_vols_grid_voids_1))

        ##### CLIP WITH AUTOCONTRAST #####
        clip_vals = modified_autocontrast(np.asarray(sub_vols_grid), s = 0.05, \
                                          normalize_sampling_factor = self.NORM_SAMPLING_FACTOR)
        sub_vols_grid = np.clip(sub_vols_grid, *clip_vals)
        
        return sub_vols_grid
    
    
    def segment_patches_to_volume(self, sub_vols_grid, p3d_grid_voids, out_vol, fe):
        '''
        Segment reconstructed patches at full resolution.
        '''
        

        min_val = sub_vols_grid[:,::self.NORM_SAMPLING_FACTOR].min()
        max_val = sub_vols_grid[:,::self.NORM_SAMPLING_FACTOR].max()
        min_max = (min_val, max_val)

        sub_vols_y_pred = fe.predict_patches("segmenter", sub_vols_grid[...,np.newaxis], \
                                               self.INF_CHUNK_SIZE, None, \
                                               min_max = min_max, \
                                               TIMEIT = self.TIMEIT_lev2)
        if self.TIMEIT_lev2: # unpack if time is returned
            sub_vols_y_pred, _ = sub_vols_y_pred
        sub_vols_y_pred = sub_vols_y_pred[...,0]

        
        p3d_grid_voids.fill_patches_in_volume(sub_vols_y_pred, out_vol, TIMEIT = self.TIMEIT_lev2)
        return
    
    def segment(self, vol_rec_b, fe):
    
        # SEGMENTATION OF BINNED RECONSTRUCTION
        p3d_grid_b = Patches(vol_rec_b.shape, \
                             initialize_by = "grid", \
                             patch_size = self.INF_INPUT_SIZE)
        sub_vols_x_b = p3d_grid_b.extract(vol_rec_b, self.INF_INPUT_SIZE)
        min_max = fe.calc_voxel_min_max(vol_rec_b, self.NORM_SAMPLING_FACTOR, TIMEIT = False)
        sub_vols_y_pred_b = fe.predict_patches("segmenter", sub_vols_x_b[...,np.newaxis], \
                                               self.INF_CHUNK_SIZE, None, \
                                               min_max = min_max, \
                                               TIMEIT = self.TIMEIT_lev2)
        if self.TIMEIT_lev2: # unpack if time is returned
            sub_vols_y_pred_b, _ = sub_vols_y_pred_b
            
        sub_vols_y_pred_b = sub_vols_y_pred_b[...,0]
        vol_seg_b = np.zeros(vol_rec_b.shape, dtype = np.uint8)
        
        p3d_grid_b.fill_patches_in_volume(sub_vols_y_pred_b, \
                                          vol_seg_b, TIMEIT = False)
        assert vol_seg_b.dtype == np.uint8, "data type check failed for vol_seg_b"
#         print("vol_seg_b shape: ", vol_seg_b.shape)
        
        cylindrical_mask(vol_seg_b, self.CIRC_MASK_FRAC, mask_val = 0)
        
        return vol_seg_b
    
    def export_voids(self, vol_seg):
        return export_voids(vol_seg, \
                            self.N_MAX_DETECT, \
                            TIMEIT = self.TIMEIT_lev2)  



if __name__ == "__main__":

    
    print('nothing here')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
