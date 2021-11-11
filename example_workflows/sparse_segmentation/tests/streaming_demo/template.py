#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import sys 
import matplotlib.pyplot as plt 
import numpy as np 
from porespy import generators 
from tomo_encoders.misc_utils.feature_maps_vis import view_midplanes 
import cupy as cp 
import time 
import h5py 
from tomopy import project 
#from recon_subvol import fbp_filter, recon_patch 
from tomo_encoders import DataFile
import os 
import signal 
import tomocg as pt



projs_path = '/data02/MyArchive/AM_part_Xuan/projs' 

########## SAMPLE CODE ###########

# # sub_vols shape is (batch_size, nz, ny, nx) so use newaxis to get to (batch_size, nz, ny, nx, 1)
# y_pred = self.predict_patches(sub_vols[...,np.newaxis], \
#                               chunk_size, out_arr, \
#                               min_max = min_max, \
#                               TIMEIT = False)[...,0]

# self.fill_patches_in_volume(y_pred, p, vol_out)



if __name__ == "__main__":

    
    print('hello world')

    
