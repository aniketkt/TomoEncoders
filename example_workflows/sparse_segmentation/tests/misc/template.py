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


fpath = '/data02/MyArchive/AM_part_Xuan/data/mli_L206_HT_650_L3_rec_1x1_uint16.hdf5' 
projs_path = '/data02/MyArchive/AM_part_Xuan/projs' 

binning = 2 

if __name__ == "__main__":

    ds = DataFile(fpath, data_tag = 'data', tiff = False, VERBOSITY = 0)
    vol = ds.read_full().astype(np.float32)
    
    sbin = tuple([slice(None,None,binning)]*3) 
    vol = vol[sbin].copy()
    
    print(vol.shape)
    
    

    
