#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import sys 
import matplotlib.pyplot as plt 
import numpy as np 
from tomo_encoders.misc_utils.feature_maps_vis import view_midplanes 
import cupy as cp 
import time 
import h5py 
from tomo_encoders import DataFile, Patches
import os 

read_fpath = '/data02/MyArchive/AM_part_Xuan/projs/mli_L206_HT_650_L3_projs_bin2_ntheta1500.hdf5'
model_size = (64,64,64)
from tomo_encoders.tasks.sparse_segmenter.recon import recon_patches_3d, recon_chunk
z_chunk = model_size[0]
sz = slice(400,400+z_chunk)

if __name__ == "__main__":

    print("#"*55, "\nReconstruct a z-chunk of fixed height %i and increasing number of 2d patches inside its cross-section\n"%z_chunk, "#"*55)
    
    
    with h5py.File(read_fpath, 'r') as hf:
        projs = np.asarray(hf['data'][:,sz,:])
        theta = np.asarray(hf['theta'][:])
        center = float(np.asarray(hf['center'][()]))    

    vol_shape = projs.shape[1:] + (projs.shape[-1],)
    print("projections shape: ", projs.shape)
    print("reconstructed volume shape: ", vol_shape)
    
    # generate some random patches
    kwargs = {"initialize_by" : 'regular-grid',
              "patch_size" : model_size[1:], \
              "stride" : 1}
    p2d = Patches(vol_shape[1:], **kwargs)
    print("total patches: ", len(p2d))
    p2d._check_valid_points()
    
    print("widths: ", p2d.widths[:1])    

    iter_list = np.linspace(1, len(p2d), 10).astype(np.uint32)
    
    for ic, plen in enumerate(iter_list):
        with cp.cuda.Device(0):
            p2d_sample = p2d.select_random_sample(plen)
            print("#"*30, "\n", "plen %i\n"%plen)
            sub_vols = recon_chunk(projs, theta, center, p2d_sample, apply_fbp = True, TIMEIT = True)
            cp.cuda.Device(0).synchronize()
        
    
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
