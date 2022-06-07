#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import sys 
import matplotlib.pyplot as plt 
import numpy as np 
from tomo_encoders.misc.feature_maps_vis import view_midplanes 
import cupy as cp 
import time 
import h5py 
from tomo_encoders import DataFile, Patches
import os 

import matplotlib as mpl
mpl.use("Agg")

read_fpath = '/data02/MyArchive/AM_part_Xuan/projs/mli_L206_HT_650_L3_projs_bin1_ntheta3000.hdf5'
save_image_path = '/home/atekawade/Dropbox/Arg/transfers/runtime_plots/binned_recon'
model_size = (64,64,64)
from tomo_encoders.reconstruction.recon import recon_binning

theta_binning = 6
z_binning = 4
col_binning = 4


binned_vals = [(8, 4, 4), (4, 4, 4), (4, 4, 8)]

if __name__ == "__main__":

#     read_fpath = os.path.join(projs_path, filename + '_projs_bin%i_ntheta%i.hdf5'%(binning,ntheta))
    with h5py.File(read_fpath, 'r') as hf:
        projs = np.asarray(hf['data'][:])
        theta = np.asarray(hf['theta'][:])
        center = float(np.asarray(hf['center'][()]))    
    vol_shape = projs.shape[1:] + (projs.shape[-1],)
    print("full projections shape: ", projs.shape)
    print("full reconstructed volume shape: ", vol_shape)
    
    for bin_params in binned_vals:
        theta_binning, z_binning, col_binning = bin_params
        print("\nbinning params: %s"%str(bin_params))
        with cp.cuda.Device(0):
            center = projs.shape[-1]//2.0

            vol = recon_binning(projs, theta, center, \
                          theta_binning, \
                          z_binning, \
                          col_binning, \
                          apply_fbp = True, \
                          TIMEIT = True)
            cp.cuda.Device(0).synchronize()
            
            fig, ax = plt.subplots(1,3, figsize = (16,8))
            view_midplanes(vol, ax = ax)
            plt.savefig(save_image_path+"%s.png"%str(bin_params))
    
    
    
    
