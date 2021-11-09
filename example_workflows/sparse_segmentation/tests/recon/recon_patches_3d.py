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
from tomo_encoders.tasks.sparse_segmenter.recon import recon_patches_3d

if __name__ == "__main__":

#     read_fpath = os.path.join(projs_path, filename + '_projs_bin%i_ntheta%i.hdf5'%(binning,ntheta))
    with h5py.File(read_fpath, 'r') as hf:
        projs = np.asarray(hf['data'][:])
        theta = np.asarray(hf['theta'][:])
        center = float(np.asarray(hf['center'][()]))    

    vol_shape = projs.shape[1:] + (projs.shape[-1],)
    print("projections shape: ", projs.shape)
    print("reconstructed volume shape: ", vol_shape)
    
    # generate some random patches
    kwargs = {"initialize_by" : 'regular-grid',
              "patch_size" : model_size, \
              "stride" : 1}
    p_grid = Patches(vol_shape, **kwargs)
    print("total patches: ", len(p_grid))
    p_grid._check_valid_points()
    p3d = p_grid.select_random_sample(500)
#     p3d = p_grid.copy()
    print("widths: ", str(p3d.widths[:1]))    

    with cp.cuda.Device(0):
        center = projs.shape[-1]//2.0
        sub_vols, p3d_new = recon_patches_3d(projs, theta, center, p3d, \
                                             mem_limit_gpu = 40.0, \
                                             apply_fbp = True, \
                                             TIMEIT = True)
        cp.cuda.Device(0).synchronize()
    
    _old = np.lexsort(p3d.points[:,::-1].T)
    _new = np.lexsort(p3d_new.points[:,::-1].T)
    
    assert len(p3d) == len(p3d_new), "input and output patches length doesn't match!!"
    for ii in range(len(_old)):
        assert tuple(p3d.points[_old[ii],:]) == tuple(p3d_new.points[_new[ii]]), "patch don't match at ii = %i!!"%ii

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
