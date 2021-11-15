#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 

from scipy.ndimage import label as label_np
from scipy.ndimage import find_objects
from cupyx.scipy.ndimage import zoom as zoom_cp
import cupy as cp
from scipy.ndimage import zoom as zoom_np
# from tensorflow.keras.layers import UpSampling3D
# import tensorflow as tf


import time
from tomo_encoders import Patches
import numpy as np

def calc_patch_size(base_size, multiplier):
    
    output = np.asarray(base_size)*multiplier
    output = np.round(output).astype(np.uint32)
    return tuple(output)


def wrapper_label(vol_seg, n_max_detect, TIMEIT = False, N_VOIDS_IGNORE = 2):
    '''
    takes in a big volume with zeros indicating voids and ones indicating material.  
    outputs labeled array with zero indicating material and labels [1, n_max_detect] indicating different voids.  
    if n_max_detect is smaller than the total number detected, the smaller voids are ignored and assigned values of background ( = 0)
    finally, array dtype is adjusted based on number of voids. e.g. if n_detected < 2**8 - 1, then array is uint8, etc.
    '''
    
    assert vol_seg.dtype == 'uint8', "vol_seg must be uint8"
    
    t0 = time.time()
    print("detecting all particles up to max possible integer precision")
    vol_lab, n_detected,  = label_np(vol_seg^1)
    s_voids = find_objects(vol_lab)
    print("found %i"%n_detected)
    
    # DEBUG ONLY
    assert len(s_voids) == n_detected, "find_objects test failed: detected number of slice objects does not match output from label"    
    
    print("finding objects and sorting by size (decreasing order)")
    p3d_voids = Patches(vol_lab.shape, initialize_by="slices", s = s_voids)
    # create feature array - ["void_id", "void_size"]
    void_features = np.zeros((len(s_voids), 2))
    for ip in range(len(s_voids)):
        void_id = ip + 1
        sub_vol = (vol_lab[s_voids[ip]] == void_id).astype(np.uint8)
        void_size = np.cbrt(np.sum(sub_vol))
        void_features[ip, 0] = void_id
        void_features[ip, 1] = void_size
    p3d_voids.add_features(void_features, names = ["void_id", "void_size"])    
    
    del s_voids
    # filter by size, "n" largest voids: hence ife = 0
    p3d_voids = p3d_voids.select_by_feature(n_max_detect, ife = 1, selection_by = "highest")
    
    p3d_voids = p3d_voids.select_by_feature(len(p3d_voids)-N_VOIDS_IGNORE, \
                                           ife = 1, \
                                           selection_by = "lowest")
    p3d_voids = p3d_voids.select_by_feature(len(p3d_voids), \
                                            ife = 1, \
                                            selection_by = "highest")    
    
    s_voids = p3d_voids.slices()
    
    # sub_vols_voids_b should contain only those voids that the sub-volume is pointing to.
    # Need to ignore others occuring in the sub_vol by coincidence.
    sub_vols_voids = []
    for ip in range(len(p3d_voids)):
        sub_vols_voids.append((vol_lab[tuple(s_voids[ip,:])] == p3d_voids.features[ip,0]).astype(np.uint8))

    t1 = time.time()
    t_tot = t1 - t0
    if TIMEIT:
        print("TIME for counting voids: %.2f seconds"%t_tot)
        
    return sub_vols_voids, p3d_voids

def to_regular_grid(sub_vols, p3d, target_patch_size, target_vol_shape, upsample_fac):
    '''
    (1) select patches where voids exist (2) rescale to upsample_value
    '''
    # make vol_seg_b
    vol = np.empty(p3d.vol_shape, dtype = sub_vols[0].dtype)
    p3d.fill_patches_in_volume(sub_vols, vol)    
    assert vol.max() == 1, "vol is not binary, which is required for the selection criteria"

    
    # make grid on binned volume
    binned_patch_size = calc_patch_size(target_patch_size, 1.0/upsample_fac)
    p3d_grid = Patches(p3d.vol_shape, initialize_by="regular-grid", patch_size = binned_patch_size)
    
    # select patches containing voids (voids are considered particles here (== 1))
    y = p3d_grid.extract(vol, binned_patch_size)
    contains_voids = np.sum(y, axis = (1,2,3)) > 0
    p3d_grid = p3d_grid.filter_by_condition(contains_voids)
    
    # upsample back
    p3d_grid = p3d_grid.rescale(upsample_fac, target_vol_shape)
    return p3d_grid


def upsample_sub_vols(sub_vols, upsampling_fac, TIMEIT = False, order = 1):
    
    '''
    all sub-volumes must have same shape.
    upsampling_factor applies to all 3 dimensions equally.
    
    to-do: split into chunks, and try higher-order 
    
    '''
    assert sub_vols.ndim == 4, "sub_vols array must have shape (batch_size, width_z, width_y, width_x) and ndim == 4 (no channel axis)"
    
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
   
    device = cp.cuda.Device()
    memory_pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(memory_pool.malloc)
    
    
    new_sub_vol_size = tuple([len(sub_vols)] + [sub_vols.shape[1+i]*upsampling_fac for i in range(3)])
    sub_vols_up = cp.empty(new_sub_vol_size, dtype = sub_vols.dtype)    
    s1 = cp.cuda.Stream()
    with s1:
        sub_vols_up = zoom_cp(cp.array(sub_vols), \
                              (1,) + tuple(3*[upsampling_fac]), \
                              output = sub_vols_up, \
                              order = order).get()
        s1.synchronize()
    
    device.synchronize()

    end_gpu.record()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    
    if TIMEIT:
        print("TIME upsampling: %.2f seconds"%(t_gpu/1000.0))
    
    print('total bytes: ', memory_pool.total_bytes())    
    
    return sub_vols_up

if __name__ == "__main__":

    print("hello world")
    
    

    
