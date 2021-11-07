#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import sys 
import matplotlib.pyplot as plt 
import numpy as np 
# from tomo_encoders.misc_utils.feature_maps_vis import view_midplanes 
import cupy as cp 
import time 
import h5py 
#from recon_subvol import fbp_filter, recon_patch 
# from tomo_encoders import DataFile
import os 


fpath = '/data02/MyArchive/AM_part_Xuan/data/mli_L206_HT_650_L3_rec_1x1_uint16.hdf5' 

binning = 1

def _rescale_data(data, min_val, max_val):
    '''
    Recales data to values into range [min_val, max_val]. Data can be any numpy or cupy array of any shape.  

    '''
    xp = cp.get_array_module(data)  # 'xp' is a standard usage in the community
    eps = 1e-12
    data = (data - min_val) / (max_val - min_val + eps)
    return data


def _find_min_max(vol, sampling_factor):

    ss = slice(None, None, sampling_factor)
    xp = cp.get_array_module(vol[ss,ss,ss])  # 'xp' is a standard usage in the community
    max_val = xp.max(vol[ss,ss,ss])
    min_val = xp.min(vol[ss,ss,ss])
    return max_val, min_val

def normalize_volume_gpu(vol, chunk_size = 64, normalize_sampling_factor = 1):
    '''
    Normalizes volume to values into range [0,1]  

    '''

    tot_len = vol.shape[0]
    nchunks = int(np.ceil(tot_len/chunk_size))
    max_val, min_val = _find_min_max(vol, normalize_sampling_factor)
    
    proc_times = []
    copy_to_times = []
    copy_from_times = []
    stream1 = cp.cuda.Stream()
    t0 = time.time()
    
    vol_gpu = cp.zeros((chunk_size, vol.shape[1], vol.shape[2]), dtype = cp.float32)
    for jj in range(nchunks):
        t01 = time.time()
        sz = slice(jj*chunk_size, min((jj+1)*chunk_size, tot_len))
        
        
        ## copy to gpu from cpu
        with stream1:    
            vol_gpu.set(vol[sz,...])
        stream1.synchronize()    
        t02 = time.time()
        copy_to_times.append(t02-t01)
        
        ## process
        with stream1:
             vol_gpu = _rescale_data(vol_gpu, min_val, max_val)
        stream1.synchronize()
        t03 = time.time()
        proc_times.append(t03-t02)
        
        ## copy from gpu to cpu
        with stream1:
            vol[sz,...] = vol_gpu.get()            
        stream1.synchronize()    
        t04 = time.time()
        copy_from_times.append(t04 - t03)
    
    print("copy to gpu time per %i size chunk: %.2f ms"%(chunk_size,np.mean(copy_to_times)*1000.0))
    print("processing time per %i size chunk: %.2f ms"%(chunk_size,np.mean(proc_times)*1000.0))
    print("copy from gpu time per %i size chunk: %.2f ms"%(chunk_size,np.mean(copy_from_times)*1000.0))
    print("total time: ", time.time() - t0)
    return vol


if len(sys.argv) > 1:
    chunk_size = int(sys.argv[1])
else:
    chunk_size = 64
    
if __name__ == "__main__":

    vol_shape = (512,1224,1224)
    vol = np.random.normal(0.0, 1.0, vol_shape).astype(np.float32)
    print("input volume: ", vol.shape)
    
    vol = normalize_volume_gpu(vol, chunk_size = chunk_size, normalize_sampling_factor = 4)
    
    

    
    

    
