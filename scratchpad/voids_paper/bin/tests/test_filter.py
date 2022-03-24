#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import sys
import time
import seaborn as sns
import pandas as pd

import cupy as cp
from tomo_encoders import Patches
from tomo_encoders.misc import viewer
from tomo_encoders import DataFile
from tomo_encoders.reconstruction.recon import fbp_filter
# from tomo_encoders.misc.voxel_processing import cylindrical_mask, normalize_volume_gpu
# from cupy.fft import rfft, irfft, rfftfreq
from cupyx.scipy.fft import rfft, irfft, rfftfreq

nz = 32
n = 2176
ntheta = 1500


n_pad = n*(1 + 0.25*2) # 1/4 padding
n_pad = int(np.ceil(n_pad/8.0)*8.0) 
pad_left = int((n_pad - n)//2)
pad_right = n_pad - n - pad_left        


if __name__ == "__main__":

    
    # arguments to recon_chunk2: data, theta, center, p3d
    data = cp.random.normal(0,1,(ntheta, nz, n)).astype(np.float32)
    data_padded = cp.empty((ntheta,nz,n_pad), dtype = cp.float32)


    start_gpu = cp.cuda.Event(); end_gpu = cp.cuda.Event(); start_gpu.record()
    stream = cp.cuda.Stream()
    with stream:
        data_padded.put(cp.arange(ntheta*nz*n_pad),cp.pad(data, ((0,0),(0,0),(pad_left, pad_right)), mode = 'edge'))
        stream.synchronize()
    end_gpu.record(); end_gpu.synchronize(); t_meas = cp.cuda.get_elapsed_time(start_gpu,end_gpu)
    print(f"overhead for making padded array: {t_meas:.2f} ms")        



    for ii in range(100):
        start_gpu = cp.cuda.Event(); end_gpu = cp.cuda.Event(); start_gpu.record()
        stream = cp.cuda.Stream()
        with stream:

            # filter mask
            t = rfftfreq(data.shape[2])
            wfilter = t.astype(cp.float32) #* (1 - t * 2)**3  # parzen

            data[:] = irfft(wfilter*rfft(data, axis=2), axis=2)

        #     for k in range(data.shape[0]):
        #         data[k] = irfft(wfilter*rfft(data[k], axis=1), axis=1)

            stream.synchronize()
        end_gpu.record(); end_gpu.synchronize(); t_meas = cp.cuda.get_elapsed_time(start_gpu,end_gpu)
        print(f"time for applying filter: {t_meas:.2f} ms")            
    
    
    
    
    
    
    
    
    
    
    
    
