#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test reconstruction code

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import sys

sys.path.append('../')
from recon4D import DataGetter
from tomo_encoders.misc import viewer
from tomo_encoders import DataFile
from config import *

if __name__ == "__main__":
    
    dget = DataGetter(*fnames)
    print("Time range: %.2f to %.2f seconds"%(dget.time_exposed_all.min(), dget.time_exposed_all.max()))
    print("Time per 180: 3.61 seconds")    

    time_elapsed_list = [0.0, 150.0] #[0, 10.0, 20.0, 30.0, 60.0, 150.0]

    for time_elapsed in time_elapsed_list:
        vol_t = dget.reconstruct_window(time_elapsed, **recon_params)
        # save it
    #     fname_tstep = os.path.join(recon_path, "idx%i.hdf5"%idx)
        fname_tstep = os.path.join(recon_path, "_%isecs"%time_elapsed)
        ds = DataFile(fname_tstep, tiff = True, \
                      VERBOSITY = 0, \
                      d_shape = vol_t.shape, \
                      d_type = vol_t.dtype, \
                      chunk_size = 0.001)
        ds.create_new(overwrite = True)
        ds.write_full(vol_t)    
        
    fig, ax = plt.subplots(1,3, figsize = (14,6))
    ax[0].imshow(vol_t[int(vol_t.shape[0]*0.2)], cmap = 'gray')
    ax[1].imshow(vol_t[int(vol_t.shape[0]*0.5)], cmap = 'gray')
    ax[2].imshow(vol_t[int(vol_t.shape[0]*0.8)], cmap = 'gray')                
    plt.show()
    plt.close()