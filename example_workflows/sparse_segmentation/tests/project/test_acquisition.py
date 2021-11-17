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
from tomo_encoders import DataFile 
import os 


fpath = '/data02/MyArchive/AM_part_Xuan/data/mli_L206_HT_650_L3_rec_1x1_uint16.hdf5' 
plot_out_path = '/home/atekawade/Dropbox/Arg/transfers/runtime_plots/'
import matplotlib as mpl
mpl.use('Agg')

from tomo_encoders.tasks.sparse_segmenter.recon import recon_binning
from tomo_encoders.tasks.sparse_segmenter.project import acquire_data


if __name__ == "__main__":

    ds = DataFile(fpath, data_tag = 'data', tiff = False, VERBOSITY = 0)
    vol = ds.read_full().astype(np.float32)
    
    sbin = tuple([slice(None,None,2)]*3) 
    vol = vol[sbin].copy()
    vol = (vol - vol.min()) / (vol.max() - vol.min())

    
#     import pdb; pdb.set_trace()
    point = (np.asarray(vol.shape)/3).astype(int)
    print("\n\npoint: ", point)
    projs, theta, center = acquire_data(vol, point, \
                                        2001, \
                                        FOV = (700,300), \
                                        pnz = 4)

    
    THETA_BINNING = 1
    DET_BINNING = 1
    
    vol = recon_binning(projs, theta, center, \
                        THETA_BINNING, DET_BINNING, DET_BINNING, \
                        apply_fbp = True, TIMEIT = True)
    
    
    view_midplanes(vol)
    plt.savefig(os.path.join(plot_out_path, "test_acquisition.png"))
    
    
    