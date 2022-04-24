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

from tomo_encoders import Patches
from tomo_encoders.misc import viewer
from tomo_encoders import DataFile
import cupy as cp
from tomo_encoders.reconstruction.project import get_projections
from tomo_encoders.reconstruction.recon import recon_binning
from tomo_encoders.misc.voxel_processing import cylindrical_mask, normalize_volume_gpu

read_path = '/data02/MyArchive/aisteer_3Dencoders/tmp_data/mosaic_raw'

id_start = np.asarray([130,219,221,225,223,225])
id_end = np.asarray([878,879,879,878,877,1027])
hts = id_end - id_start
ntheta = 3000

if __name__ == "__main__":

    
    tot_ht = int(np.sum(hts))
    FULL_SHAPE = (ntheta, tot_ht, 4200)
    print(f'shape of full projection array {FULL_SHAPE}')
    center = 2100.0
    theta = np.linspace(0,np.pi,ntheta,dtype='float32')

    hf = h5py.File(os.path.join(read_path, 'all_layers'), 'w')
    hf.create_dataset("data",FULL_SHAPE)
    hf.create_dataset("theta", data = theta)
    hf.create_dataset("center", data = center)
    hf.close()


    
    for ii in range(0,6):
        
        if ii == 0:
            sw = slice(0, hts[0])        
        else:
            sw = slice(sw.stop, sw.stop + hts[ii])
        
        hf = h5py.File(os.path.join(read_path,'layer%i'%(ii+1)), 'r')
        projs = np.asarray(hf["data"][:,id_start[ii]:id_end[ii],:])
        hf.close() 
        print(f'shape of projection data from layer {ii+1}: {projs.shape}')    
        hf_out = h5py.File(os.path.join(read_path, 'all_layers'), 'a')
        print(f'position in stitched projection array {sw.start}, {sw.stop}')
        hf_out["data"][:,sw,:] = projs
        hf_out.close()

        
        

    
    
    
    

    
    
    