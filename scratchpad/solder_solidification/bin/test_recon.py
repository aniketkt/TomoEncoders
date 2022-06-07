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


from tifffile import imwrite

from tomo_encoders.reconstruction.recon import recon_all, recon_binning, recon_patches_3d
parent_path = '/data02/MyArchive/tomo_datasets/solder_imaging/projs'
flat_file = 'dark_fields_Sample07_801_0_04_1mmCu_Annealing_054.h5'
dark_file = 'flat_fields_Sample07_801_0_04_1mmCu_AfterScan_054.h5'
projs_file = 'Sample07_801_0_04_1mmCu_AfterScan_054.h5'
b = 4
b_K = 4

if __name__ == "__main__":

    with h5py.File(os.path.join(parent_path, flat_file), 'r') as hf:
        flat = np.median(hf['exchange/data'][:], axis = 0).astype(np.float32)
    
    with h5py.File(os.path.join(parent_path, dark_file), 'r') as hf:
        dark = np.median(hf['exchange/data'][:], axis = 0).astype(np.float32)

    with h5py.File(os.path.join(parent_path, projs_file), 'r') as hf:
        projs = np.asarray(hf['exchange/data'][:]).astype(np.float32)
        theta = np.radians(np.asarray(hf['exchange/theta'][:])).astype(np.float32)

    
    [ntheta, nz, n] = projs.shape
    center_guess = 916.0 #n/2.0
    projs = projs[:, :-int(nz%32), :-int(n%32)].copy()
    flat = flat[:-int(nz%32), :-int(n%32)]
    dark = dark[:-int(nz%32), :-int(n%32)]

    # rec = recon_all(projs[:,nz//2:nz//2+1,:], theta, center_guess, 1, dark_flat = (dark, flat))[0]
    rec = recon_binning(projs, theta, center_guess, b_K, b, dark_flat = (dark, flat)).get()
    imwrite(os.path.join(parent_path, "recon_binned_b%i_b_K%i.tiff"%(b, b_K)), rec)
    
    rec = recon_all(projs, theta, center_guess, 32, dark_flat = (dark, flat))
    imwrite(os.path.join(parent_path, "recon_full.tiff"%(b, b_K)), rec)
    
    
    