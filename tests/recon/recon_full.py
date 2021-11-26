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
from tomo_encoders.reconstruction.solver import solver

nz = 200
nx = 1920
ntheta = 1200
PROJS_SHAPE = (ntheta, nz, nx)

if __name__ == "__main__":

#     read_fpath = os.path.join(projs_path, filename + '_projs_bin%i_ntheta%i.hdf5'%(binning,ntheta))
    projs = np.random.normal(0, 2**16-1, PROJS_SHAPE)
    theta = np.linspace(0, np.pi, ntheta)
    center = float(nz/2)
    dark = np.zeros((nz, nx))
    flat = np.ones((nz, nx))*(2**16-1)
    
    vol_shape = projs.shape[1:] + (projs.shape[-1],)
    print("full projections shape: ", projs.shape)
    print("full reconstructed volume shape: ", vol_shape)
    
    with cp.cuda.Device(0):
        center = projs.shape[-1]//2.0
        vol = solver(projs, theta, center, flat, dark, TIMEIT = True)
        cp.cuda.Device(0).synchronize()
    
    
    
    
