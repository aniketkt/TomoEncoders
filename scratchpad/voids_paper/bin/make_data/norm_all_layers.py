#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 

import numpy as np
import os
import h5py
import sys
import time



raw_fname = '/data02/MyArchive/aisteer_3Dencoders/tmp_data/mosaic_raw/all_layers.hdf5'
raw_fname_8bit = '/data02/MyArchive/aisteer_3Dencoders/tmp_data/mosaic_raw/all_layers_8bit.hdf5'
raw_fname_16bit = '/data02/MyArchive/aisteer_3Dencoders/tmp_data/mosaic_raw/all_layers_16bit.hdf5'
center = 2100.0

if __name__ == "__main__":

    print("estimating data range (min, max)")
    hf = h5py.File(raw_fname, 'r')
    binned_projs = np.asarray(hf["data"][::4,::4,::4])
    [ntheta, nz, n] = hf["data"].shape
    theta = np.asarray(hf["theta"][:])
    min_val = binned_projs.min()
    max_val = binned_projs.max()
    print("\tdone")

    hf_8bit = h5py.File(raw_fname_8bit,'w')
    hf_8bit.create_dataset("data",(ntheta,nz,n), dtype = np.uint8)
    hf_8bit.create_dataset("theta", data = theta)
    hf_8bit.create_dataset("center", data = center)

    hf_16bit = h5py.File(raw_fname_16bit,'w')
    hf_16bit.create_dataset("data",(ntheta,nz,n), dtype = np.uint16)
    hf_16bit.create_dataset("theta", data = theta)
    hf_16bit.create_dataset("center", data = center)

    from tqdm import trange
    for ii in trange(ntheta):
        proj = np.asarray(hf["data"][ii,...]).astype(np.float32)
        proj = (proj - min_val)/(max_val - min_val)
        hf_8bit["data"][ii,...] = (255*proj).astype(np.uint8)
        hf_16bit["data"][ii,...] = ((2**16-1)*proj).astype(np.uint16)

    hf.close()
    hf_8bit.close()
    hf_16bit.close()