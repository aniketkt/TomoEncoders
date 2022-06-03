#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
from audioop import mul
import h5py
import numpy as np
import sys
import os

# coarse mapping parameters
b = 4
b_K = 4
wd = 32

# detailed mapping parameters
pixel_res = 1.17
size_um = -1 # um
void_rank = 1
radius_around_void_um = 1000.0 # um


# U-net parameters
sys.path.append('/home/atekawade/TomoEncoders/scratchpad/voids_paper/configs')
from params import model_path, get_model_params
model_tag = "M_a07"
model_names = {"segmenter" : "segmenter_Unet_%s"%model_tag}



## file names
raw_fname = '/data02/MyArchive/aisteer_3Dencoders/tmp_data/mosaic_raw/all_layers_16bit.hdf5'
raw_fname_b2 = '/data02/MyArchive/aisteer_3Dencoders/tmp_data/mosaic_raw/all_layers_b2_16bit.hdf5'
voids_dir = '/data02/MyArchive/aisteer_3Dencoders/tmp_data/voids_data'
data_output = '/data02/MyArchive/aisteer_3Dencoders/tmp_data/output_vis'
time_logs = '/home/atekawade/Dropbox/Arg/transfers/runtime_plots/voids_paper/time_logs'
ply_dir = '/home/atekawade/Dropbox/Arg/transfers/runtime_plots/voids_paper/ply'
ply_subset = os.path.join(ply_dir, 'subset.ply')
ply_coarse = os.path.join(ply_dir, 'coarse.ply')

if not os.path.exists(time_logs):
    os.makedirs(time_logs)
if not os.path.exists(ply_dir):
    os.makedirs(ply_dir)
if not os.path.exists(voids_dir):
    os.makedirs(voids_dir)

def read_raw_data(fname, multiplier):
    hf = h5py.File(fname, 'r')
    crop_wdb = int(hf["data"].shape[-1]%(multiplier))
    if crop_wdb:
        sw = slice(None,-crop_wdb)
    else:
        sw = slice(None,None)
    crop_z_wdb = int(hf["data"].shape[1]%(multiplier))
    # if crop_z_wdb:
    #     sz = slice(None,-crop_z_wdb)
    # else:
    #     sz = slice(None,None)
    sz = slice(-4096,None)
    projs = np.asarray(hf["data"][:, sz, sw])    
    theta = np.asarray(hf["theta"][:])
    center = float(np.asarray(hf["center"]))
    hf.close()

    return projs, theta, center
    # [multiplier*10:,]

def read_raw_data_b2():
    hf = h5py.File(raw_fname_b2, 'r')
    projs = np.asarray(hf["data"][:])    
    print(projs.shape)
    theta = np.asarray(hf["theta"][:])
    center = float(np.asarray(hf["center"]))
    hf.close()

    return projs, theta, center




if __name__ == "__main__":

    pass    
    
