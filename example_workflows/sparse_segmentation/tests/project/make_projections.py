#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import sys 
import matplotlib.pyplot as plt 
import numpy as np 
from porespy import generators 
from tomo_encoders.misc_utils.feature_maps_vis import view_midplanes 
import cupy as cp 
import time 
import h5py 
from tomopy import project 
sys.path.append('/data02/MyArchive/aisteer_3Dencoders/TomoEncoders/tomo_encoders/tasks/sparse_segmenter/') 
#from recon_subvol import fbp_filter, recon_patch 
from ct_segnet.data_utils.data_io import DataFile 
import os 
import signal 
import tomocg as pt


fpath = '/data02/MyArchive/AM_part_Xuan/data/mli_L206_HT_650_L3_rec_1x1_uint16.hdf5' 
projs_path = '/data02/MyArchive/AM_part_Xuan/projs' 
if not os.path.exists(projs_path): os.makedirs(projs_path)
binning = 2 
ntheta = 1500
pnz = 4  # number of slices for simultaneous processing in tomography 


if __name__ == "__main__":

    ds = DataFile(fpath, data_tag = 'data', tiff = False, VERBOSITY = 0)
    vol = ds.read_full().astype(np.float32)
    
    sbin = tuple([slice(None,None,binning)]*3) 
    vol = vol[sbin].copy()
    vol = (vol - vol.min()) / (vol.max() - vol.min())

    
#     import pdb; pdb.set_trace()
    # make sure the width of projection is divisible by four after padding
    proj_w = vol.shape[-1]
    tot_width = int(proj_w*(1 + 0.25*2)) # 1/4 padding
    tot_width = int(np.ceil(tot_width/8)*8) 
    padding = int((tot_width - proj_w)//2)
    
    # make sure the height of projection is divisible by pnz  
    proj_h = vol.shape[0]
    tot_ht = int(np.ceil(proj_h/(2*pnz))*(2*pnz)) 
    padding_z = int(tot_ht - proj_h)
    
    
    vol = np.pad(vol, ((0,padding_z),\
                       (padding,padding),\
                       (padding,padding)), mode = 'edge')
    theta = np.linspace(0,np.pi,ntheta,dtype='float32') 
    
    nz = vol.shape[0] 
    n = vol.shape[-1] 
    center = n/2.0 
    r = nz//2 # +overlap? because some border pixels are being missed by solver 
    s1 = slice(None,r,None) 
    s2 = slice(-r,None,None) 
    u0 = vol[s1] + 1j*vol[s2] 
    # u0 = vol+1j*0 
    ngpus=1 
    # Class gpu solver 
    t0 = time.time() 
    with pt.SolverTomo(theta, ntheta, r, n, pnz, center, ngpus) as slv: 
        # generate data 
        data = slv.fwd_tomo_batch(u0) 

        projs = np.zeros((ntheta, nz, n), dtype = 'float32') 
        projs[:,s1,:] = data.real 
        projs[:,s2,:] = data.imag 
    #     projs = data.real 

    
    print("padded projections shape: %s"%str(projs.shape)) 
    
    if padding_z == 0:
        projs = projs[:, :, padding:-padding]     
    else:
        projs = projs[:, :-padding_z, padding:-padding]     
    
    print("time %.4f"%(time.time()- t0)) 
    print("projections shape: %s"%str(projs.shape)) 

    filename = os.path.split(fpath)[-1].split('_rec_1x1_uint16.hdf5')[0]
    save_fpath = os.path.join(projs_path, filename + '_projs_bin%i_ntheta%i.hdf5'%(binning,ntheta))
    
    with h5py.File(save_fpath, 'w') as hf:
        hf.create_dataset('data', data = projs)
        hf.create_dataset('theta', data = theta)
        hf.create_dataset('center', data = projs.shape[2]/2)
