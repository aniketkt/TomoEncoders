#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
from operator import mod
from tomo_encoders.misc.voxel_processing import modified_autocontrast
from tomo_encoders.reconstruction.recon import recon_binning, recon_patches_3d
import cupy as cp
import numpy as np
from skimage.filters import threshold_otsu
from tomo_encoders import Grid


def coarse_segmentation(projs, theta, center, b_K, b, blur_sigma):
    '''
    coarse reconstruct and thresholding
    input numpy projection data
    return cupy array
    '''    
    st_rec = cp.cuda.Event(); end_rec = cp.cuda.Event(); st_rec.record()
    V_bin = recon_binning(projs, theta, center, b_K, b, blur_sigma = blur_sigma)    
    
    print("\tTODO: modified_autocontrast invokes GPU-CPU transfer. need to implement it on GPU")
    min_max = modified_autocontrast(V_bin.get(), s = 0.05, normalize_sampling_factor=1)

    end_rec.record(); end_rec.synchronize(); t_rec = cp.cuda.get_elapsed_time(st_rec,end_rec)
    print(f"\tTIME reconstructing with binning - {t_rec/1000.0:.2f} secs")
    
    # segmentation
    thresh = cp.float32(threshold_otsu(V_bin[::4,::4,::4].reshape(-1).get()))
    V_bin = (V_bin < thresh).astype(cp.uint8)
    return V_bin, min_max    


def guess_surface(V_bin, b, wd):
    
    # find patches on surface
    wdb = int(wd//b)
    p3d = Grid(V_bin.shape, width = wdb)
    
    x = p3d.extract(V_bin)
    is_surf = (np.std(x, axis = (1,2,3)) > 0.0)
    is_ones = (np.sum(x, axis = (1,2,3))/(wdb**3) == 1)
    is_zeros = (np.sum(x, axis = (1,2,3))/(wdb**3) == 0)
    
    p3d = p3d.rescale(b)
    p3d_surf = p3d.filter_by_condition(is_surf)
    p3d_ones = p3d.filter_by_condition(is_ones)
    p3d_zeros = p3d.filter_by_condition(is_zeros)
    eff = len(p3d_surf)*(wd**3)/np.prod(p3d_surf.vol_shape)
    print(f"\tSTAT: r value: {eff*100.0:.2f}")        
    return p3d_surf, p3d_ones, p3d_zeros

def process_patches(projs, theta, center, fe, p_surf, min_max):

    # SCHEME 1: integrate reconstruction and segmention (segments data on gpu itself)
    # st_proc = cp.cuda.Event(); end_proc = cp.cuda.Event(); st_proc.record()
    # x_surf, p_surf = recon_patches_3d(projs, theta, center, p_surf, \
    #                                   apply_fbp = True, segmenter = fe, \
    #                                   segmenter_batch_size = 256)
    # end_proc.record(); end_proc.synchronize(); t_surf = cp.cuda.get_elapsed_time(st_proc,end_proc)
    
    
    # SCHEME 2: reconstruct and segment separately (copies rec data from gpu to cpu)
    st_rec = cp.cuda.Event(); end_rec = cp.cuda.Event(); st_rec.record()
    x_surf, p_surf = recon_patches_3d(projs, theta, center, p_surf, \
                                      apply_fbp =True)
    end_rec.record(); end_rec.synchronize(); t_rec = cp.cuda.get_elapsed_time(st_rec,end_rec)
    st_seg = cp.cuda.Event(); end_seg = cp.cuda.Event(); st_seg.record()
    
    x_surf = np.clip(x_surf, *min_max)
    x_surf = fe.predict_patches("segmenter", x_surf[...,np.newaxis], 256, None, min_max = min_max)[...,0]
    end_seg.record(); end_seg.synchronize(); t_seg = cp.cuda.get_elapsed_time(st_seg,end_seg)
    
    print(f'\tTIME: local reconstruction - {t_rec/1000.0:.2f} secs')    
    print(f'\tTIME: local segmentation - {t_seg/1000.0:.2f} secs')
    print(f'\tSTAT: total patches in neighborhood: {len(p_surf)}')    
    return x_surf, p_surf
    
    
    

