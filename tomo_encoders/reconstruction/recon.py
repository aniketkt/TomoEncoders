#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""



"""
import numpy as np
import cupy as cp
import time
import tensorflow as tf
# from cupyx.scipy.fft import rfft, irfft, rfftfreq

from cupyx.scipy.fft import rfft, irfft, rfftfreq, get_fft_plan
from cupyx.scipy.ndimage import gaussian_filter, median_filter
from tomo_encoders import Patches, Grid
from cupyx.scipy import ndimage
from tomo_encoders.reconstruction.retrieve_phase import paganin_filter
from tomo_encoders.reconstruction.cpp_kernels import rec_patch, rec_mask, rec_all
from tomo_encoders.reconstruction.prep import fbp_filter, preprocess    
from tomo_encoders.misc.voxel_processing import TimerGPU

def recon_all(projs, theta, center, nc, dark_flat = None):

    ntheta, nz, n = projs.shape
    data = cp.empty((ntheta, nc, n), dtype = cp.float32)
    theta = cp.array(theta, dtype = cp.float32)
    center = cp.float32(center)
    if dark_flat is not None:
        dark, flat = dark_flat
        dark = cp.array(dark)
        flat = cp.array(flat)
    obj_gpu = cp.empty((nc, n, n), dtype = cp.float32)
    obj_out = np.zeros((nz, n, n), dtype = np.float32)
    
    
    for ic in range(int(np.ceil(nz/nc))):
        s_chunk = slice(ic*nc, (ic+1)*nc)
        # COPY DATA TO GPU
        start_gpu = cp.cuda.Event(); end_gpu = cp.cuda.Event(); start_gpu.record()
        stream = cp.cuda.Stream()
        with stream:
            data.set(projs[:,s_chunk,:].astype(np.float32))
        end_gpu.record(); end_gpu.synchronize(); t_cpu2gpu = cp.cuda.get_elapsed_time(start_gpu,end_gpu)
        # print(f"\tTIME copying data to gpu: {t_cpu2gpu:.2f} ms")            
            
        # PREPROCESS
        if dark_flat is not None:
            t_prep = preprocess(data, dark[s_chunk], flat[s_chunk])

        # FBP FILTER
        t_filt = fbp_filter(data)
        # print(f'\tTIME fbp filter: {t_filt:.2f} ms')
        
        # BACK-PROJECTION
        t_rec = rec_all(obj_gpu, data, theta, center)
        # print(f'\tTIME back-projection: {t_rec:.2f} ms')
        
        obj_out[s_chunk] = obj_gpu.get()

    del obj_gpu, data, theta, center    
    cp._default_memory_pool.free_all_blocks()    
    
    return obj_out

def recon_coarse(projs, theta, center, blur_sigma = 0, dark_flat = None, median_kernel = 1):
    '''reconstruct with full projection array on gpu and apply some convolutional filters in post-processing
    projection array must fit in GPU memory'''    

    timer = TimerGPU()
    timer.tic()
    device = cp.cuda.Device()
    memory_pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(memory_pool.malloc)
    
    stream_copy = cp.cuda.Stream()
    with stream_copy:
        data = cp.array(projs)
        if dark_flat is not None:
            dark, flat = dark_flat
            dark = cp.array(dark)
            flat = cp.array(flat)
            t_prep = preprocess(data, dark, flat)
        
        # theta and center
        theta = cp.array(theta, dtype = 'float32')
        center = cp.float32(center)
    
        fbp_filter(data) # need to apply filter to full projection  
        
        vol_shape = (data.shape[1], data.shape[2], data.shape[2])
        obj = cp.empty(vol_shape, dtype = cp.float32)
        rec_all(obj, data, theta, center)
   
        if blur_sigma > 0:
            obj = gaussian_filter(obj, blur_sigma)

        if median_kernel > 1:
            obj = ndimage.median_filter(obj, median_kernel)

        stream_copy.synchronize()

    cp._default_memory_pool.free_all_blocks()    
    device.synchronize()
    _ = timer.toc(f"reconstruction shape {obj.shape}, median filter {median_kernel}, gaussian filter {blur_sigma}")
    return obj

def recon_binning(projs, theta, center, b_K, b, apply_fbp = True, TIMEIT = False, blur_sigma = 0, dark_flat = None, median_kernel = 1):
    
    '''
    reconstruct with binning projections and theta
    
    Parameters
    ----------
    projs : np.ndarray  
        array of projection images shaped as ntheta, nrows, ncols
    theta : np.ndarray
        array of theta values (length = ntheta)  
    center : float  
        center value for the projection data  
    theta_binning : int
        binning of theta
    z_binning : int
        vertical binning of projections
    col_binning : int  
        horizontal binning of projections  
    
    Returns
    -------
    
    '''
    
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    
    device = cp.cuda.Device()
    memory_pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(memory_pool.malloc)
    
    stream_copy = cp.cuda.Stream()
    with stream_copy:
        [_, nz, n] = projs.shape
        
        # option 1: average pooling
        projs = projs[::b_K].copy()
        data = cp.array(projs.reshape(projs.shape[0], nz//b, b, n//b, b).mean(axis=(2,4)))
        if dark_flat is not None:
            dark, flat = dark_flat
            dark = cp.array(dark.reshape(nz//b, b, n//b, b).mean(axis=(1,3)))
            flat = cp.array(flat.reshape(nz//b, b, n//b, b).mean(axis=(1,3)))
            t_prep = preprocess(data, dark, flat)
        
        # option 2: simple binning
        # data = cp.array(projs[::b_K, ::b, ::b].copy())
        # if dark_flat is not None:
        #     dark, flat = dark_flat
        #     dark = cp.array(dark[::b,::b])
        #     flat = cp.array(flat[::b,::b])
        #     t_prep = preprocess(data, dark, flat)

        # theta and center
        theta = cp.array(theta[::b_K], dtype = 'float32')
        center = cp.float32(center/b)
        vol_shape = (data.shape[1], data.shape[2], data.shape[2])
        stream_copy.synchronize()
    
    if apply_fbp:
        fbp_filter(data) # need to apply filter to full projection  
        
    # st* - start, p* - number of points
    stz, sty, stx = (0,0,0)
    pz, py, px = vol_shape
    st = time.time()
    obj = rec_patch(data, theta, center, \
              stx, px, \
              sty, py, \
              0,   pz) # 0 since projections were cropped vertically
    
    if blur_sigma > 0:
        obj = gaussian_filter(obj, blur_sigma)

    if median_kernel > 1:
        obj = ndimage.median_filter(obj, median_kernel)

   # obj_cpu = obj.get()
    # del obj
    cp._default_memory_pool.free_all_blocks()    
    device.synchronize()
#     print('total bytes: ', memory_pool.total_bytes())    
    
    end_gpu.record()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    
    if TIMEIT:
        print("TIME binned reconstruction: %.2f ms"%t_gpu)
        return obj, t_gpu
    else:
        return obj


def recon_patches_3d(projs, theta, center, p3d, apply_fbp = True, TIMEIT = False, segmenter = None, segmenter_batch_size = 256, dark_flat = None):

    z_pts = np.unique(p3d.points[:,0])

    ntheta, nc, n = projs.shape[0], p3d.wd, projs.shape[2]
    data = cp.empty((ntheta, nc, n), dtype = cp.float32)
    theta = cp.array(theta, dtype = cp.float32)
    center = cp.float32(center)
    obj_mask = cp.empty((nc, n, n), dtype = cp.float32)
    x = []
    times = []
    cpts_all = []
    for z_pt in z_pts:
        cpts = p3d.filter_by_condition(p3d.points[:,0] == z_pt).points
        cpts_all.append(cpts.copy())
        cpts[:,0] = 0
        
        # COPY DATA TO GPU
        start_gpu = cp.cuda.Event(); end_gpu = cp.cuda.Event(); start_gpu.record()
        stream = cp.cuda.Stream()
        with stream:
            data.set(projs[:,z_pt:z_pt+nc,:])
        end_gpu.record(); end_gpu.synchronize(); t_cpu2gpu = cp.cuda.get_elapsed_time(start_gpu,end_gpu)
        # print(f"overhead for copying data to gpu: {t_cpu2gpu:.2f} ms")            
            
        if dark_flat is not None:
            dark = cp.array(dark_flat[0][z_pt:z_pt+nc,...])
            flat = cp.array(dark_flat[1][z_pt:z_pt+nc,...])
            t_prep = preprocess(data, dark, flat)
        
        # FBP FILTER
        if apply_fbp:
            t_filt = fbp_filter(data)
        
        # BACK-PROJECTION
        t_mask = make_mask(obj_mask, cpts, p3d.wd)
        t_rec = rec_mask(obj_mask, data, theta, center)
        
        # EXTRACT PATCHES AND SEND TO CPU
        if segmenter is not None:
            # do segmentation
            xchunk, t_gpu2cpu, t_seg = extract_segmented(obj_mask, cpts, p3d.wd, segmenter, segmenter_batch_size)
            times.append([ntheta, nc, n, t_cpu2gpu, t_filt, t_mask, t_rec, t_gpu2cpu, t_seg])
            # print(f"rec: {t_rec:.2f}; gpu2cpu: {t_gpu2cpu:.2f}; seg: {t_seg:.2f}")
            pass
        else:
            xchunk, t_gpu2cpu = extract_from_mask(obj_mask, cpts, p3d.wd)
            times.append([ntheta, nc, n, t_cpu2gpu, t_filt, t_mask, t_rec, t_gpu2cpu])
            # print(f"rec: {t_rec:.2f}; gpu2cpu: {t_gpu2cpu:.2f}")
        

        # APPEND AND GO TO NEXT CHUNK
        x.append(xchunk)

    del obj_mask, data, theta, center    
    cp._default_memory_pool.free_all_blocks()    
    
    cpts_all = np.concatenate(cpts_all, axis = 0)
    x = np.concatenate(x, axis = 0)
    
    
    p3d = Grid(p3d.vol_shape, initialize_by = "data", \
               points = cpts_all, width = p3d.wd)
    if TIMEIT:
        return x, p3d, np.asarray(times)
    else:
        return x, p3d


def extract_from_mask(obj_mask, cpts, wd):
    
    start_gpu = cp.cuda.Event(); end_gpu = cp.cuda.Event(); start_gpu.record()
    stream = cp.cuda.Stream()
    with stream:
        
        sub_vols = []
        for idx in range(len(cpts)):
            s = (slice(cpts[idx,0], cpts[idx,0] + wd), \
                 slice(cpts[idx,1], cpts[idx,1] + wd), \
                 slice(cpts[idx,2], cpts[idx,2] + wd))
            sub_vols.append(obj_mask[s].get())
        stream.synchronize()
    end_gpu.record(); end_gpu.synchronize(); t_gpu2cpu = cp.cuda.get_elapsed_time(start_gpu,end_gpu)
    # print(f"overhead for extracting sub_vols to cpu: {t_gpu2cpu:.2f} ms")        
    
    return sub_vols, t_gpu2cpu


def extract_segmented(obj_mask, cpts, wd, segmenter, batch_size):
    
    ''' this is a chunk of reconstructed data along some z-chunk size (typically 32 or 64).
    wd is 32 (patch width).
    if total projection width is 2048, then we get 2*(2048/32)**2 = 8192 patches.
    let's say we pick r = 0.05, we have 410 patches.
    batch_size of 256 seems reasonable. recommended mapping
    projection_width r             batch_size
    2048             0.05 - 1.0    256
    2048             0.01 - 0.05   128
    1024             0.05 - 1.0    128
    4096             0.01 - 1.0    256

    '''
    start_gpu = cp.cuda.Event(); end_gpu = cp.cuda.Event(); start_gpu.record()
    stream = cp.cuda.Stream()
    yp = cp.empty((batch_size, wd, wd, wd, 1), dtype = cp.float32)

    with stream:
        
        sub_vols = []
        t_seg = []
        ib = 1
        for idx in range(len(cpts)):
            # print(r' %i '%ib, end = "")
            s = (slice(cpts[idx,0], cpts[idx,0] + wd), \
                 slice(cpts[idx,1], cpts[idx,1] + wd), \
                 slice(cpts[idx,2], cpts[idx,2] + wd))
            
            yp[ib-1,..., 0] = obj_mask[s]
            batch_is_full = (ib == batch_size) # is batch full?
            end_of_chunk = (idx == len(cpts) - 1) # are we at the end of the z-chunk?
            if batch_is_full or end_of_chunk:
                
                min_val = yp[:ib].min()
                max_val = yp[:ib].max()
                yp[:] = (yp - min_val) / (max_val - min_val)
                # use DLPack here as yp is cupy array                
                cap = yp.toDlpack()
                yp_in = tf.experimental.dlpack.from_dlpack(cap)
                # if DLPack doesn't work. Just transfer to cpu instead
                # yp_in = yp.get()
                st_seg = cp.cuda.Event(); end_seg = cp.cuda.Event(); st_seg.record()                
                yp_cpu = np.round(segmenter.models["segmenter"](yp_in))
                end_seg.record(); end_seg.synchronize(); t_seg.append(cp.cuda.get_elapsed_time(st_seg,end_seg))
                sub_vols.append(yp_cpu[:ib,...,0])
                ib = 0
            ib+=1
            # print(f'sub_vols shape: {np.shape(sub_vols)}')
            
        stream.synchronize()
    end_gpu.record(); end_gpu.synchronize(); t_gpu2cpu = cp.cuda.get_elapsed_time(start_gpu,end_gpu)
    
    sub_vols = np.concatenate(sub_vols, axis = 0)
    t_seg = np.sum(t_seg)
    t_gpu2cpu -= t_seg
    # import pdb; pdb.set_trace()
    # print(f"voxel processing time for U-net: {t_seg/(np.prod(sub_vols.shape))*1e6:.2f} ns")
    return sub_vols, t_gpu2cpu, t_seg

    
def make_mask(obj_mask, corner_pts, wd):
    # MAKE OBJ_MASK FROM PATCH COORDINATES
    start_gpu = cp.cuda.Event(); end_gpu = cp.cuda.Event(); start_gpu.record()
    stream = cp.cuda.Stream()
    with stream:
        obj_mask.put(cp.arange(obj_mask.size),cp.zeros(obj_mask.size, dtype='float32'))    
        for idx in range(len(corner_pts)):
            s = (slice(corner_pts[idx,0], corner_pts[idx,0] + wd), \
                 slice(corner_pts[idx,1], corner_pts[idx,1] + wd), \
                 slice(corner_pts[idx,2], corner_pts[idx,2] + wd))
            obj_mask[s] = cp.ones((wd, wd, wd), dtype = 'float32')
        stream.synchronize()
    end_gpu.record(); end_gpu.synchronize(); t_meas = cp.cuda.get_elapsed_time(start_gpu,end_gpu)
    # print(f"overhead for making mask from patch coordinates: {t_meas:.2f} ms")        
    return t_meas





if __name__ == "__main__":
    
    print('just a bunch of functions')

    
