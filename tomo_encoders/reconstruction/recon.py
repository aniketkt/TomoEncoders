#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""



"""
import numpy as np
import cupy as cp
import time
import tensorflow as tf
# from cupyx.scipy.fft import rfft, irfft, rfftfreq
import tqdm
from cupyx.scipy.fft import rfft, irfft, rfftfreq, get_fft_plan
from cupyx.scipy.ndimage import gaussian_filter, median_filter
from tomo_encoders import Patches, Grid
from cupyx.scipy import ndimage
from tomo_encoders.reconstruction.retrieve_phase import paganin_filter
from tomo_encoders.reconstruction.cuda_kernels import rec_patch, rec_mask, rec_all
from tomo_encoders.reconstruction.prep import fbp_filter, preprocess, calc_padding    
from tomo_encoders.misc.voxel_processing import TimerGPU
from multiprocessing import Pool, Process
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
    obj_out = np.empty((nz, n, n), dtype = np.float32)
    
    for ic in tqdm.trange(int(np.ceil(nz/nc))):
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

def recon_all_gpu(projs, theta, center, obj, dark_flat = None):
    '''reconstruct with full projection array on gpu and apply some convolutional filters in post-processing
    projection array must fit in GPU memory'''    
    
    # timer = TimerGPU()
    # timer.tic()
    device = cp.cuda.Device()
    memory_pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(memory_pool.malloc)
    # memory_pool.set_limit(size=44*1024**3)
    
    stream_copy = cp.cuda.Stream()
    with stream_copy:
        data = cp.array(projs.astype(np.float32), dtype = cp.float32)
        if dark_flat is not None:
            dark, flat = dark_flat
            dark = cp.array(dark)
            flat = cp.array(flat)
            t_prep = preprocess(data, dark, flat)
        
        # theta and center
        theta = cp.array(theta, dtype = 'float32')
        center = cp.float32(center)
    
        fbp_filter(data) # need to apply filter to full projection  
        rec_all(obj, data, theta, center)
        stream_copy.synchronize()
    del data
    cp.fft.config.clear_plan_cache(); memory_pool.free_all_blocks()    
    device.synchronize()
    # _ = timer.toc(f"reconstruction shape {obj.shape}, median filter {median_kernel}, gaussian filter {blur_sigma}")
    
    return


def recon_all_gpu0(projs, theta, center, obj):
    '''reconstruct with full projection array on gpu and apply some convolutional filters in post-processing
    projection array must fit in GPU memory'''    
    
    # timer = TimerGPU()
    # timer.tic()
    device = cp.cuda.Device()
    memory_pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(memory_pool.malloc)
    # memory_pool.set_limit(size=44*1024**3)
    stream = cp.cuda.Stream()
    
    ntheta, nz, n = projs.shape
    data = cp.empty((ntheta,1,n), dtype = cp.float32)
    theta = cp.array(theta, dtype = cp.float32)
    center = cp.float32(center)
    obj_slice = cp.empty((1,n,n), dtype = cp.float32)

    from tqdm import trange
    for ii in trange(nz):
        with stream:
            data[:,0,:] = cp.array(projs[:,ii,:].astype(np.float32))
            t_filt = fbp_filter(data) # need to apply filter to full projection  
            rec_all(obj_slice, data, theta, center)
            obj[ii] = obj_slice[0]
            stream.synchronize()

    del data
    cp.fft.config.clear_plan_cache(); memory_pool.free_all_blocks()    
    device.synchronize()
    # _ = timer.toc(f"reconstruction shape {obj.shape}, median filter {median_kernel}, gaussian filter {blur_sigma}")
    
    return obj


def recon_patches_3d(projs, theta, center, p3d, apply_fbp = True, TIMEIT = False, segmenter = None, segmenter_batch_size = 256, dark_flat = None, rec_min_max = None):

    timer = TimerGPU("ms")
    memory_pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(memory_pool.malloc)
    device = cp.cuda.Device()
    stream = cp.cuda.Stream()
    z_pts = np.unique(p3d.points[:,0])
    ntheta, nc, n = projs.shape[0], p3d.wd, projs.shape[2]
    with stream:
        data = cp.empty((ntheta, nc, n), dtype = cp.float32)
        theta = cp.array(theta, dtype = cp.float32)
        center = cp.float32(center)
        obj_mask = cp.empty((nc, n, n), dtype = cp.float32)
        
        # fft stuff        
        pad_left, pad_right = calc_padding(data.shape)    
        data_padded = cp.empty((ntheta,nc,n+pad_left+pad_right), dtype = cp.float32)
        data0 = cp.empty((ntheta,nc, (n+pad_left+pad_right)//2+1), dtype = cp.complex64)
        plan_fwd = get_fft_plan(data_padded, axes=2, value_type='R2C')
        plan_inv = get_fft_plan(rfft(data_padded,axis=2), axes=2, value_type='C2R')
        t = rfftfreq(data_padded.shape[2])
        wfilter = t.astype(cp.float32) #* (1 - t * 2)**3  # parzen
        stream.synchronize()

    x = []
    times = []
    cpts_all = []
    from tqdm import tqdm
    pbar = tqdm(total=len(z_pts))
    
    for z_pt in z_pts:
        cpts = p3d.filter_by_condition(p3d.points[:,0] == z_pt).points
        cpts_all.append(cpts.copy())
        cpts[:,0] = 0
        
        with stream:
            # COPY DATA TO GPU
            timer.tic()
            data.set(projs[:,z_pt:z_pt+nc,:].astype(np.float32))
            t_cpu2gpu = timer.toc()
            
            if dark_flat is not None:
                dark = cp.array(dark_flat[0][z_pt:z_pt+nc,...])
                flat = cp.array(dark_flat[1][z_pt:z_pt+nc,...])
                t_prep = preprocess(data, dark, flat)
            stream.synchronize()

        # FBP FILTER
        timer.tic()

        with plan_fwd:
            # filter mask and fft
            #(1 - t * 2)**3  # parzen
            data_padded[:] = cp.pad(data, ((0,0),(0,0),(pad_left, pad_right)), mode = 'edge')
            data0[:] = wfilter*rfft(data_padded, axis=2)
            
        with plan_inv:
            # inverse fft
            data[:] = irfft(data0, axis=2)[...,pad_left:-pad_right]
        t_filt = timer.toc()
            
        with stream:
            # BACK-PROJECTION
            t_mask = make_mask(obj_mask, cpts, p3d.wd)
            t_rec = rec_mask(obj_mask, data, theta, center)
            stream.synchronize()
        
        # EXTRACT PATCHES AND SEND TO CPU
        if segmenter is not None:
            # do segmentation
            xchunk = extract_segmented(obj_mask, cpts, p3d.wd, segmenter, segmenter_batch_size, rec_min_max)
            # xchunk = extract_segmented_cpu(obj_mask, cpts, p3d.wd, segmenter, segmenter_batch_size, rec_min_max)
            times.append([ntheta, nc, n, t_cpu2gpu, t_filt, t_mask, t_rec])
            pass
        else:
            xchunk, t_gpu2cpu = extract_from_mask(obj_mask, cpts, p3d.wd)
            times.append([ntheta, nc, n, t_cpu2gpu, t_filt, t_mask, t_rec, t_gpu2cpu])


        # APPEND AND GO TO NEXT CHUNK
        x.append(xchunk)
        pbar.update(1)
        
    pbar.close()

    device.synchronize()
    del obj_mask, data, theta, center, data_padded, data0, t, wfilter    
    cp.fft.config.clear_plan_cache(); memory_pool.free_all_blocks()    
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


def extract_segmented(obj_mask, cpts, wd, segmenter, batch_size, rec_min_max):
    
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
    
    memory_pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(memory_pool.malloc)
    # cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)

    stream = cp.cuda.Stream()
    yp = cp.empty((batch_size, wd, wd, wd, 1), dtype = cp.float32)

    with stream:
        
        sub_vols = []
        ib = 1
        for idx in range(len(cpts)):
            s = (slice(cpts[idx,0], cpts[idx,0] + wd), \
                 slice(cpts[idx,1], cpts[idx,1] + wd), \
                 slice(cpts[idx,2], cpts[idx,2] + wd))
            
            yp[ib-1,..., 0] = obj_mask[s]
            batch_is_full = (ib == batch_size) # is batch full?
            end_of_chunk = (idx == len(cpts) - 1) # are we at the end of the z-chunk?
            if batch_is_full or end_of_chunk:
                yp[:] = cp.clip(yp, *rec_min_max)
                min_val, max_val = rec_min_max
                yp[:] = (yp - min_val) / (max_val - min_val)
                # use DLPack here as yp is cupy array                
                cap = yp.toDlpack()
                yp_in = tf.experimental.dlpack.from_dlpack(cap)
                yp_cpu = np.round(segmenter.models["segmenter"](yp_in)).astype(np.uint8)
                
                sub_vols.append(yp_cpu[:ib,...,0])
                ib = 0
            ib+=1
        stream.synchronize()
    
    del yp, yp_in, cap
    memory_pool.free_all_blocks()  
    sub_vols = np.concatenate(sub_vols, axis = 0)
    return sub_vols

def extract_segmented_cpu(obj_mask, cpts, wd, segmenter, batch_size, rec_min_max):
    
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
    
    yp = np.empty((batch_size, wd, wd, wd, 1), dtype = np.float32)

        
    sub_vols = []
    ib = 1
    for idx in range(len(cpts)):
        s = (slice(cpts[idx,0], cpts[idx,0] + wd), \
                slice(cpts[idx,1], cpts[idx,1] + wd), \
                slice(cpts[idx,2], cpts[idx,2] + wd))
        
        yp[ib-1,..., 0] = obj_mask[s].get()
        batch_is_full = (ib == batch_size) # is batch full?
        end_of_chunk = (idx == len(cpts) - 1) # are we at the end of the z-chunk?
        if batch_is_full or end_of_chunk:
            yp[:] = np.clip(yp, *rec_min_max)
            min_val, max_val = rec_min_max
            yp[:] = (yp - min_val) / (max_val - min_val)
            yp_out = np.round(segmenter.models["segmenter"](yp)).astype(np.uint8)
            sub_vols.append(yp_out[:ib,...,0])
            ib = 0
        ib+=1
    sub_vols = np.concatenate(sub_vols, axis = 0)
    return sub_vols


















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

    
