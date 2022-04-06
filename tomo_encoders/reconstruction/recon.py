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
from tomo_encoders import Patches, Grid

def darkflat_correction(data, dark, flat):
    """Dark-flat field correction"""
    xp = cp.get_array_module(data)
    for k in range(data.shape[0]):
        data[k] = (data[k]-dark)/xp.maximum(flat-dark, 1e-6)
    return data

def minus_log(data):
    """Taking negative logarithm"""
    data = -cp.log(cp.maximum(data, 1e-6))
    return data

def show_orthoplane(self, axis, idx, prev_img = None):

    # convert p3d to p2d
    # fill patches on prev_img (if input) and return

    raise NotImplementError("not implemented yet")


source = """
extern "C" {
    void __global__ rec_pts(float *f, float *g, float *theta, int *pts, float center, int ntheta, int nz, int n, int npts)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        if (tx >= npts)
            return;
        int s0 = 0;
        int ind = 0;
        float f0 = 0;
        float sp = 0;
        
        for (int k = 0; k < ntheta; k++)
        {
            sp = (pts[3*tx+2] - n / 2) * __cosf(theta[k]) - (pts[3*tx+1] - n / 2) * __sinf(theta[k]) + center; //polar coordinate
            //linear interpolation
            s0 = roundf(sp);
            ind = k * n * nz + pts[3*tx+0] * n + s0;
            if ((s0 >= 0) & (s0 < n - 1))
                f0 += g[ind] + (g[ind+1] - g[ind]) * (sp - s0) / n;
        }
        f[tx] = f0;
    }



    void __global__ rec_pts_xy(float *f, float *g, float *theta, int *pts, float center, int ntheta, int nz, int n, int npts)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        if (tx >= npts || ty >= nz)
            return;
        int s0 = 0;
        int ind = 0;
        float f0 = 0;
        float sp = 0;
        
        for (int k = 0; k < ntheta; k++)
        {
            sp = (pts[2*tx+1] - n / 2) * __cosf(theta[k]) - (pts[2*tx] - n / 2) * __sinf(theta[k]) + center; //polar coordinate
            //linear interpolation
            s0 = roundf(sp);
            ind = k * n * nz + ty * n + s0;
            if ((s0 >= 0) & (s0 < n - 1))
                f0 += g[ind] + (g[ind+1] - g[ind]) * (sp - s0) / n;
        }
        f[ty*npts+tx] = f0;
    }


    void __global__ rec(float *f, float *g, float *theta, float center, int ntheta, int nz, int n, int stx, int px, int sty, int py, int stz, int pz)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        int tz = blockDim.z * blockIdx.z + threadIdx.z;
        if (tx >= px || ty >= py || tz>=pz)
            return;
        stx += tx;
        sty += ty;
        stz += tz;
        int s0 = 0;
        int ind = 0;
        float f0 = 0;
        float sp = 0;
        
        for (int k = 0; k < ntheta; k++)
        {
            sp = (stx - n / 2) * __cosf(theta[k]) - (sty - n / 2) * __sinf(theta[k]) + center; //polar coordinate
            //linear interpolation
            s0 = roundf(sp);
            ind = k * n * nz + stz * n + s0;
            if ((s0 >= 0) & (s0 < n - 1))
                f0 += g[ind] + (g[ind+1] - g[ind]) * (sp - s0) / n;
        }
        f[tz*px*py+ty*px+tx] = f0;
    }

    void __global__ rec_mask(float *f, float *g, float *theta, float center, int ntheta, int nz, int n)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        int tz = blockDim.z * blockIdx.z + threadIdx.z;
        if (tx >= n || ty >= n || tz>=nz)
            return;
        int s0 = 0;
        int ind = 0;
        float f0 = 0;
        float sp = 0;
        
        if (f[tz*n*n+ty*n+tx] > 0)
            for (int k = 0; k < ntheta; k++)
            {
                sp = (tx - n / 2) * __cosf(theta[k]) - (ty - n / 2) * __sinf(theta[k]) + center; //polar coordinate
                //linear interpolation
                s0 = roundf(sp);
                ind = k * n * nz + tz * n + s0;
                if ((s0 >= 0) & (s0 < n - 1))
                    f0 += g[ind] + (g[ind+1] - g[ind]) * (sp - s0) / n;
            }
            f[tz*n*n+ty*n+tx] = f0;
    }



}

"""
    
    
module = cp.RawModule(code=source)
rec_kernel = module.get_function('rec')
rec_pts_kernel = module.get_function('rec_pts')
rec_pts_xy_kernel = module.get_function('rec_pts_xy')
rec_mask_kernel = module.get_function('rec_mask')

def rec_pts(data, theta, center, pts):
    
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    stream_rec = cp.cuda.Stream()
    
    with stream_rec:
        [ntheta, nz, n] = data.shape
        obj = cp.zeros(len(pts),dtype='float32', order='C')
        data = cp.ascontiguousarray(data)
        theta = cp.ascontiguousarray(theta)     
        pts = cp.ascontiguousarray(pts)     

        rec_pts_kernel((int(np.ceil(len(pts)/1024)),1), (1024,1), \
                   (obj, data, theta, pts, cp.float32(center), ntheta, nz, n, len(pts)))
        stream_rec.synchronize()

    end_gpu.record()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    print("TIME rec_pts: %.2f ms"%t_gpu)
        
    return obj
    
def rec_pts_xy(data, theta, center, pts):
    
    '''
    pts is a sorted array of points y,x with shape (npts,2)
    
    '''
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    stream_rec = cp.cuda.Stream()
    
    with stream_rec:
        [ntheta, nz, n] = data.shape
        obj = cp.zeros((len(pts)*nz),dtype='float32', order='C')
        data = cp.ascontiguousarray(data)
        theta = cp.ascontiguousarray(theta)     
        pts = cp.ascontiguousarray(pts)     

        nbkx = 256
        nthx = int(np.ceil(len(pts)/nbkx))
        nbkz = 4
        nthz = int(np.ceil(nz/nbkz))
        rec_pts_xy_kernel((nthx,nthz), (nbkx,nbkz), \
                   (obj, data, theta, pts, cp.float32(center), ntheta, nz, n, len(pts)))
        stream_rec.synchronize()

    end_gpu.record()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    print("TIME rec_pts: %.2f ms"%t_gpu)
        
    return obj
    

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
            
            yp[ib-1,..., 0] = obj_mask[s].copy()
            batch_is_full = (ib == batch_size) # is batch full?
            end_of_chunk = (idx == len(cpts) - 1) # are we at the end of the z-chunk?
            if batch_is_full or end_of_chunk:
                
                st_seg = cp.cuda.Event(); end_seg = cp.cuda.Event(); st_seg.record()                
                min_val = yp[:ib].min()
                max_val = yp[:ib].max()
                yp[:] = (yp - min_val) / (max_val - min_val)
                
                # use DLPack here as yp is cupy array                
                # cap = yp.toDlpack()
                # yp_tf = tf.experimental.dlpack.from_dlpack(cap)
                yp_cpu = np.round(segmenter.models["segmenter"].predict(yp.get()))
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
    print(f"voxel processing time for U-net: {t_seg/(np.prod(sub_vols.shape))*1e6:.2f} ns")
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
    

def rec_mask(obj, data, theta, center):
    """Reconstruct mask on GPU"""
    [ntheta, nz, n] = data.shape
    
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    stream_rec = cp.cuda.Stream()
    with stream_rec:
        
        data = cp.ascontiguousarray(data)
        theta = cp.ascontiguousarray(theta)
        
        rec_mask_kernel((int(cp.ceil(n/16)), int(cp.ceil(n/16)), \
                    int(cp.ceil(nz/4))), (16, 16, 4), \
                   (obj, data, theta, cp.float32(center),\
                    ntheta, nz, n))
        stream_rec.synchronize()
    
    end_gpu.record()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    
    # print("TIME rec_mask: %.2f ms"%t_gpu)
    return t_gpu
    
    
def rec_patch(data, theta, center, stx, px, sty, py, stz, pz, TIMEIT = False):
    """Reconstruct subvolume [stz:stz+pz,sty:sty+py,stx:stx+px] on GPU"""
    [ntheta, nz, n] = data.shape
    
    
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    stream_rec = cp.cuda.Stream()
    with stream_rec:
        
        obj = cp.zeros([pz, py, px], dtype='float32', order = 'C')
        data = cp.ascontiguousarray(data)
        theta = cp.ascontiguousarray(theta)
        
        rec_kernel((int(cp.ceil(px/16)), int(cp.ceil(py/16)), \
                    int(cp.ceil(pz/4))), (16, 16, 4), \
                   (obj, data, theta, cp.float32(center),\
                    ntheta, nz, n, stx, px, sty, py, stz, pz))
        stream_rec.synchronize()
    
    end_gpu.record()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    
    
    if TIMEIT:
#         print("TIME rec_patch: %.2f ms"%t_gpu)
        return obj, t_gpu
    else:
        return obj

def _msg_exec_time(self, func, t_exec):
    print("TIME: %s: %.2f seconds"%(func.__name__, t_exec))
    return


def calc_padding(data_shape):
    # padding, make sure the width of projection is divisible by four after padding
    [ntheta, nz, n] = data_shape
    n_pad = n*(1 + 0.25*2) # 1/4 padding
    n_pad = int(np.ceil(n_pad/8.0)*8.0) 
    pad_left = int((n_pad - n)//2)
    pad_right = n_pad - n - pad_left    
    
    # print(f'n: {n}, n_pad: {n_pad}')
    # print(f'pad_left: {pad_left}, pad_right: {pad_right}')    
    return pad_left, pad_right

def fbp_filter(data, TIMEIT = False):
    """FBP filtering of projections"""
    
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    
    pad_left, pad_right = calc_padding(data.shape)    
    # padding
    data_padded = cp.pad(data, ((0,0),(0,0),(pad_left, pad_right)), mode = 'edge')

    # fft plan
    plan_fwd = get_fft_plan(data_padded, axes=2, value_type='R2C')
    plan_inv = get_fft_plan(rfft(data_padded,axis=2), axes=2, value_type='C2R')
    
    with plan_fwd:

        # filter mask
        t = rfftfreq(data_padded.shape[2])
        wfilter = t.astype(cp.float32) #* (1 - t * 2)**3  # parzen

        # fft
        data0 = wfilter*rfft(data_padded, axis=2)

    with plan_inv:
        # inverse fft
        data[:] = irfft(data0, axis=2)[...,pad_left:-pad_right]
        
    end_gpu.record()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    
    if TIMEIT:
        # print("TIME fbp_filter: %.2f ms"%t_gpu)
        pass
    
    return t_gpu



def recon_binning(projs, theta, center, b_K, b, apply_fbp = True, TIMEIT = False):
    
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
        # option 2: simple binning
        # data = cp.array(projs[::b_K, ::b, ::b].copy())

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
    
    obj = obj.get()
    
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


def recon_patches_3d(projs, theta, center, p3d, apply_fbp = True, TIMEIT = False, segmenter = None, segmenter_batch_size = 256):

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
            pass
        else:
            xchunk, t_gpu2cpu = extract_from_mask(obj_mask, cpts, p3d.wd)
            times.append([ntheta, nc, n, t_cpu2gpu, t_filt, t_mask, t_rec, t_gpu2cpu])
        

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


    







if __name__ == "__main__":
    
    print('just a bunch of functions')

    
