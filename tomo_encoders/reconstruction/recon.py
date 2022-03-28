#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""



"""
import numpy as np
import cupy as cp
import time
import h5py
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
    

def extract_from_mask(obj_mask, cpts):
    
    start_gpu = cp.cuda.Event(); end_gpu = cp.cuda.Event(); start_gpu.record()
    stream = cp.cuda.Stream()
    with stream:
        
        sub_vols = []
        for idx in range(len(cpts)):
            s = (slice(cpts[idx,0], cpts[idx,0] + 32), \
                 slice(cpts[idx,1], cpts[idx,1] + 32), \
                 slice(cpts[idx,2], cpts[idx,2] + 32))
            sub_vols.append(obj_mask[s].get())
        stream.synchronize()
    end_gpu.record(); end_gpu.synchronize(); t_gpu2cpu = cp.cuda.get_elapsed_time(start_gpu,end_gpu)
    # print(f"overhead for extracting sub_vols to cpu: {t_gpu2cpu:.2f} ms")        
    
    return sub_vols, t_gpu2cpu
    
    
def make_mask(obj_mask, corner_pts):
    # MAKE OBJ_MASK FROM PATCH COORDINATES
    start_gpu = cp.cuda.Event(); end_gpu = cp.cuda.Event(); start_gpu.record()
    stream = cp.cuda.Stream()
    with stream:
        obj_mask.put(cp.arange(obj_mask.size),cp.zeros(obj_mask.size, dtype='float32'))    
        for idx in range(len(corner_pts)):
            s = (slice(corner_pts[idx,0], corner_pts[idx,0] + 32), \
                 slice(corner_pts[idx,1], corner_pts[idx,1] + 32), \
                 slice(corner_pts[idx,2], corner_pts[idx,2] + 32))
            obj_mask[s] = cp.ones((32, 32, 32), dtype = 'float32')
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
        print("TIME fbp_filter: %.2f ms"%t_gpu)
    
    return t_gpu



def recon_binning(projs, theta, center, theta_binning, z_binning, col_binning, apply_fbp = True, TIMEIT = False):
    
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
        data = cp.array(projs[::theta_binning, ::z_binning, ::col_binning].copy())
        theta = cp.array(theta[::theta_binning], dtype = 'float32')
        center = cp.float32(center/col_binning)
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


def recon_patches_3d_2(projs, theta, center, p3d, apply_fbp = True, TIMEIT = False):

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
        t_mask = make_mask(obj_mask, cpts)
        t_rec = rec_mask(obj_mask, data, theta, center)
        
        # EXTRACT PATCHES AND SEND TO CPU
        xchunk, t_gpu2cpu = extract_from_mask(obj_mask, cpts)
        times.append([ntheta, nc, n, t_cpu2gpu, t_filt, t_mask, t_rec, t_gpu2cpu])
        x.append(xchunk)

    del obj_mask, data, theta, center    
    cp._default_memory_pool.free_all_blocks()    
    
    cpts_all = np.concatenate(cpts_all, axis = 0)
    x = np.concatenate(x, axis = 0)
    x = np.asarray(x).reshape(-1,p3d.wd,p3d.wd,p3d.wd)
    
    p3d = Grid(p3d.vol_shape, initialize_by = "data", \
               points = cpts_all, width = p3d.wd)
    if TIMEIT:
        return x, p3d, np.asarray(times)
    else:
        return x, p3d


def recon_patches_3d(projs, theta, center, p3d, apply_fbp = True, TIMEIT = False):
    '''
    
    Assumes the patches are on a regular (non-overlapping) grid.  
    '''
    
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    
    vol_shape = p3d.vol_shape
    cond0 = projs.shape[1] != vol_shape[0]
    cond1 = projs.shape[-1] != vol_shape[1]
    cond2 = projs.shape[-1] != vol_shape[2]
    if any([cond0, cond1, cond2]):
        raise ValueError("vol_shape and projections array are incompatible")
    
    # assume z-widths are all same
    z_width = p3d.widths[0,0]
    # assume no overlap between patches
    z_points = p3d.points[:,0]
    
    p2d_sorted = Patches(tuple(vol_shape[1:]), initialize_by = "data", \
                  points = p3d.points[:,1:], \
                  widths = p3d.widths[:,1:])
    p2d_sorted.add_features(z_points.reshape(-1,1), names = ["z_points"])
    p2d_sorted = p2d_sorted.sort_by_feature(ife = 0) # sort in increasing z value
    z_points_unique = np.unique(z_points)    

    sub_vols = []
    from tqdm import tqdm
    print("reconstructing selected 3D patches", \
          tuple(p3d.widths[0,...]), \
          "along %i z-chunks"%len(z_points_unique))
    for icount, z_point in enumerate(z_points_unique):
        p2d_z = p2d_sorted.filter_by_condition(p2d_sorted.features[:,0] == z_point)
#         print("index %i, number of patches: %i"%(z_point, len(p2d_z)))  
        sub_vols.append(recon_chunk(projs[:,z_point:z_point+z_width,:], theta, center, p2d_z, apply_fbp = apply_fbp))
        
        widths_arr = np.concatenate([np.ones((len(p2d_z),1))*z_width, p2d_z.widths], axis = 1)
        points_arr = np.concatenate([np.ones((len(p2d_z),1))*z_point, p2d_z.points], axis = 1)
        
        p3d_tmp = Patches(vol_shape, initialize_by = "data", \
                          points = points_arr, widths = widths_arr)
        
        if icount == 0:
            p3d_new = p3d_tmp.copy()
        else:
            p3d_new.append(p3d_tmp)
        del p3d_tmp

    sub_vols = np.concatenate(sub_vols, axis = 0)    
    
    end_gpu.record()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    
    if TIMEIT:
        print("TIME reconstruct 3D patches: %.2f seconds"%(t_gpu/1000.0))
        
    return sub_vols, p3d_new
    
def recon_chunk(projs, theta, center, p2d, apply_fbp = True, TIMEIT = False):
    
    '''
    reconstruct a region within full volume defined by 2d patches with corner points (y, x) and widths (wy, wx) and a height  
    
    Parameters
    ----------
    projs : np.ndarray  
        array of projection images shaped as ntheta, nrows, ncols
    theta : np.ndarray
        array of theta values (length = ntheta)  
    center : float  
        center value for the projection data  
    p2d : Patches  
        patches (2d) on a given slice
    
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
        data = cp.array(projs)
        center = cp.float32(center)    
        theta = cp.array(theta, dtype = 'float32')
        stream_copy.synchronize()
    
    if apply_fbp:
        fbp_filter(data) # need to apply filter to full projection  

    # st* - start, p* - number of points
    stz = 0
    pz = data.shape[1] # this is the chunk size
    sub_vols = []
    for ip in range(len(p2d)):
        sty, stx = p2d.points[ip]
        py, px = p2d.widths[ip]
        
        tmp_rec = rec_patch(data, theta, center, \
                  stx, px, sty, py, stz, pz)
        sub_vols.append(tmp_rec.get())
    
    sub_vols = np.asarray(sub_vols)

    device.synchronize()
#     print('total bytes: ', memory_pool.total_bytes())    
    
    end_gpu.record()
    end_gpu.synchronize()
    t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    
    if TIMEIT:
        print("TIME reconstruct z-chunk: %.2f seconds"%(t_gpu/1000.0))
    
    return sub_vols







if __name__ == "__main__":
    
    print('just a bunch of functions')

    
