#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""



"""
import numpy as np
import cupy as cp
import time
import h5py
from cupyx.scipy.fft import rfft, irfft, rfftfreq

def darkflat_correction(data, dark, flat):
    """Dark-flat field correction"""
    for k in range(data.shape[0]):
        data[k] = (data[k]-dark)/cp.maximum(flat-dark, 1e-6)
    return data
def minus_log(data):
    """Taking negative logarithm"""
    data = -cp.log(cp.maximum(data, 1e-6))
    return data



source = """
extern "C" {
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
}
"""
module = cp.RawModule(code=source)
rec_kernel = module.get_function('rec')
def rec(data, theta, center, stx, px, sty, py, stz, pz):
    """Reconstruct subvolume [stz:stz+pz,sty:sty+py,stx:stx+px] on GPU"""
    [ntheta, nz, n] = data.shape
    
    obj = cp.zeros([pz, py, px], dtype='float32')
    rec_kernel((int(cp.ceil(px/16)), int(cp.ceil(py/16)), \
                int(cp.ceil(pz/4))), (16, 16, 4), \
               (obj, data, theta, cp.float32(center), \
                ntheta, nz, n, stx, px, sty, py, stz, pz))
    
    return obj



# def fbp_filter(data):
#         '''
#         FBP filtering of projections
#         '''
        
#         ntheta, nz, n = data.shape
        
#         ne = 3*n//2
#         t = cp.fft.rfftfreq(ne).astype('float32')
#         w = t * (1 - t * 2)**3  # parzen
# #         w = w*cp.exp(2*cp.pi*1j*t*(center-n/2)) # center fix
#         data = cp.pad(data,((0,0),(0,0),(ne//2-n//2,ne//2-n//2)),mode='edge')
#         data = irfft(
#             w*rfft(data, axis=2), axis=2).astype('float32')  # note: filter works with complex64, however, it doesnt take much time
#         data = data[:,:,ne//2-n//2:ne//2+n//2]
#         return data
def fbp_filter(projs, nzc = 2):
    """FBP filtering of projections"""
    t = rfftfreq(projs.shape[2])
    wfilter = t #* (1 - t * 2)**3  # parzen
    wfilter = cp.tile(wfilter, [projs.shape[1]//nzc, 1])
    
    # loop over z chunks
    if projs.shape[1]%nzc:
        raise ValueError("height of projection must be divisible by nzc = %i"%nzc)
#     import pdb; pdb.set_trace()
    else:
        wzc = projs.shape[1]//nzc
    
    data = []
    for ic in range(nzc):
        # loop over slices to minimize fft memory overhead
        data.append(_apply_ffilter_to_projs(cp.array(projs[:, ic*wzc:ic*wzc + wzc]), wfilter).get())
    return cp.concatenate(cp.array(data), axis = 1)

def _apply_ffilter_to_projs(data, wfilter):
    for k in range(data.shape[0]):

        data[k] = irfft(\
                     wfilter*rfft(data[k], overwrite_x=True, axis=1), \
                     overwrite_x=True, axis=1)
    cp.cuda.stream.get_current_stream().synchronize()
        
    return data
    
    
def recon_patch(projs, theta, center, point, width, mem_limit_gpu = 5.0, apply_fbp = True, nzc = 2):
    
    '''
    reconstruct a region within full volume shaped as cuboid, defined corner points (z, y, x) and widths (wz, wy, wx).  
    
    Parameters
    ----------
    projs : np.ndarray  
        array of projection images shaped as ntheta, nrows, ncols
    theta : np.ndarray
        array of theta values (length = ntheta)  
    center : float  
        center value for the projection data  
    mem_limit_gpu : float  
        mem limit in GB for GPU  
    point : np.ndarray  
        array of 3 corner points z, y, x  
    width : np.ndarray  
        array of 3 widths wz, wy, wx  
    

    
    Returns
    -------
    
    '''
    
    sz = slice(point[0], point[0] + width[0])
    # first crop the projections along z
    projs = projs[:, sz, :].copy()
    
    
    # make sure the width of projection is divisible by four after padding
    proj_w = projs.shape[-1]
    tot_width = int(proj_w*(1 + 0.25*2)) # 1/4 padding
    tot_width = int(np.ceil(tot_width/8)*8) 
    padding = int((tot_width - proj_w)//2)
    projs = np.pad(projs, ((0,0),(0,0),(padding, padding)), mode = 'edge')
    
    
    # send to gpu; check if memory limit is crossed  
    # Q: what should be memory limit?  
    if projs.nbytes/1.0e9 > mem_limit_gpu:
        raise ValueError("mem limit breached")
    
    stream1 = cp.cuda.Stream(non_blocking=False)
    with stream1:
        theta = cp.array(theta, dtype = 'float32')
        if apply_fbp:
            data = fbp_filter(projs, nzc = nzc) # need to apply filter to full projection  
            print(data.shape)
        else:
            data = cp.array(projs)

        center = cp.float32(center)    
    stream1.synchronize()
    
    stream2 = cp.cuda.Stream(non_blocking=False)
    with stream2:
        # st* - start, p* - number of points
        stz, sty, stx = point
        pz, py, px = width
        st = time.time()
        obj = rec(data, theta, center+padding, \
                  stx+padding, px, \
                  sty+padding, py, \
                  0,           pz) # 0 since projections were cropped vertically
    print(time.time()-st)
    stream2.synchronize()
    return obj.get()

def recon_patches_3d(projs, theta, center, p3d, mem_limit_gpu = 5.0, apply_fbp = True, nzc = 1):
    '''
    
    Assumes the patches are on a regular (non-overlapping) grid.  
    '''
    
    # send to gpu; check if memory limit is crossed  
    # Q: what should be memory limit?  
    if projs.nbytes/1.0e9 > mem_limit_gpu:
        raise ValueError("mem limit breached")
    
    z_idxs = p3d.points[:,0]
    wz = p3d.widths[0,0]
    
    p2d_sorted = Patches(tuple(vol_shape[1:]), initialize_by = "data", \
                  points = p3d.points[:,1:], \
                  widths = p3d.widths[:,1:])
    p2d_sorted.add_features(z_idxs.reshape(-1,1), names = ["z_idx"])

    p2d_sorted = p2d.sort_by_feature(ife = 0)
    z_idxs_unique = np.unique(z_idxs)    

    for z_idx in z_idxs_unique:
        p2d_z = p2d.filter_by_condition(p2d.features[:,0] == z_idx)
        print("index %i, number of patches: %i"%(z_idx, len(p2d_z)))  
        sub_vols = recon_chunk(projs, theta, center, p2d_z, apply_fbp = True)
        #TO-DO return both sub_vols and p3d?
        
    raise NotImplementedError("WIP")
    return None
    
    
    
def recon_chunk(projs, theta, center, p2d, apply_fbp = True, nzc = 4):
    
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
    mem_limit_gpu : float  
        mem limit in GB for GPU  
    

    
    Returns
    -------
    
    '''
    
    # make sure the width of projection is divisible by four after padding
    proj_w = projs.shape[-1]
    tot_width = int(proj_w*(1 + 0.25*2)) # 1/4 padding
    tot_width = int(np.ceil(tot_width/8)*8) 
    padding = int((tot_width - proj_w)//2)
    projs = np.pad(projs, ((0,0),(0,0),(padding, padding)), mode = 'edge')
    
    theta = cp.array(theta, dtype = 'float32')
    if apply_fbp:
        data = fbp_filter(projs, nzc = nzc) # need to apply filter to full projection  
        print(data.shape)
    else:
        data = cp.array(projs)

    center = cp.float32(center)    
    
    # st* - start, p* - number of points
    stz = 0
    pz = projs.shape[1]
    sub_vols = []
    for ip in range(len(p2d)):
        sty, stx = p2d.points[ip]
        py, px = p2d.widths[ip]
        
        sub_vols.append(rec(data, theta, center+padding, \
                  stx+padding, px, sty+padding, py, stz, pz).get())
        cp.cuda.stream.get_current_stream().synchronize()
    
    
    return sub_vols








if __name__ == "__main__":
    
    print('just a bunch of functions')

    
