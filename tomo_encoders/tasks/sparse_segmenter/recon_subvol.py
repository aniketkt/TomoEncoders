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
    rec_kernel((int(cp.ceil(px/16)), int(cp.ceil(py/16)), int(cp.ceil(pz/4))), (16, 16, 4),
                  (obj, data, theta, cp.float32(center), ntheta, nz, n, stx, px, sty, py, stz, pz))
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
def fbp_filter(data):
    """FBP filtering of projections"""
    t = rfftfreq(data.shape[2])
    wfilter = t #* (1 - t * 2)**3  # parzen
    wfilter = cp.tile(wfilter, [data.shape[1], 1])
    # loop over slices to minimize fft memory overhead
    for k in range(data.shape[0]):
        data[k] = irfft(
            wfilter*rfft(data[k], overwrite_x=True, axis=1), overwrite_x=True, axis=1)
    return data

    
def recon_patch(projs, theta, center, point, width, mem_limit_gpu = 5.0, apply_fbp = True):
    
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
    
    
    # TO-DO: for loop in z-direction
    
    # send to gpu; check if memory limit is crossed  
    # Q: what should be memory limit?  
    if projs.nbytes/1.0e9 > mem_limit_gpu:
        raise ValueError("mem limit breached")
    else:
        data = cp.array(projs)
        theta = cp.array(theta, dtype = 'float32')
        if apply_fbp:
            data = fbp_filter(data) # need to apply filter to full projection  
            data = data #- cp.mean(data, axis = (0,2))
        print(data.shape)
        center = cp.float32(center)    
    
    # st* - start, p* - number of points
    stz, sty, stx = point
    pz, py, px = width
    st = time.time()
    obj = rec(data, theta, center+padding, \
              stx+padding, px, \
              sty+padding, py, \
              0,           pz) # 0 since projections were cropped vertically
    cp.cuda.stream.get_current_stream().synchronize()
    print(time.time()-st)
    
    return obj.get()


if __name__ == "__main__":
    
    print('just a bunch of functions')

    
