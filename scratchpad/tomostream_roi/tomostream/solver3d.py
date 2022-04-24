'''

Adaption of Viktor Nikitin's code Tomostream (https://github.com/nikitinvv/tomostream) for doing full 3d reconstructions.  

'''


import cupy as cp
import numpy as np
from cupyx.scipy.fft import rfft, irfft
from cupyx.scipy import ndimage
import signal
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import time


detect_flag = False
UPDATE_TIME_INTERVAL = 60.0 # seconds
from change_detector import change_detector
ANOMALY_DETECT_FLAG = False


mpl.use('Agg')

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

def backproject3D(data, theta, center):
    """Reconstruct subvolume [stz:stz+pz,sty:sty+py,stx:stx+px] on GPU"""
    
    #up_nz, up_n = data.shape[1:] #CHECKBIN#
    #data = data[:, ::2, ::2].copy() #CHECKBIN#
    #center = center/2 #CHECKBIN#
    
    stx, sty, stz = (0,0,0)
    pz, py, px = (data.shape[1], data.shape[2], data.shape[2])
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
    
    
    #obj_up = cp.zeros([up_nz, up_n, up_n], dtype = 'float32', order = 'C') #CHECKBIN#
    #obj_up[:pz,:py,:px] = obj #CHECKBIN#
    #obj = obj_up #CHECKBIN#

    if 0:
        print("TIME backproject 3D: %.2f ms"%t_gpu)
    
    return obj


class Solver():
    """Class for tomography reconstruction of ortho-slices through direct 
    discreatization of circular integrals in the Radon transform.

    Parameters
    ----------
    ntheta : int
        The number of projections in the buffer (for simultaneous reconstruction)
    n, nz : int
        The pixel width and height of the projection.
    """

    def __init__(self, ntheta, n, nz, center, idx, idy, idz, rotx, roty, rotz, fbpfilter, dezinger, data_type):
        
        self.n = n
        self.nz = nz
        self.ntheta = ntheta
        
        # GPU storage for dark and flat fields
        self.dark = cp.array(cp.zeros([nz, n]), dtype='float32')
        self.flat = cp.array(cp.ones([nz, n]), dtype='float32')
        # GPU storages for the projection buffer, ortho-slices, and angles
        self.data = cp.zeros([ntheta, nz, n], dtype=data_type)  
        self.obj = cp.zeros([n, 3*n], dtype='float32')# ortho-slices are concatenated to one 2D array
        self.theta = cp.zeros([ntheta], dtype='float32')
        
        self.rec_vol = cp.zeros([nz, n, n], dtype = 'float32')
        if ANOMALY_DETECT_FLAG:
            self.rec_vol_prev = None
        else:
            self.rec_vol_prev = cp.zeros([nz, n, n], dtype = 'float32')
        
        self.roi_pt = np.asarray([nz//2, n//2, n//2])

        self.last_update_time = time.time()
        # reconstruction parameters
 
        self.idx = np.int32(idx)
        self.idy = np.int32(idy)
        self.idz = np.int32(idz)
        self.rotx = np.float32(rotx/180*np.pi)
        self.roty = np.float32(roty/180*np.pi)
        self.rotz = np.float32(rotz/180*np.pi)
        self.center = np.float32(center)     
        self.fbpfilter = fbpfilter         
        self.dezinger = dezinger

        # flag controlling appearance of new dark and flat fields   
        self.new_dark_flat = False
    
    def free(self):
        """Free GPU memory"""

        cp.get_default_memory_pool().free_all_blocks()

    def set_dark(self, data):
        """Copy dark field (already averaged) to GPU"""

        self.dark = cp.array(data.astype('float32'))        
        self.new_dark_flat = True
    
    def set_flat(self, data):
        """Copy flat field (already averaged) to GPU"""

        self.flat = cp.array(data.astype('float32'))
        self.new_dark_flat = True
    
    def fbp_filter(self, data):
        """FBP filtering of projections"""

        t = cp.fft.rfftfreq(self.n)
        if (self.fbpfilter=='Parzen'):
            wfilter = t * (1 - t * 2)**3    
        elif (self.fbpfilter=='Ramp'):
            wfilter = t
        elif (self.fbpfilter=='Shepp-logan'):
            wfilter = np.sin(t)
        elif (self.fbpfilter=='Butterworth'):# todo: replace by other
            wfilter = t / (1+pow(2*t,16)) # as in tomopy

        wfilter = cp.tile(wfilter, [self.nz, 1])    
        for k in range(data.shape[0]):# work with 2D arrays to save GPU memory
            data[k] = irfft(
                wfilter*rfft(data[k], overwrite_x=True, axis=1), overwrite_x=True, axis=1)
        return data

    def darkflat_correction(self, data):
        """Dark-flat field correction"""

        for k in range(data.shape[0]):# work with 2D arrays to save GPU memory
            data[k] = (data[k]-self.dark)/cp.maximum(self.flat-self.dark, 1e-6)
        return data

    def minus_log(self, data):
        """Taking negative logarithm"""

        data = -cp.log(cp.maximum(data, 1e-6))
        return data
    
    def remove_outliers(self, data):
        """Remove outliers"""
        if(int(self.dezinger)>0):
            r = int(self.dezinger)            
            fdata = ndimage.median_filter(data,[1,r,r])
            ids = cp.where(cp.abs(fdata-data)>0.5*cp.abs(fdata))
            data[ids] = fdata[ids]        
        return data

    def detect_roi(self):
        '''
        detect changes from previous and current volume
        '''
        # change detection
        if ANOMALY_DETECT_FLAG:
            change_locations = change_detector(None, self.rec_vol.get(), self.dn_model)    
        else:
            change_locations = change_detector(self.rec_vol_prev.get(), self.rec_vol.get(), self.dn_model)
        # output     is (n_pts, 3) where iz, iy, ix are returned            
        # note: n_pts is hard-code to be '1' here
        roi_pt = change_locations[0] # to-do: what to do with these?
        print(roi_pt)
        assert len(roi_pt) == 3, "roi_pt must be three integer values as tuple, list or nd array"

        return roi_pt        
    
    def update_now(self, data, theta):
        """Reconstruction with the standard processing pipeline on GPU"""

        data = data.astype('float32')
        data = self.darkflat_correction(data)
        data = self.remove_outliers(data)
        data = self.minus_log(data)
        data = self.fbp_filter(data)

        self.rec_vol = backproject3D(data, theta*np.pi/180, self.center)
        #min_val = self.rec_vol.min()
        #max_val = self.rec_vol.max()
        #self.rec_vol = (self.rec_vol - min_val) / (max_val - min_val)
        
        time_elapsed = time.time() - self.last_update_time
        is_time_to_update = time_elapsed > UPDATE_TIME_INTERVAL
        if is_time_to_update:
            roi_pt = self.detect_roi()
            self.roi_pt = roi_pt
            obj = self.draw_obj()
            #time.sleep(10.0) # to-do is this necessary?
            self.rec_vol_prev = self.rec_vol.copy()
            self.last_update_time = time.time()            
            print("#"*30)
            print(f"ROI POINT: {roi_pt}")
        else:
            obj = self.draw_obj()
        
        return obj

    def draw_obj(self):
        '''
        input rec_vol is a numpy array (not cupy!)
        roi_pt is [idz, idy, idx]
        '''
        
        _idxs = self.roi_pt
        #_idxs = [self.rec_vol.shape[i]//2 for i in range(3)]
        imgz, imgy, imgx = [self.rec_vol.take(_idxs[i], axis = i) for i in range(3)]        
        #assert np.std(imgz) > 0.0, "imgz is empty"
        
        obj = cp.zeros([self.n, 3*self.n], dtype='float32')
        obj[:self.n,         :self.n  ] = cp.array(imgz)
        obj[:self.nz, self.n  :2*self.n] = cp.array(imgy)
        obj[:self.nz , 2*self.n:3*self.n] = cp.array(imgx)
        #obj /= self.ntheta
        
        return obj

    def recon_optimized(self, data, theta, ids, center, idx, idy, idz, rotx, roty, rotz, fbpfilter, dezinger, dbg=False):
        """Optimized reconstruction of the object
        from the whole set of projections in the interval of size pi.
        Resulting reconstruction is obtained by replacing the reconstruction part corresponding to incoming projections, 
        objnew = objold + recon(datanew) - recon(dataold) whenever the number of incoming projections is less than half of the buffer size.
        Reconstruction is done with using the whole buffer only when: the number of incoming projections is greater than half of the buffer size,
        idx/idy/idz, center, fbpfilter are changed, or new dark/flat fields are acquired.

        Parameters
        ----------
        data : np.array(nproj,nz,n)
            Projection data 
        theta : np.array(nproj)
            Angles corresponding to the projection data
        ids : np.array(nproj)
            Ids of the data in the circular buffer array
        center : float
            Rotation center for reconstruction            
        idx, idy, idz: int
            X-Y-Z ortho slices for reconstruction
        rotx, roty, rotz: float
            Rotation angles for X-Y-Z slices
        fbpfilter: str
            Reconstruction filter
        dezinger: str
            None or radius for removing outliers

        Return
        ----------
        obj: np.array(n,3*n) 
            Concatenated reconstructions for X-Y-Z orthoslices
        """
 
        idx = np.int32(idx)
        idy = np.int32(idy)
        idz = np.int32(idz)
        rotx = np.float32(rotx/180*np.pi)
        roty = np.float32(roty/180*np.pi)
        rotz = np.float32(rotz/180*np.pi)
        center = np.float32(center)       
        
        # update data in the buffer
        self.data[ids] = cp.array(data.reshape(data.shape[0], self.nz, self.n))
        self.theta[ids] = cp.array(theta.astype('float32'))        
        self.idx = idx
        self.idy = idy
        self.idz = idz
        self.rotx = rotx
        self.roty = roty
        self.rotz = rotz
        self.center = center
        self.fbpfilter = fbpfilter
        self.dezinger = dezinger
        self.new_dark_flat = False

        self.obj = self.update_now(self.data, self.theta)

        return  self.obj.get()

