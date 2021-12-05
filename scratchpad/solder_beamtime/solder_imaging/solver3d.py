'''

Adaption of Viktor Nikitin's code Tomostream (https://github.com/nikitinvv/tomostream) for doing full 3d reconstructions.  

'''


import cupy as cp
import numpy as np
from cupyx.scipy.fft import rfft, irfft
from cupyx.scipy import ndimage
from tomostream import kernels
from tomostream import util
import signal
import sys
from noise2self import *
from tomo_encoders.reconstruction.recon import rec_patch
import matplotlib.pyplot as plt
import matplotlib as mpl


detect_flag = False
detect_pts = 100
from tsne_detector import change_detector

mpl.use('Agg')
full_flag = True
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
        if full_flag:
            self.rec_vol = cp.zeros([nz, n, n], dtype = 'float32')
        else:
            self.rec_vol = None
        
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
    
    def ortho_backprojection(self, data, theta):
        """Compute backprojection to orthogonal slices"""

        # Zhengchun's denoiser
        denoise_flag = False 

        if denoise_flag:
            data_in = data.get() # move to cpu
            
            for i in range(len(data_in)):
                if i > 800: break
                proj_frame = data_in[i]
                ht, wd = proj_frame.shape
                new_ht = int(8*np.ceil(ht/8))
                new_wd = int(8*np.ceil(wd/8))
                pad_ht = new_ht - ht
                pad_wd = new_wd - wd
                proj_frame = np.pad(proj_frame, ((pad_ht, 0), (pad_wd, 0)), mode = 'edge')
#                proj_frame = np.random.rand(proj_frame.shape).astype(np.float32)
                proj_frame = denoise_one_frame(self.dn_model, proj_frame)
                proj_frame = proj_frame[pad_ht:, pad_wd:]
                data_in[i] = proj_frame
            data = cp.array(data_in) # back to gpu
        
        self.rec_vol = None
        obj = cp.zeros([self.n, 3*self.n], dtype='float32') # ortho-slices are concatenated to one 2D array
        obj[:self.nz,         :self.n  ] = kernels.orthox(data, theta, self.center, self.idx, self.rotx)
        obj[:self.nz, self.n  :2*self.n] = kernels.orthoy(data, theta, self.center, self.idy, self.roty)
        obj[:self.n , 2*self.n:3*self.n] = kernels.orthoz(data, theta, self.center, self.idz, self.rotz)
        obj /= self.ntheta
        return obj

    def full_backprojection(self, data, theta):
        stx, sty, stz = (0,0,0)
        pz, py, px = (data.shape[1], data.shape[2], data.shape[2])

#         assert data.shape[0] == 800, "DATA SHAPE is not 800"

        self.rec_vol = rec_patch(data, theta, self.center, \
                            stx, px, sty, py, stz, pz, \
                            TIMEIT = False)
        min_val = self.rec_vol.min()
        max_val = self.rec_vol.max()
        self.rec_vol = (self.rec_vol - min_val) / (max_val - min_val)
        
        # put the tsne code
        if detect_flag:
            rec_vol, change_locations = change_detector(self.rec_vol.get(), n_pts = detect_pts)
            self.rec_vol = cp.array(rec_vol)
            # output     is (n_pts, 3) where iz, iy, ix are returned            
        
        assert self.rec_vol.shape == (data.shape[1], data.shape[2], data.shape[2])

        _idxs = [self.rec_vol.shape[i]//2 for i in range(3)]
        imgz, imgy, imgx = [self.rec_vol.take(_idxs[i], axis = i) for i in range(3)]        
        assert np.std(imgz) > 0.0, "imgz is empty"
        obj = cp.zeros([self.n, 3*self.n], dtype='float32')
        plt.imshow(imgz.get())
        plt.savefig("/home/beams9/7BMB/solder_beamtime/solder_imaging/tmp_imgz.png")
        plt.close()
 
        obj[:self.nz,         :self.n  ] = imgx
        obj[:self.nz, self.n  :2*self.n] = imgy
        obj[:self.n , 2*self.n:3*self.n] = imgz
        obj /= self.ntheta
        return obj
    
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

    def recon(self, data, theta):
        """Reconstruction with the standard processing pipeline on GPU"""

        data = data.astype('float32')
        data = self.darkflat_correction(data)
        data = self.remove_outliers(data)
        data = self.minus_log(data)
        data = self.fbp_filter(data)

        if full_flag:
            obj = self.full_backprojection(data, theta*np.pi/180)
            return obj
        else:
            obj = self.ortho_backprojection(data, theta*np.pi/180)
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
        
        # recompute only by replacing a part of the data in the buffer, or by using the whole buffer
        recompute_part = False        
        if(recompute_part):            
            # subtract old part
            if full_flag:
                old_rec_vol = self.rec_vol
            self.obj -= self.recon(self.data[ids], self.theta[ids])    
            if full_flag:
                old_rec_vol -= self.rec_vol
        
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

        if recompute_part:
            self.obj += self.recon(self.data[ids], self.theta[ids])   
            if full_flag:
                old_rec_vol += self.rec_vol
                self.rec_vol = old_rec_vol
        else:
            self.obj = self.recon(self.data, self.theta)

        return  self.obj.get()

