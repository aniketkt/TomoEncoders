#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cupy as cp
from tomo_encoders.tasks.sparse_segmenter.recon import rec_patch

if __name__ == "__main__":
    import time
    
    data = cp.ones((100,1800,1500)) # (nz, ntheta, ncols)
    print("data shape: ", data.shape)
    theta = cp.linspace(0, cp.pi, 1800)
    center = cp.float32(900)
    
    for z_size in [1, 5, 10, 20, 30, 60, 100]:
        print("z_size: %i"%z_size)
        out = rec_patch(data, theta, center, 0, 1250, 0, 1250, 0, z_size, TIMEIT = True)
        # ix, ixstep, iy, iystep, iz, izstep
#        print("output shape: ", out.shape)
