#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 

import cupy as cp
import numpy as np
import time
pinned_memory_pool = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)
def pinned_array(array):
        """Allocate pinned memory and associate it with numpy array"""
        mem = cp.cuda.alloc_pinned_memory(array.nbytes)
        src = np.frombuffer(
            mem, array.dtype, array.size).reshape(array.shape)
        src[...] = array
        return src
if __name__ == "__main__":
    arr_cpu = np.zeros((128, 4200, 4200))
    arr_pinned = pinned_array(arr_cpu)
    arr_gpu = cp.array(arr_cpu)#empty_like(arr_cpu)
    print('pinned')
    t00 = time.time()
    s1 = cp.cuda.Stream()
    for k in range(5):
        t00 = time.time()
        with s1:
            arr_gpu.set(arr_pinned)
        s1.synchronize()
        print("run", k, "time ", time.time() - t00)
    print('regular')
    s1 = cp.cuda.Stream()
    for k in range(5):
        t00 = time.time()
        with s1:
            arr_gpu.set(arr_cpu)
        s1.synchronize()
        print("run", k, "time ", time.time() - t00)

