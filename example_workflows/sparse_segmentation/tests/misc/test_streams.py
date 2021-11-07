#!/usr/bin/env python

"""
stream_example.py
Inefficiently calculates a matrix power through repeated matrix multiplication.  
"""

import numpy as np
import cupy
import sys
import time

def main(N, power):
    compute_stream = cupy.cuda.stream.Stream(non_blocking=True)

    with compute_stream:
        d_mat = cupy.random.randn(N*N, dtype=cupy.float64).reshape(N, N)
        d_ret = d_mat

        cupy.matmul(d_ret, d_mat)

        start_time = time.time()
        for i in range(power - 1):
            d_ret = cupy.matmul(d_ret, d_mat)
        end_time = time.time()
        print(f"Time spent on cupy.matmul for loop: {end_time - start_time}")

        start_time = time.time()
        compute_stream.synchronize()
        end_time = time.time()
        print(f"Time spent compute_stream.synchronize(): {end_time - start_time}")

if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))