import numpy as np
from skimage import io
import time
from skimage.morphology import convex_hull_image
from skimage import measure as ms
from scipy.spatial.distance import pdist
import itertools
import os

import sys
sys.path.append('/data01/AMPolyCalc/code')
from rw_utils import read_raw_data_1X, save_path
from recon import recon_slice, recon_binned
from void_mapping import void_map_gpu
from params import pixel_size_1X as pixel_size
from tomo_encoders.misc.voxel_processing import edge_map
from scipy.spatial import ConvexHull

def max_feret_dm(x_void):
    st = time.time()
    # im_arrs_convexhull = convex_hull_image(x_void)
    verts = np.asarray(np.where(edge_map(x_void))).T
    verts = verts[ConvexHull(verts).vertices]
    en = time.time()
    #print("Convex Hull:", en-st)
    # verts = np.asarray(np.where(edge_map(im_arrs_convexhull))).T
    # im_arrs_convexhull = convex_hull_image(x_void)
    # im_arrs_convexhull = np.pad(im_arrs_convexhull, 2, mode='constant', constant_values = 0)
    # verts, faces, normals, values = ms.marching_cubes(im_arrs_convexhull, 0.5)
    dist = pdist(verts, 'euclidean')

    idx = np.argmax(dist)
    feret_dm = dist[idx]
    coord = list(itertools.combinations(verts,2))[idx]
    vect = coord[1]-coord[0]

    return (feret_dm, vect)

if __name__ == "__main__":
    projs, theta, center, dark, flat = read_raw_data_1X("1", "1")
    b = 4
    voids_4 = void_map_gpu(projs, theta, center, dark, flat, b, pixel_size)
    x_voids = voids_4["x_voids"]

    info = []
    meta_list = []
    start = time.time()
    for i in range(len(x_voids)):
        st = time.time()
        #print(f'shape: {x_voids[i].shape}; volume enc: {np.sum(x_voids[i])}')
        info.append(max_feret_dm(x_voids[i]))
        en = time.time()
        meta_info = [en-st, x_voids[i].shape, np.sum(x_voids[i])]
        meta_list.append(meta_info)
        # print(f'info: {info}')
        # print(f'time: {en-st} secs')
        # print('\n' + '#'*55)

    end = time.time()
    print("Time:", end-start)
    

