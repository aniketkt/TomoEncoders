#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import sys 
import matplotlib.pyplot as plt 
import numpy as np 
import cupy as cp 
import time 
import os 
import tensorflow as tf
import vedo

######## START GPU SETTINGS ############
########## SET MEMORY GROWTH to True ############
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass        
######### END GPU SETTINGS ############

# /data02/MyArchive/aisteer_3Dencoders/TomoEncoders/example_workflows/sparse_segmentation/tests/streaming_demo

### READING DATA
PROJS_PATH = '/data02/MyArchive/AM_part_Xuan/projs' 
read_fpath = '/data02/MyArchive/AM_part_Xuan/data/mli_L206_HT_650_L3_rec_1x1_uint16.hdf5' 
from tomo_encoders.tasks.sparse_segmenter.sparse_segmenter import normalize_volume_gpu
from tomo_encoders import DataFile, Patches
import h5py 
import pdb

### DETECTOR / RECONSTRUCTION
DET_BINNING = 4 # detector binning # ONLY VALUES THAT DIVIDE 64 into whole parts.
THETA_BINNING = 4
DET_NTHETA = 2000
DET_FOV = (1920,1000)
DET_PNZ = 4 # may speed up projection compututation (or simulation of acquisition)
from tomo_encoders.tasks.sparse_segmenter.recon import recon_patches_3d
from tomo_encoders.tasks.sparse_segmenter.project import acquire_data, read_data
from tomo_encoders.tasks.sparse_segmenter.detect_voids import to_regular_grid

### SEGMENTATION
sys.path.append('../trainer')
from params import *
from tomo_encoders.tasks import SparseSegmenter
INF_INPUT_SIZE = (64,64,64)
INF_CHUNK_SIZE = 32 # full volume
NORM_SAMPLING_FACTOR = 4
model_name = "M_a02_64-64-64"

### VOID ANALYSIS
N_MAX_DETECT = 25 # 3 for 2 voids - first one is surface
N_SURFACES = 1

### VISUALIZATION
from tomo_encoders.misc_utils.feature_maps_vis import view_midplanes 
demo_out_path = '/data02/MyArchive/AM_part_Xuan/demo_output'
plot_out_path = '/home/atekawade/Dropbox/Arg/transfers/runtime_plots/'
#import matplotlib as mpl
#mpl.use('Agg')


### TIMING
TIMEIT_lev1 = True
TIMEIT_lev2 = False



from utils import VoidDetector
def calc_vol_shape(projs_shape):
    ntheta, nz, nx = projs_shape
    return (nz, nx, nx)

def process_data(projs, theta, center, fe, vs, DIGITAL_ZOOM = False):

    t000 = time.time()
    print("\n\nSTART PROCESSING\n\n")

    PROJS_SHAPE_1 = projs.shape
    VOL_SHAPE_1 = calc_vol_shape(PROJS_SHAPE_1)
    print("projections shape: ", PROJS_SHAPE_1)
    print("reconstructed volume shape: ", VOL_SHAPE_1)
    vol_rec_b = vs.reconstruct(projs, theta, center)
    vol_seg_b = vs.segment(vol_rec_b, fe)
    
    fig, ax = plt.subplots(1, 3, figsize = (8,4))
    view_midplanes(vol = fe.rescale_data(vol_rec_b, \
                                         np.min(vol_rec_b), \
                                         np.max(vol_rec_b)), ax = ax)
    view_midplanes(vol = vol_seg_b, cmap = 'copper', alpha = 0.3, ax = ax)
    plt.savefig(os.path.join(plot_out_path, "vols_b_%s.png"%model_name))
    plt.show()
    plt.close()

    ##### VOID DETECTION STEP ############
    sub_vols_voids_b, p3d_voids_b, vol_surf, vol_voids = vs.export_voids(vol_seg_b)
    
#     vol_voids_surf = vedo.Volume(vol_voids + vol_surf)
    vol_voids_surf = vedo.Volume(vol_voids + 0)
    surf = vol_voids_surf.isosurface(0.5).smooth().subdivide()
    print("READY FOR SCENE OF PORE MORPHOLOGY")
    vedo.show(surf, bg = 'wheat', bg2 = 'lightblue')
    
    
    
    if DIGITAL_ZOOM:
        p3d_grid_1_voids = to_regular_grid(sub_vols_voids_b, \
                                           p3d_voids_b, \
                                           INF_INPUT_SIZE, \
                                           VOL_SHAPE_1, \
                                           DET_BINNING)


        sub_vols_grid_voids_1 = vs.reconstruct_patches(projs, theta, center, \
                                                       p3d_grid_1_voids)

        vol_seg_1 = np.ones(VOL_SHAPE_1, dtype = np.uint8)
        vs.segment_patches_to_volume(sub_vols_grid_voids_1, p3d_grid_1_voids, vol_seg_1, fe)
        print("vol_seg_1 shape: ", vol_seg_1.shape)

        fig, ax = plt.subplots(1, 3, figsize = (8,4))
        view_midplanes(vol = vol_seg_1, cmap = 'copper', ax = ax)
        plt.savefig(os.path.join(plot_out_path, "vols_1_%s.png"%model_name))
        print("TOTAL TIME ELAPSED: %.2f seconds"%(time.time() - t000))
        plt.show()
        plt.close()

    #     vol_voids_zoom = vedo.Volume(vol_seg_1)
    #     surf = vol_voids_zoom.isosurface(0.5).smooth().subdivide()
    #     print("READY FOR DIGITAL ZOOM")
    #     import pdb; pdb.set_trace()
    #     vedo.show(surf, bg = 'wheat', bg2 = 'lightblue')
    
    
    return



if __name__ == "__main__":

    
    model_names = {"segmenter" : "segmenter_Unet_%s"%model_name}

    print("#"*55, "\nWorking on model %s\n"%model_name, "#"*55)
    fe = SparseSegmenter(model_initialization = 'load-model', \
                         model_names = model_names, \
                         model_path = model_path, \
                         input_size = INF_INPUT_SIZE)    
    fe.test_speeds(INF_CHUNK_SIZE)
    
    ds = DataFile(read_fpath, data_tag = 'data', tiff = False, VERBOSITY = 0)
    vol = ds.read_full().astype(np.float32)
    vol = normalize_volume_gpu(vol, normalize_sampling_factor = NORM_SAMPLING_FACTOR, chunk_size = 1)
    
    iter_count = 0
    while True:
        print("\n\n", "#"*55, "\n")
        print("ITERATION %i: \n"%iter_count, vol.shape)
        print("\nDOMAIN SHAPE: ", vol.shape)

#         point = (550, 2000, 1800)
        point = (550, 2100, 2100)
        print("READY TO NAVIGATE TO NEXT LOCATION")        
        print("CURRENT LOCATION: ", point)
        pdb.set_trace()        
        # Read if already written
        projs, theta, center = read_data(PROJS_PATH, \
                                         point, \
                                         DET_NTHETA, \
                                         FOV = DET_FOV)    
        if projs is None:
            projs, theta, center = acquire_data(vol, PROJS_PATH, point, \
                                                DET_NTHETA, \
                                                FOV = DET_FOV, \
                                                pnz = DET_PNZ)    
        
        vs = VoidDetector(THETA_BINNING, DET_BINNING, \
                          INF_INPUT_SIZE, INF_CHUNK_SIZE,\
                          N_MAX_DETECT, N_SURFACES, \
                          TIMEIT_lev1 = TIMEIT_lev1,\
                          TIMEIT_lev2 = TIMEIT_lev2,\
                          NORM_SAMPLING_FACTOR = NORM_SAMPLING_FACTOR)
        
        process_data(projs, theta, center, fe, vs)
        
        iter_count += 1
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    