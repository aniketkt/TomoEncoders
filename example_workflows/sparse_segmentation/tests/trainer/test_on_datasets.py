#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import sys
from tomo_encoders import Patches
from tomo_encoders import DataFile
import tensorflow as tf
import time
from tomo_encoders.tasks import SparseSegmenter

from tomo_encoders.misc_utils.feature_maps_vis import view_midplanes
from tomo_encoders.misc_utils.img_stats import calc_jac_acc, calc_SNR, Parallelize
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
save_path = '/home/atekawade/Dropbox/Arg/transfers/model_history'

descriptor_tag = 'tmp'#'test_noblanks_pt2cutoff_nostd'
test_binning = 2
from datasets import get_datasets, dataset_names
from params import *
n_samples = 1000
model_names = {"segmenter" : "segmenter_Unet_%s"%descriptor_tag}


def infer(fe, Xs, Ys):
    return

if __name__ == "__main__":

    fe = SparseSegmenter(model_initialization = 'load-model', \
                         model_names = model_names, \
                         model_path = model_path)    
    
    datasets = get_datasets(dataset_names[:1], test_binning = test_binning)
    
    Xs, Ys = fe.load_datasets(datasets, TIMEIT = True)
    
    for ii, key in enumerate(datasets):
        print("working dataset: ", key)
        print("shape : ", Xs[ii].shape)
    
    
    # SAVE PATH
    history_path = os.path.join(save_path, descriptor_tag)
    if not os.path.exists(history_path):
        os.makedirs(history_path)
    
    # TEST ONE VOLUME at idx = 0
    patches = fe.get_patches(Xs[0].shape, "regular-grid", 1, n_samples, \
                             cutoff = 0.3, Y_gt = Ys[0])
    
    x = patches.extract(Xs[0], fe.model_size)
    y = patches.extract(Ys[0], fe.model_size)
    x = x[...,np.newaxis]
    y = y[...,np.newaxis]
    
    min_max = fe.calc_voxel_min_max(Xs[0], 4, TIMEIT = True)
    y_pred, tot_time = fe._predict_patches(x, 32, None, \
                                           min_max = min_max, \
                                           TIMEIT = True)
    
    y_pred = np.round(y_pred[...,0]).astype(np.uint8)
    y = y[...,0]
    
    # plot some images and save
    ii = 800
    fig, ax = plt.subplots(2,3, figsize = (8,6))
    view_midplanes(vol = fe.rescale_data(x[ii], *min_max), ax = ax[0])
    view_midplanes(vol = fe._edge_map(y_pred[ii]), \
                   cmap = 'copper', alpha = 0.3, ax = ax[0])
    view_midplanes(vol = fe.rescale_data(x[ii], *min_max), ax = ax[1])
    view_midplanes(vol = fe._edge_map(y[ii]),      \
                   cmap = 'copper', alpha = 0.3, ax = ax[1])
    plt.savefig(os.path.join(history_path, "patch_midplane.png"))
    
    # ACCURACY USING JACCARD METRIC - to-do: put this into a function within fe
    t0 = time.time()
    IoU = Parallelize(list(zip(y, y_pred)), calc_jac_acc, procs = 12)
    SNR = Parallelize(list(zip(y, y_pred)), calc_SNR, procs = 12)
    ystd = np.std(y, axis = (1,2,3))
    void_frac = np.sum(y, axis = (1,2,3))/np.prod(fe.model_size)
    print("time %.2f seconds"%(time.time() - t0))    
    
    df = pd.DataFrame(columns = ["SNR", "IoU", "ystd", "void_frac"], \
                      data = np.asarray([SNR, IoU, ystd, void_frac]).T)    
    df.to_csv(os.path.join(history_path, "stats.csv"), index = False)
        
    print("Done")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
