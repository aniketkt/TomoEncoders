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
from tomo_encoders.tasks import SparseSegmenter, load_dataset_pairs


import seaborn as sns
import pandas as pd

from tomo_encoders.misc_utils.feature_maps_vis import view_midplanes
from tomo_encoders.misc_utils.img_stats import calc_jac_acc, calc_SNR, Parallelize
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
save_path = '/home/atekawade/Dropbox/Arg/transfers/model_history'

test_binning = 2
cutoff = 0.2
from datasets import get_datasets, dataset_names
from params import *
n_samples = 3000
cpu_procs = 1

INFERENCE_INPUT_SIZE = (256,256,256)
CHUNK_SIZE = 2
# model_tags = ["M_a02", "M_a04", "M_a05", "M_a01", "M_a03"]
model_tags = ["M_a02"] #["M_a01", "M_a02"]


def infer(fe, Xs, Ys):
    return



if __name__ == "__main__":

    print("\n", "#"*55, "\nTest on dataset pairs")
    datasets = get_datasets(dataset_names[1:], test_binning = test_binning)
    Xs, Ys = load_dataset_pairs(datasets, TIMEIT = True)
    for ii, key in enumerate(datasets):
        print("working dataset: ", key)
        print("shape : ", Xs[ii].shape)

    # TEST ONE VOLUME
        
    for im, model_tag in enumerate(model_tags):
        model_params = get_model_params(model_tag)
        model_names = {"segmenter" : "segmenter_Unet_%s"%model_tag}
        
        print("#"*55, "\nWorking on model %s\n"%model_tag, "#"*55)
        fe = SparseSegmenter(model_initialization = 'load-model', \
                             model_names = model_names, \
                             model_path = model_path, \
                             input_size = INFERENCE_INPUT_SIZE)    
        
        fe.test_speeds(CHUNK_SIZE)
        
        
        if im == 0:
            patches = fe.get_patches(Xs[0].shape, "random-fixed-width", n_samples, \
                                     cutoff = cutoff, Y_gt = Ys[0])

            x = patches.extract(Xs[0], fe.input_size)
            y = patches.extract(Ys[0], fe.input_size)
            min_max = fe.calc_voxel_min_max(Xs[0], 4, TIMEIT = True)
        
        
        
        # SAVE PATH
        history_path = os.path.join(save_path, model_tag)
        if not os.path.exists(history_path):
            os.makedirs(history_path)
        
        y_pred, tot_time = fe.predict_patches(x[...,np.newaxis], CHUNK_SIZE, None, \
                                               min_max = min_max, \
                                               TIMEIT = True)
        y_pred = np.round(y_pred[...,0]).astype(np.uint8)

        
        # At this point y and y_pred should be similarly shaped arrays
        assert y.shape == y_pred.shape, "y and y_pred are not similar shapes"
        
        for ii in np.random.randint(0, n_samples, 20):
            # plot some images and save
            fig, ax = plt.subplots(2,3, figsize = (8,6))
            view_midplanes(vol = fe.rescale_data(x[ii], *min_max), ax = ax[0])
            view_midplanes(vol = fe._edge_map(y_pred[ii]), \
                           cmap = 'copper', alpha = 0.3, ax = ax[0])
            view_midplanes(vol = fe.rescale_data(x[ii], *min_max), ax = ax[1])
            view_midplanes(vol = fe._edge_map(y[ii]),      \
                           cmap = 'copper', alpha = 0.3, ax = ax[1])
            plt.savefig(os.path.join(history_path, "patch_midplane_idx%04d.png"%ii))
            plt.close()

        # ACCURACY USING JACCARD METRIC - to-do: put this into a function within fe

        t0 = time.time()
        IoU = Parallelize(list(zip(y, y_pred)), calc_jac_acc, procs = cpu_procs)
        SNR = Parallelize(list(zip(y, y_pred)), calc_SNR, procs = cpu_procs)
        ystd = np.std(y, axis = (1,2,3))
        void_frac = np.sum(y, axis = (1,2,3))/np.prod(fe.input_size)
        print("time %.2f seconds"%(time.time() - t0))    

        df = pd.DataFrame(columns = ["SNR", "IoU", "ystd", "void_frac"], \
                          data = np.asarray([SNR, IoU, ystd, void_frac]).T)    
        df.to_csv(os.path.join(history_path, "stats.csv"), index = False)
        
        sns.jointplot(data=df, x="SNR", y="IoU")
        plt.savefig(os.path.join(history_path, "SNR_IoU.png"))
        plt.title("model: %s"%model_tag)
        plt.close()
        
        sns.jointplot(data=df, x="void_frac", y="IoU")
        plt.savefig(os.path.join(history_path, "voidfrac_IoU.png"))
        plt.title("model: %s"%model_tag)
        plt.close()
        
        sns.jointplot(data=df, x="ystd", y="IoU")
        plt.savefig(os.path.join(history_path, "ystd_IoU.png"))
        plt.title("model: %s"%model_tag)
        plt.close()
        
#         sns.jointplot(data=df, x="ystd", y="IoU")
#         plt.savefig(os.path.join(history_path, "ystd_IoU.png"))
#         plt.close()
        
        print("Done")        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
