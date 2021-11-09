#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import sys 
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np 
import cupy as cp 
import time 
import h5py 
from tomo_encoders import DataFile, Patches
from tomo_encoders.tasks import SparseSegmenter
import os 
import tqdm
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
save_path = '/home/atekawade/Dropbox/Arg/transfers/runtime_plots'
csv_path = os.path.join(save_path, "times.csv")
from params import *


#### THIS EXPERIMENT ####
model_tag = "M_a01"
chunk_size = 32

######### DEFINE EXPERIMENT ON 'nb'
# nb = chunk_size*2**np.random.randint(0, 4, 20)
# nb = chunk_size*2**np.arange(0, 5)
# nb = np.random.randint(16, 512+1, 20)
# nb = nb.tolist()
nb = [1000]*1
min_max = (-1,1)
print("\nnb list: ", nb)

if min_max is None:
    print("\tnormalization is turned off")

def fit(fe):
    
    batch_size = training_params["batch_size"]
    n_epochs = training_params["n_epochs"]
    
    dg = fe.random_data_generator(batch_size)
    
    t0 = time.time()
    tot_steps = 1000
    val_split = 0.2
    steps_per_epoch = int((1-val_split)*tot_steps//batch_size)
    validation_steps = int(val_split*tot_steps//batch_size)
    
    fe.models["segmenter"].fit(x = dg, epochs = n_epochs, batch_size = batch_size,\
              steps_per_epoch=steps_per_epoch,\
              validation_steps=validation_steps, verbose = 1)    
    t1 = time.time()
    training_time = (t1 - t0)
    print("training time per epoch = %.2f seconds"%(training_time/n_epochs))        
    return

def infer(fe):

#     Possible slowdown of first iteration due to tensorflow Dataset creation?
#     https://github.com/tensorflow/tensorflow/issues/46950
    nb_init = chunk_size
    num_inits = 3

    print("EXPERIMENT WITH CHUNK_SIZE = %i"%chunk_size)
    for jj in range(num_inits):
        x = np.random.uniform(0, 1, tuple([nb_init] + list(model_size) + [1])).astype(np.float32)
        y_pred, t_unit = fe._predict_patches(x, chunk_size, None, \
                                                min_max = min_max, \
                                                TIMEIT = True)
    unit_times = []
    fig, ax = plt.subplots(1,1,figsize = (8,6))
    
    for jj in range(len(nb)):
        x = np.random.uniform(0, 1, tuple([nb[jj]] + list(model_size) + [1])).astype(np.float32)
        y_pred, t_unit = fe._predict_patches(x, chunk_size, None, \
                                                min_max = min_max, \
                                                TIMEIT = True)
        unit_times.append(t_unit)
    
    ax.scatter(nb, unit_times, marker = 'o', color = 'black')
    ax.set_xlabel('BATCH SIZE')
    ax.set_ylabel('INFERENCE TIME PER UNIT PATCH (ms)')
#     ax.set_ylim([0,80])
#     ax.set_xticks(nb)
    plt.savefig(os.path.join(save_path, "batch_sizes.png"))
    plt.close()

    df = pd.DataFrame(columns = ["nb", "t_unit"], data = np.asarray([nb, unit_times]).T)
    df.to_csv(csv_path, index = False)
    print("MEAN INFERENCE TIME PER UNIT PATCH: %.2f ms"%np.mean(unit_times))
    
if __name__ == "__main__":

    model_params = get_model_params(model_tag)
    fe = SparseSegmenter(model_initialization = 'define-new', \
                         model_size = model_size, \
                         descriptor_tag = model_tag,\
                         gpu_mem_limit = gpu_mem_limit,\
                         **model_params)        
#     fe.print_layers("segmenter")    
    if len(sys.argv) > 1:
        if sys.argv[1] == "infer":
            infer(fe)
        elif sys.argv[1] == "fit":
            fit(fe)
    else:
        fit(fe)
        infer(fe)
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
