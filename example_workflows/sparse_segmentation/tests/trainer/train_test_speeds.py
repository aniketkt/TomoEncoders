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

# to-do: get these inputs from command line or config file
model_size = (64,64,64)
data_path = '/data02/MyArchive/AM_part_Xuan' #ensure this path matches where your data is located.
model_path = '/data02/MyArchive/aisteer_3Dencoders/models/AM_part_segmenter'
descriptor_tag = 'temp'
gpu_mem_limit = 48.0
chunk_size = 32

######### DEFINE EXPERIMENT ON 'nb'
# nb = chunk_size*2**np.random.randint(0, 4, 20)
# nb = chunk_size*2**np.arange(0, 5)
# nb = np.random.randint(16, 512+1, 20)
# nb = nb.tolist()
nb = [1000]*1
print("\nnb list: ", nb)

print("###### EXPERIMENT WITH CHUNK_SIZE = %i"%chunk_size)
############ MODEL PARAMETERS ############
# model_params = {"n_filters" : [32, 64],\
#                 "n_blocks" : 2,\
#                 "activation" : 'lrelu',\
#                 "batch_norm" : True,\
#                 "isconcat" : [True, True],\
#                 "pool_size" : [2,4],\
#                 "stdinput" : False}

model_params = {"n_filters" : [16, 32, 64],\
                "n_blocks" : 3,\
                "activation" : 'lrelu',\
                "batch_norm" : True,\
                "isconcat" : [True, True, True],\
                "pool_size" : [2,2,2],\
                "stdinput" : False}

training_params = {"sampling_method" : "random", \
                   "batch_size" : 24, \
                   "n_epochs" : 2,\
                   "random_rotate" : True, \
                   "add_noise" : 0.05, \
                   "max_stride" : 8, \
                   "cutoff" : 0.2, \
                   "normalize_sampling_factor": 4}

def _rescale_data(data, min_val, max_val):
    '''
    Recales data to values into range [min_val, max_val]. Data can be any numpy or cupy array of any shape.  

    '''
    xp = cp.get_array_module(data)  # 'xp' is a standard usage in the community
    eps = 1e-12
    data = (data - min_val) / (max_val - min_val + eps)
    return data

def predict_by_chunk(fe, x, chunk_size, out_arr, min_max = None):

    t0 = time.time()
    nb = len(x)
    nchunks = int(np.ceil(nb/chunk_size))
    nb_padded = nchunks*chunk_size
    padding = nb_padded - nb

    if out_arr is None:
        out_arr = np.empty_like(x) # use numpy since return from predict is numpy
    else:
        # to-do: check dims
        pass
    
    for k in range(nchunks):

        sb = slice(k*chunk_size , min((k+1)*chunk_size, nb))
        x_in = x[sb,...]

        if padding != 0:
            if k == nchunks - 1:
                x_in = np.pad(x_in, ((0,padding), (0,0), (0,0), (0,0), (0,0)), mode = 'constant')
            
            if min_max is not None:
                min_val, max_val = min_max
                x_in = _rescale_data(x_in, float(min_val), float(max_val))
            
            x_out = fe.models['segmenter'].predict(x_in)

            if k == nchunks -1:
                x_out = x_out[:-padding,...]
        else:
            
            if min_max is not None:
                min_val, max_val = min_max
                x_in = _rescale_data(x_in, float(min_val), float(max_val))
            
            x_out = fe.models['segmenter'].predict(x_in)
        out_arr[sb,...] = x_out
    
    t_unit = (time.time() - t0)*1000.0/nb
    print("inf. time p. input patch = %.2f ms, nb = %i"%(t_unit, nb))        
    print("\n")
    return out_arr, t_unit


def random_data_generator(patch_size, batch_size):
    
    while True:
        
        data_shape = tuple([batch_size] + list(patch_size) + [1])
        print(data_shape)
        x = np.random.uniform(0, 1, data_shape)#.astype(np.float32)
        y = np.random.randint(0, 2, data_shape)#.astype(np.uint8)
        x[x == 0] = 1.0e-12
        yield x, y
        
def fit(fe):
    
    batch_size = training_params["batch_size"]
    n_epochs = training_params["n_epochs"]
    
    dg = random_data_generator(model_size, batch_size)
    
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

import matplotlib as mpl
mpl.use('Agg')
save_path = '/home/atekawade/Dropbox/Arg/transfers/runtime_plots'
csv_path = os.path.join(save_path, "times.csv")
def infer(fe):

#     Possible slowdown of first iteration due to tensorflow Dataset creation?
#     https://github.com/tensorflow/tensorflow/issues/46950
    nb_init = chunk_size
    num_inits = 3
    
    for jj in range(num_inits):
        x = np.random.uniform(0, 1, tuple([nb_init] + list(model_size) + [1])).astype(np.float32)
        y_pred, t_unit = predict_by_chunk(fe, x, chunk_size, None, min_max = (-1,1))
        
    
        
    unit_times = []
    fig, ax = plt.subplots(1,1,figsize = (8,6))
    
    for jj in range(len(nb)):
        x = np.random.uniform(0, 1, tuple([nb[jj]] + list(model_size) + [1])).astype(np.float32)
        y_pred, t_unit = predict_by_chunk(fe, x, chunk_size, None)
        unit_times.append(t_unit)
    
    ax.scatter(nb, unit_times, marker = 'o', color = 'black')
    ax.set_xlabel('batch size')
    ax.set_ylabel('inference time per unit patch (ms)')
#     ax.set_ylim([0,80])
#     ax.set_xticks(nb)
    plt.savefig(os.path.join(save_path, "batch_sizes.png"))
    plt.close()

    df = pd.DataFrame(columns = ["nb", "t_unit"], data = np.asarray([nb, unit_times]).T)
    df.to_csv(csv_path, index = False)
    print("mean inference time per unit patch: %.2f ms"%np.mean(unit_times))
if __name__ == "__main__":

    fe = SparseSegmenter(model_initialization = 'define-new', \
                         model_size = model_size, \
                         descriptor_tag = descriptor_tag,\
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
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
