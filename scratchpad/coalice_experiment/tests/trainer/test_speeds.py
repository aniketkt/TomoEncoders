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

import os 
import tqdm
import pandas as pd
from tomo_encoders.neural_nets.autoencoders import SelfSupervisedCAE
import matplotlib as mpl
mpl.use('Agg')
from params import *


#### THIS EXPERIMENT ####
model_size = tuple([int(sys.argv[1])]*3)
chunk_size = int(sys.argv[2])
max_chunk_size = int(sys.argv[3]) if len(sys.argv) > 3 else 512
model_tag = sys.argv[4] if len(sys.argv) > 4 else "M_a01"
######### DEFINE EXPERIMENT ON 'nb'
# nb = chunk_size*2**np.random.randint(0, 4, 20)
nb = []
for ii in range(5):
    nb_ = chunk_size*2**ii
    if nb_ <= max_chunk_size:
        nb.append(nb_)
    else:
        break
min_max = (-1,1)
print("\nnb list: ", nb)

if min_max is None:
    print("\tnormalization is turned off")

def fit(fe):
    training_params = get_training_params()
    batch_size = training_params["batch_size"]
    dg = fe.random_data_generator(batch_size)
    vols = np.random.normal(0, 1, (3, 600, 960, 960))
    
    t0 = time.time()
    fe.train(vols, **training_params)
    t1 = time.time()
    training_time = (t1 - t0)
    print("training time per epoch = %.2f seconds"%(training_time/n_epochs))        
    return

def infer(fe):
    
    
    unit_times = []
    fig, ax = plt.subplots(1,1,figsize = (8,6))
    patch_size = fe.model_size
    print("#"*55,"\n")
    for jj in range(len(nb)):
        x = np.random.uniform(0, 1, tuple([nb[jj]] + list(patch_size) + [1])).astype(np.float32)
        
        z, t_unit = fe.predict_embeddings(x, chunk_size, min_max = min_max, TIMEIT = True)
        
        print("inf. time per voxel %.2f nanoseconds"%(t_unit/(np.prod(patch_size))*1.0e6))
        print("\n","#"*55,"\n")
        unit_times.append(t_unit)
#         print('next')
    
    ax.scatter(nb, unit_times, marker = 'o', color = 'black')
    ax.set_xlabel('BATCH SIZE')
    ax.set_ylabel('INFERENCE TIME PER UNIT PATCH (ms)')
#     ax.set_ylim([0,80])
#     ax.set_xticks(nb)
    plt.close()

    df = pd.DataFrame(columns = ["nb", "t_unit"], data = np.asarray([nb, unit_times]).T)
    t_unit_mean = np.mean(unit_times)
    print("MEAN INFERENCE TIME PER UNIT PATCH: %.2f ms"%t_unit_mean)
    t_vox = t_unit_mean/(np.prod(patch_size))*1.0e6
    print("MEAN INFERENCE TIME PER VOXEL: %.2f nanoseconds"%t_vox)
    
    
if __name__ == "__main__":

    print("EXPERIMENT WITH MODEL %s"%model_tag)
    model_params = get_model_params(model_tag)
    
    fe = SelfSupervisedCAE(model_initialization = 'define-new', \
                           model_size = model_size, \
                           descriptor_tag = model_tag, \
                           **model_params)    
    
    print("EXPERIMENT WITH INPUT_SIZE = ", model_size)
    infer(fe)
#     fit(fe)
    
#     if len(sys.argv) > 1:
#         if sys.argv[1] == "infer":
#             infer(fe)
#         elif sys.argv[1] == "fit":
#             fit(fe)
#     else:
#         fit(fe)
#         infer(fe)
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
