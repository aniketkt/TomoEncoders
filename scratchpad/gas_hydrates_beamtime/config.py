#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
#### THIS EXPERIMENT ####
TRAINING_MIN_INPUT_SIZE = (32,32,32)
TRAINING_BATCH_SIZE = 8
INFERENCE_BATCH_SIZE = 4
INFERENCE_INPUT_SIZE = (32,32,32)
N_EPOCHS = 10 
N_STEPS_PER_EPOCH = 20 
model_tag = "M_a06"

fpath = '/data02/MyArchive/tomo_datasets/gas_hydrates/data/exp2_time_19p6_101_102_107to110_113to185.h5'

import h5py
import numpy as np
import tqdm
from tomo_encoders import DataFile
def get_tsteps(fpath):
    hf = h5py.File(fpath)
    l = [int(key) for key in hf.keys()]
    hf.close()
    return np.asarray(l)   

def load_datasets(fpath, tsteps = None):
    
    if tsteps is None:
        tsteps = get_tsteps(fpath)
    vols = []
    
    for ii, tstep in enumerate(tqdm.tqdm(tsteps)):
        ds = DataFile(fpath, data_tag = "%03d"%tstep, tiff = False, VERBOSITY = 0)        
        vol = ds.read_full()
        vol = (vol/255.0).astype(np.float16)
        
        # to-do should we apply auto-contrast here?
        vols.append(vol)
        
    return vols

if __name__ == "__main__":

    print("EXPERIMENT WITH MODEL %s"%model_tag)
    
    
    
    
    
    
    
    
    
    
    
