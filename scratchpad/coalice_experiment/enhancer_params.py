#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import os
# to-do: get these inputs from command line or config file
model_path = '/data02/MyArchive/aisteer_3Dencoders/models/coalice_enhancer'
if not os.path.exists(model_path):
    os.makedirs(model_path)

############ MODEL PARAMETERS ############
def get_training_params(TRAINING_INPUT_SIZE):
    
    training_params = {"sampling_method" : "random", \
                       "training_input_size" : (64,64,64),\
                       "batch_size" : 24, \
                       "n_epochs" : 30,\
                       "random_rotate" : True, \
                       "add_noise" : 0.05, \
                       "max_stride" : 2, \
                       "mask_ratio" : 0.95}
    
    if TRAINING_INPUT_SIZE == (64,64,64):
        # default
        pass

    elif TRAINING_INPUT_SIZE == (32,32,32):
        training_params["training_input_size"] = TRAINING_INPUT_SIZE
        training_params["batch_size"] = 4
        training_params["max_stride"] = 2
        training_params["cutoff"] = 0.1
    
    elif TRAINING_INPUT_SIZE == (256,256,256):
        training_params["training_input_size"] = TRAINING_INPUT_SIZE
        training_params["batch_size"] = 4
        training_params["max_stride"] = 2
        training_params["cutoff"] = 0.0
    else:
        raise ValueError("input size not catalogued yet")

    print("\n", "#"*55, "\n")
    print("\nTraining parameters\n")
    for key, value in training_params.items():
        print(key, value)
    
    return training_params
        
############ MODEL PARAMETERS ############

def get_model_params(model_tag):

    m = {"n_filters" : [16, 32, 64], \
         "n_blocks" : 3, \
         "activation" : 'lrelu', \
         "batch_norm" : True, \
         "isconcat" : [True, True, True], \
         "pool_size" : [2,2,2]}
    
    # default
    model_params = m.copy()
    
    if model_tag == "M_a01":
        pass
    
    # a02 - very fast model
    elif model_tag == "M_a02":
        model_params["n_filters"] = [16, 32]
        model_params["pool_size"] = [ 2,  4]
    
    # a03 - very deep (slow) model with more filters
    elif model_tag == "M_a03":
        model_params["n_filters"] = [32, 64, 128]
        model_params["pool_size"] = [ 2,  2,   2]
    
    # a04 - shallow model - 2 max pools with more filters
    elif model_tag == "M_a04":
        model_params["n_filters"] = [32, 64]
        model_params["pool_size"] = [ 2,  4]
    
    # a05 - shallow model - 2 max equal-sized max pools with more filters (results in bigger bottleneck size?)
    elif model_tag == "M_a05":
        model_params["n_filters"] = [32, 64]
        model_params["pool_size"] = [ 2,  2]
        
    elif model_tag == "M_a06":
        model_params["n_filters"] = [16, 32]
        model_params["pool_size"] = [ 2,  2]
        
    else:
        raise ValueError("model_tag not found")
        
    model_params["n_blocks"] = len(model_params["n_filters"])
    model_params["isconcat"] = [True]*len(model_params["n_filters"])

    print("\n", "#"*55, "\n")
    print("\nModel is %s"%model_tag)
    for key, value in model_params.items():
        print(key, value)
    
    return model_params


if __name__ == "__main__":

    fe = SparseSegmenter(model_initialization = 'define-new', \
#                          input_size = , \
                         descriptor_tag = "M_a01",\
                         gpu_mem_limit = gpu_mem_limit,\
                         **model_params)        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
