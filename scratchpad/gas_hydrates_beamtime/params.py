#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
Stuff that should eventually go into a model.cfg file.

""" 
import os
# to-do: get these inputs from command line or config file
model_path = '/data02/MyArchive/aisteer_3Dencoders/models/gashydrates_enhancer'
if not os.path.exists(model_path):
    os.makedirs(model_path)

############ MODEL PARAMETERS ############
def get_training_params(TRAINING_INPUT_SIZE, N_EPOCHS = None, N_STEPS_PER_EPOCH = None, BATCH_SIZE = None):
    
    training_params = {"sampling_method" : "random", \
                       "training_input_size" : (64,64,64),\
                       "batch_size" : 24, \
                       "n_epochs" : 10,\
                       "random_rotate" : True, \
                       "add_noise" : 0.1, \
                       "max_stride" : 2, \
                       "mask_ratio" : 0.95, \
                       "steps_per_epoch" : 100}
    
    if TRAINING_INPUT_SIZE == (64,64,64):
        # default
        pass

    elif TRAINING_INPUT_SIZE == (32,32,32):
        training_params["training_input_size"] = TRAINING_INPUT_SIZE
        training_params["batch_size"] = 4
        training_params["max_stride"] = 2
    
    elif TRAINING_INPUT_SIZE == (256,256,256):
        training_params["training_input_size"] = TRAINING_INPUT_SIZE
        training_params["batch_size"] = 4
        training_params["max_stride"] = 2
    
    elif TRAINING_INPUT_SIZE == (128,128,128):
        training_params["training_input_size"] = TRAINING_INPUT_SIZE
        training_params["batch_size"] = 8
        training_params["max_stride"] = 2
        training_params["n_epochs"] = 5
        training_params["steps_per_epoch"] = 100

    elif TRAINING_INPUT_SIZE == (32,128,128):
        training_params["training_input_size"] = TRAINING_INPUT_SIZE
        training_params["batch_size"] = 16
        training_params["max_stride"] = 2
        training_params["n_epochs"] = 5
        training_params["steps_per_epoch"] = 100
        training_params["random_rotate"] = False
    else:
        raise ValueError("input size not catalogued yet")

    if N_EPOCHS is not None:
        training_params["n_epochs"] = N_EPOCHS
    if N_STEPS_PER_EPOCH is not None:
        training_params["steps_per_epoch"] = N_STEPS_PER_EPOCH
    if BATCH_SIZE is not None:
        training_params["batch_size"] = BATCH_SIZE
    
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
    
    # a02 - shallow first, deep later. should be faster with high-level context. try with 128
    elif model_tag == "M_a02":
        model_params["n_filters"] = [16, 64]
        model_params["pool_size"] = [ 2,  4]
    
    # a03 - very deep (slow) model with more filters
    elif model_tag == "M_a03":
        model_params["n_filters"] = [16, 32]
        model_params["pool_size"] = [ 2,  2]
    
    # a04 - super shallow model - 1 max pool
    elif model_tag == "M_a04":
        model_params["n_filters"] = [16]
        model_params["pool_size"] = [2]
    
    # a05 - flat CNN - no pooling
    elif model_tag == "M_a05":
        model_params["n_filters"] = [16]
        model_params["pool_size"] = [1]
        
    elif model_tag == "M_a06":
        model_params["n_filters"] = [8]
        model_params["pool_size"] = [2]
        
    else:
        raise ValueError("model_tag not found")
        
    model_params["n_blocks"] = len(model_params["n_filters"])
    model_params["isconcat"] = [True]*len(model_params["n_filters"])
    
    # BATCH_NORM_OVERRIDE
#     model_params["batch_norm"] = False

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
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
