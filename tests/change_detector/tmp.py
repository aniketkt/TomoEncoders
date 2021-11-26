#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing the regularized_autoencoders for gas hydrates data

"""

#Allocate memory First
import tensorflow as tf
GPU_mem_limit=8.0
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU_mem_limit*1000.0)])

    except RuntimeError as e:
        print(e) 
        
        

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import sys
from ct_segnet import viewer
# from features3D import FeatureExtractor4D
# from patches import Patches
import tensorflow as tf
from tomo_encoders import *
from tomo_encoders import neural_nets
from tomo_encoders.neural_nets import regularized_encoder_decoders
from tomo_encoders.neural_nets import porosity_encoders
from tomo_encoders.neural_nets.regularized_encoder_decoders import *

# from tomo_encoders.neural_nets.porosity_encoders import build_CAE_3D

from tomo_encoders import Patches

# sys.path.append('../.')
sys.path.append('../../coalice_experiment/')
# sys.path.append('../../../tomo_encoders/')
# sys.path.append('../../../tomo_encoders/structures/')

# from recon4D import *
import time



# from vis_utils import *


if __name__ == "__main__":
    print('Training the regularized autoencoders for gas hydrates data.')

    vols = [np.random.rand(600,64,64,64,1), np.random.rand(6000,64,64,64,1)]
    print(vols[0].shape)
    print(vols[1].shape)


    # Feature Extraction stuff
    model_path = '/data02/AZC/Tomography/models/'

    model_size = (64,64,64)
    # model_size = (32,32,32)
    # model_params = {"n_filters" : [16, 32, 64],\
    #                 "n_blocks" : 3,\
    #                 "activation" : 'lrelu',\
    #                 "batch_norm" : True,\
    #                 "hidden_units" : [128, 128],\
    #                 "isconcat" : [True, True, True],\
    #                 "pool_size" : [2,2,2],\
    #                 "stdinput" : False,\
    #                 "loss_func" : 'mse'
    #                }

    model_params = {"n_filters" : [16, 32, 64],\
                    "n_blocks" : 3,\
                    "activation" : 'lrelu',\
                    "batch_norm" : True,\
                    "hidden_units" : [128, 128],\
                    "pool_size" : [2,2,2]                    
                   }

    training_params = {"sampling_method" : "random-fixed-width", \
                       "batch_size" : 10, \
                       "n_epochs" : 30,\
                       "random_rotate" : True, \
                       "add_noise" : 0.15, \
                       "max_stride" : 1}
    descriptor_tag = 'numpy-random-4D'

    
    fe = SelfSupervisedCAE(vols[0].shape, model_initialization = 'define-new', \
                             model_size = model_size, \
                             descriptor_tag = descriptor_tag, \
                             **model_params)    



    fe.models['encoder'].summary()
    fe.models['decoder'].summary()
    NAE=Regularized_Autoencoder(fe.models['encoder'],fe.models['decoder'],weight=1/250,regularization_type='kl')
    NAE.compile(optimizer='adam')
    NAE.fit(vols[0], epochs = 10 ,\
                      batch_size=16)