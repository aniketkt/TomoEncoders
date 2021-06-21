#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: atekawade
"""

import sys
import os
import time

import matplotlib.pyplot as plt
import matplotlib as mpl

import pickle
figw = 12

from tomo_encoders.porosity_encoders import build_CAE_3D
from tomo_encoders.data_sampling import *
from tomo_encoders.latent_vis import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# tensorflow configs
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# list of architecture configurations to be trained
model_tags = ["110d32_set6", "111d32_set6", "110d16_set6", "111d16_set6", "110d8_set6", "111d8_set6"]

# paths  
data_path = "/data02/MyArchive/aisteer_3Dencoders/data_TomoTwin/"
model_path = "/data02/MyArchive/aisteer_3Dencoders/models/"
plot_path_pca = "/data02/MyArchive/aisteer_3Dencoders/plots/"
csv_path = os.path.join(data_path, "datalist_train.csv")

# training  
tot_steps = 500
val_split = 0.2
batch_size = 24
n_epochs = 30

# data sampling  
n_samples = 5000
patch_size = (64,64,64)
downres = 2
vol_shape = (patch_size) + (1,)
other_tag = "set6" # this will be overridden by an iterable

# model architecture
n_filters = [32, 64, 128]
n_blocks = 3
activation = 'lrelu'
batch_norm = True
hidden_units = [128, 16] # the last element will be overridden by an iterable
isconcat = [True, True, False] # this will be overridden by an iterable
pool_size = [2,2,2]
stdinput = True
add_noise = 0.18
random_rotate = True

if __name__ == "__main__":

    
    Xs, Ys, plot_labels = get_data_from_flist(csv_path, \
                                              normalize = True, \
                                              data_tags = ("recon", "gt_labels"),\
                                              group_tags = ["tomo"], \
                                              downres = downres)

    
    for model_tag in model_tags:    
        
        # iterate over model architectures
        isconcat = list(map(int,model_tag.split('d')[0]))
        hidden_units[-1] = int(model_tag.split('_')[0].split('d')[-1])
        other_tag = str(model_tag.split('_')[-1])

        #     Define architecture
        encoder, segmenter = build_CAE_3D(vol_shape, \
                                          n_filters = n_filters,\
                                          n_blocks = n_blocks,\
                                          activation = activation,\
                                          batch_norm = batch_norm,\
                                          hidden_units = hidden_units,\
                                          isconcat = isconcat, \
                                          pool_size = pool_size, \
                                          stdinput = stdinput)


        segmenter.compile(optimizer= Adam(),\
                      loss=BinaryCrossentropy())
        encoder.compile()    

        #     Training
        dg = data_generator_4D(Xs, Ys, patch_size, \
                                     batch_size, \
                                     add_noise = add_noise, \
                                     random_rotate = random_rotate)
        x, y = next(dg)
        print("Shape of x: %s"%str(x.shape))

        steps_per_epoch = int((1-val_split)*tot_steps//batch_size)
        validation_steps = int(val_split*tot_steps//batch_size)

        segmenter.fit(x = dg, epochs = n_epochs,\
                  steps_per_epoch=steps_per_epoch,\
                  validation_steps=validation_steps, verbose = 1)        

        #     PCA reduction
        dg = data_generator_4D(Xs, Ys, \
                                     patch_size, n_samples, \
                                     scan_idx = True, add_noise = add_noise)
        x, y, sample_labels = next(dg)
        print("Shape of x: %s"%str(x.shape))    

        dfN = get_latent_vector(encoder, x, sample_labels, plot_labels)
        pca, df = fit_PCA(dfN, hidden_units[-1], ncomps = 2, transform = True)

        #     Save PCA plot
        fig, ax = plt.subplots(1,1,figsize = (figw,figw))

        sns.scatterplot(data = df, x = "$z_0$", y = "$z_1$", \
                        hue = "param", \
                        palette = "viridis", ax = ax, \
                        legend = "full", \
                        style = "measurement")
        #                 size = "param", sizes = (50,100))
        if ax is None:
            fig.tight_layout()

        plt.savefig(os.path.join(plot_path_pca, model_tag + '.png'))

        #     Save model
        
        model_names = {"segmenter" : "segmenter%s.hdf5"%model_tag, \
                       "encoder" : "encoder%s.hdf5"%model_tag, \
                       "PCA" : "PCA%s.pkl"%model_tag}
        
        pkl_filename = os.path.join(model_path, model_names["PCA"])
        with open(pkl_filename, 'wb') as file:
            pickle.dump(pca, file)
        segmenter.save(os.path.join(model_path, model_names["segmenter"]))
        encoder.save(os.path.join(model_path, model_names["encoder"]))


    
    
    
    
    
    
    
    
    
    
    
    
    
