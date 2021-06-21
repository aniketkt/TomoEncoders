#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Util functions for:  
1. data (patched volume) generator from any given volume data pair  
2. Handle inference on 3D auto-encoder  
3. Porespy data generator  

https://porespy.readthedocs.io/en/master/getting_started.html#generating-an-image  

"""
import h5py
from tensorflow import keras
import numpy as np
import os
import pandas as pd
from tomo_encoders.img_stats import calc_SNR

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
import seaborn as sns
import matplotlib.pyplot as plt

################### PCA Stuff ###########################


def get_latent_vector(encoder, x, sample_labels, plot_labels):

    '''
    extract latent vector from series of volumes x.  
    Parameters
    ----------
    encoders : tf.Keras.models.model  
        keras model with output of n dimensions  
    x : np.array  
        series of volumes of shape n_imgs, patch_size, patch_size, patch_size, 1  
    sample_labels  
        list of integers indicating a label value for each input volume corresponding  
    plot_labels  
        list of strings describing each possible label value in sample_labels  
    
    Returns
    -------
    pd.DataFrame  
        dataframe with code vector and meta data  
    
    '''
    h = encoder.predict(x)
    data = np.concatenate([sample_labels.reshape(-1,1), h], axis = 1)
    h_labels = ["$h_%i$"%ii for ii in range(h.shape[-1])]
    dfN = pd.DataFrame(data = data, columns = ["label_idx"] + h_labels)
    
    if plot_labels is not None:    
        dfN["label"] = ""
        for ii, plt_label in enumerate(plot_labels):
            dfN.loc[dfN["label_idx"] == ii, 'label'] = plt_label
        dfN[["datafor", "shape", "energy", "det_dist", "param"]] = dfN["label"].str.split("_", expand = True)
        dfN["measurement"] = dfN["energy"] + dfN["det_dist"]
        dfN["param"] = dfN["param"].apply(lambda X : float(X[:-1]))
        dfN.drop("energy", inplace = True, axis = 1)
        dfN.drop("det_dist", inplace = True, axis = 1)
        dfN.drop("datafor", inplace = True, axis = 1)
        
    return dfN


def fit_tSNE(dfN, N, ncomps = 2):
    '''
    projects N-dimensional data to 2D using sklearn's t-SNE  
    
    Parameters
    ----------
    dfN : pd.DataFrame  
        dataframe with the code vector and labels  
    N : int  
        input dimensionality  
    
    '''
    
    tsne = TSNE(n_components = ncomps, random_state = 2021)
    h = np.asarray(dfN[["$h_%i$"%i for i in range(N)]])
    z = tsne.fit_transform(h) # normalization using scale()?
    
    df = pd.DataFrame(data = z, columns = ["$z_%i$"%ii for ii in range(ncomps)])
    df["label"] = dfN["label"]
    df["measurement"] = dfN["measurement"]
    df["shape"] = dfN["shape"]
    df["param"] = dfN["param"]    
    
    return df
    
    
def fit_PCA(dfN, N, ncomps = 2, transform = False):
    '''
    fits N-dimensional data to 2D using sklearn's PCA  
    
    Parameters
    ----------
    dfN : pd.DataFrame  
        dataframe with the code vector and labels  
    N : int  
        input dimensionality  
    
    '''
    
    pca = IncrementalPCA(n_components = ncomps, batch_size = 500)
    h = np.asarray(dfN[["$h_%i$"%i for i in range(N)]])
    tmp = pca.fit(h) # normalization using scale()
    
    if transform:
        df = transform_PCA(dfN, N, pca, ncomps = ncomps)
        return pca, df
    else:
        return pca
    
def transform_PCA(dfN, N, pca, ncomps = 2):
    '''
    projects N-dimensional data to 2D or 3D using sklearn's PCA  
    
    Parameters
    ----------
    dfN : pd.DataFrame  
        dataframe with the code vector and labels  
    N : int  
        input dimensionality  
    pca : sklearn.PCA  
        fitted model  
    
    '''
    h = np.asarray(dfN[["$h_%i$"%i for i in range(N)]])
    
    z = pca.transform(h)
    df = pd.DataFrame(data = z, columns = ["$z_%i$"%ii for ii in range(ncomps)])
    
    if "label" in dfN.columns:
        df["label"] = dfN["label"]
        df["measurement"] = dfN["measurement"]
        df["shape"] = dfN["shape"]
        df["param"] = dfN["param"]    
    else:
        df["label_idx"] = dfN["label_idx"]
    return df
    

def rescale_z(df):
    '''
    Rescale z vector in range [-5,5]  
    '''
    z = np.asarray(df[["$z_0$", "$z_1$"]].copy())
    for ii in range(2):
        z[:,ii] = ((z[:,ii] - z[:,ii].min()) / (z[:,ii].max() - z[:,ii].min()) - 0.5)*10.0
    df[["$z_0$", "$z_1$"]] = z
    return df
    
    
def plot_2Dprojection(df, ax = None, figw = 8):
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize = (figw,figw))

    sns.scatterplot(data = df, x = "$z_0$", y = "$z_1$", \
                    hue = "shape", \
                    palette = "deep", ax = ax, \
                    legend = "full", \
                    style = "measurement")
    if ax is None:
        fig.tight_layout()
        
    return

def plot_2Dprojection_ellipsoid(df, ax = None, figw = 8):
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize = (figw,figw))

    sns.scatterplot(data = df, x = "$z_0$", y = "$z_1$", \
                    hue = "label_idx", \
                    palette = "deep", ax = ax, \
                    legend = "full")
    if ax is None:
        fig.tight_layout()
        
    return

def plot_3Dprojection(df, figw = 8):
    
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(111, projection = '3d')
    markers= ('x', 'o', '>', '<', 's', 'v', 'H', 'D', '3', '1', '2')
    # colors = ('g', 'b', 'gold', 'yellow', 'tan', 'cyan', 'magenta', 'black', 'orange', 'darkgreen')
    colors = ('g', 'b', 'gold', 'tan', 'magenta', 'black', 'orange', 'darkgreen')

    data_labels = list(df["label"].unique())
    for idx, lab in enumerate(data_labels):

        ax.scatter(df[df["label"] == lab]["$z_0$"], \
                   df[df["label"] == lab]["$z_1$"], \
                   df[df["label"] == lab]["$z_2$"], \
                   marker = markers[0] if "rock" in lab else markers[1], c = colors[idx], s = 10, label = lab)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

    ax.legend(bbox_to_anchor=(0., 1.0, 1., .1), ncol=2, loc=3, fancybox=False, framealpha=0.5, fontsize=10)
    plt.subplots_adjust(top = 0.8)    
    
    return
    
    
    
    
    
    
    
    
    
    
    





