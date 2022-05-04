#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: M Arshad Zahangir Chowdhury
Email: arshad.zahangir.bd[at]gmail[dot]com
"""

from array import array
from sklearn import cluster
from sklearn.cluster import KMeans, DBSCAN, OPTICS
import numpy as np
from matplotlib import pyplot as plt

from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors

import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import plotly.express as px


import random
import datetime
import math





def find_roi_voids(features,estimators, keep_roi_voids_only = True, visualizer = False):
    '''
    Function to cluster all voids using their x,y,z center coordinates.
    For DBSCAN, adjust eps based on elbow of reachability plot.
    For OPTICS, eps is auto-adjusted and core void indices will not be returned.
    
    Parameters
    ----------
    features: numpy array, 
        x,y,z coordinates for each void center
    estimators: list, 
        each clustering algorithm is provided as (name, estimator class)
    keep_high_density_voids_only: boolean, 
        removes the voids which are large from the clustering plot. These are all points whom DBSCAN/OPTICS would label -1.
        
        
    example usage (uncomment the following and use it)

    # import void_clustering
    # from void_clustering import *

    # size = 500
    # features = np.empty((size,3))
    # for idx in range(size):
    #     features[idx][0]=float(random.randint(-100, 100))
    #     features[idx][1]=float(random.randint(-100, 100))
    #     features[idx][2]=float(random.randint(-100, 100))

    # print(features.shape)

    # df= pd.DataFrame()
    # df['x']=features[:,0]
    # df['y']=features[:,1]
    # df['z']=features[:,2]

    # roi_voids_indices,roi_voids_coordinates = find_roi_voids(features, 
    #                               estimators = [
    #     ("DBSCAN", DBSCAN(eps=17,min_samples=5)),
    # #     ("OPTICS", OPTICS(min_samples=10, cluster_method='xi')),
    # ],
    #                               keep_core_voids_only = True, visualizer = True)
    
    # count and rank best 3 core voids
    # rank_voids_spherical(core_voids_coordinates,features,15)
    # rank_voids_elliptical(core_voids_coordinates,features,15,18,23)
  
    
    Returns
    ----------
    elbow plot for eps selection and clustering in plotly and saved figures
    and return roi_void_indices and roi_void_coordinates. Each roi_void comprise of dbscan core voids.
    
    '''
    fignum = 1
    
    
    df= pd.DataFrame()
    df['x']=features[:,0]
    df['y']=features[:,1]
    df['z']=features[:,2]
    
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(features)
    distances, indices = nbrs.kneighbors(features)

    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    
    # fig = plt.figure(fignum, figsize=(4, 3))
    # plt.plot(distances)
    # plt.title('Reachability (sorted distances to closest neighbor)')
    # plt.ylabel('distances');
    # if visualizer == True: 
    #     plt.show()
    # else:
    #     plt.close()
        
    # plt.savefig('figure' + '_reachability' +'.jpg')
    

    for name, est in estimators:
        

        t_start = datetime.datetime.now()
        est.fit(df.to_numpy())
        t_end = datetime.datetime.now()
        delta = t_end - t_start
        Time_elapsed=delta.total_seconds()

        print('Time elaspsed (s): ', Time_elapsed) #
        
        labels = est.labels_
        core_voids_indices = est.core_sample_indices_  #dbscan only
        core_voids_coordinates = est.components_  #dbscan only
        df['labels'] = labels
        print('name: ', name)
    #     print('labels: ',labels)
    #     print('core_voids_indices: ',core_voids_indices) #dbscan only

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Estimated number of clusters containing small high density voids: %d" % n_clusters_)
        print("Estimated number of (noisy, outlier) voids: %d" % n_noise_)

        if keep_roi_voids_only == True:
            secondarydf = df[df['labels'] !=-1]
        else:
            secondarydf = df


        # fig = px.scatter_3d(secondarydf, x='x', y='y', z='z', size_max=15,
        #               color='labels', template='simple_white',width=600, height=600);


        # fig.update_traces(marker_size = 4)
        # fig.update_coloraxes(colorbar_orientation='h')
        # fig.update_layout(scene = dict(
        #                     xaxis_title='x',
        #                     yaxis_title='y',
        #                     zaxis_title='z'),
        #                   title=name
        #                     );


        # fig.update_layout(
        #     legend=dict(
        #         x=0.8,
        #         y=1.0,
        #         traceorder="normal",
        #         font=dict(
        #             family="sans-serif",
        #             size=12,
        #             color="black"
        #         ),
        #     )
        # );


        # if visualizer == True:   
        #     fig.show()
        # else:
        #     plt.close()
        # fig.write_image('figure' + str(fignum) +'.jpg')



        # fignum = fignum + 1
        
        # get roi_void indices and centers here and select top 5 via np.argsort

        # # calculate cluster center of a core void and get the index and make corrections

        if n_clusters_ == 1:
            roi_voids_indices = core_voids_indices[0]
            roi_voids_coordinates = core_voids_coordinates[0]

        elif n_clusters_ <= 0:
            return None, None

        elif n_clusters_ >= 1:

            roi_voids_indices = []
            roi_voids_coordinates = []
            for label in np.unique(labels):
                if label != -1:
                    temp_df = df[df['labels'] == label] 
                    roi_voids_indices.append(temp_df.index.to_numpy()[0])
                    roi_voids_coordinates.append(features[core_voids_indices])
        
            
            roi_voids_indices = np.array(roi_voids_indices)
            roi_voids_coordinates = np.array(roi_voids_coordinates)[-1]

        return roi_voids_indices,roi_voids_coordinates

    
def n_voids_in_sphere(roi_void_coord, features, thresh_dist):
    '''
    Finds voids in and on sphere with radius of thres_dist
    '''
    
    distance = np.sqrt(np.sum((features - roi_void_coord)**2, axis = 1))
    
    return np.sum(distance <= thresh_dist)

def n_voids_in_ellipsoid(roi_void_coord, features, a,b,c):
    '''
    Finds voids in and on ellipsoid with a,b,c parameters of ellipsoid equation (x/a)**2 + (y/a)**2 + (z/c)**2 =1
    '''
    
    
    distance = np.sqrt(np.sum((features/np.array([a,b,c]) - roi_void_coord/np.array([a,b,c]))**2, axis = 1))
    
    
    return np.sum(distance <= 1)    

def rank_voids_spherical(roi_voids_coordinates,features,thresh_dist, n_rois=3):
    '''
    Rank n_roi core voids which have most points inside a sphere.
    '''
    if roi_voids_coordinates is None:
        print("\tWARNING: No ROIs were detected")
        return None
    else:
        print("\tSTAT: ROIs are being ranked.")
        inside_pts_list = []
        for roi_void_coordinate in roi_voids_coordinates:
            n_inside = n_voids_in_sphere(roi_void_coordinate, features, thresh_dist)
            inside_pts_list.append(n_inside)
            # print(f'core {roi_void_coordinate}; n within spherical core {n_inside}')

        # return the coordinates of the top three core voids that contain the highest tiny voids in an ellipsoid/sphere

        n_rois = min(n_rois, len(roi_voids_coordinates))

        ranked_cells_list = []
        for idx in range(n_rois):
            ranked_cell = roi_voids_coordinates[np.argsort(np.array(inside_pts_list))[-1-idx]]
            ranked_cells_list.append(ranked_cell)

        print("\tSTAT: ROIs ranking complete.")
        return np.array(ranked_cells_list)


def rank_voids_elliptical(roi_voids_coordinates,features,a,b,c,n_rois=3):
    '''
    Rank n_rois core voids which have most points inside a ellipsoid.
    '''
    if roi_voids_coordinates is None:
        print("\tWARNING: No ROIs were detected")
        return None
    else:
        print("\tSTAT: ROIs are being ranked.")
        inside_pts_list = []
        for roi_void_coordinate in roi_voids_coordinates:
            n_inside = n_voids_in_ellipsoid(roi_void_coordinate, features,a,b,c)
            inside_pts_list.append(n_inside)
    #         print(f'core {roi_void_coordinate}; n within ellipical core {n_inside}')


        n_rois = min(n_rois, len(roi_voids_coordinates))
        ranked_cells_list = []
        for idx in range(n_rois):
            ranked_cell = roi_voids_coordinates[np.argsort(np.array(inside_pts_list))[-1-idx]]
            ranked_cells_list.append(ranked_cell)
        print("\tSTAT: ROIs ranking complete.")
        return np.array(ranked_cells_list)
