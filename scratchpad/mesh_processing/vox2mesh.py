#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Yashas Satapathy
Email: yashsatapathy[at]yahoo[dot]com

"""


from skimage import io
import vedo
import numpy as np
import os


         
def make_void_mesh(arr,loc):
    '''
    Creates a surface mesh of of the void and gets shifted to the correct location.
    
    Parameters
    ----------
    arr : list of ndarrays
        each ndarray is a binarized volume representing a void
    locs : list
        each item in the list is a tuple (z, y, x) of the center location of the void in the global coordinate system  
    '''
    loc_z, loc_y, loc_x = loc
    
    surf = vedo.Volume(arr).isosurface(0.5).smooth()
    
    face = surf.faces()
    vert = surf.points()

    #Shift void to their proper location given the center
    for j in range(len(vert)):
        vert[j][0] = vert[j][0]+loc_x
        vert[j][1] = vert[j][1]+loc_y
        vert[j][2] = vert[j][2]+loc_z
    
    return (vert,face)

def voids2ply(x_voids,cpts):
    '''
    Takes in a list of binarized volumes (each representing a single void) and a list of corner locations in the global coordinate
    system of the reconstructed object and outputs a ply mesh visualizing all voids.
    
    Parameters
    ----------
    x_voids : list  
        each item in the list is a 3d np.ndarray
    cpts : list
        each item in the list is a tuple (iz, iy, ix) of the corner coordinates in the global coordinate system  
    output_name : string
        the name of the ply file
    '''
    id_len = 0
    v_list = []
    f_list = []
    for i in range(len(x_voids)): 
        v,f = make_void_mesh(x_voids[i],cpts[i])
        v_list.append(v)
        f_list.append(np.array(f)+id_len)
        id_len = id_len+len(v)
    v_final = np.concatenate(v_list,axis=0)
    f_final = np.concatenate(f_list,axis=0)
    return (v_final, f_final)


