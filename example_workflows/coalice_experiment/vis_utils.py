#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class implementations for real-time 3D feature extraction


"""

import os
import matplotlib.pyplot as plt
import numpy as np
from tomo_encoders.misc_utils.feature_maps_vis import view_midplanes


def show_before_after(vols, p_sel, sel_idx):
    sz, sy, sx = p_sel.slices()[sel_idx]
    fig, ax = plt.subplots(1,3, figsize = (8,2*len(p_sel.points)))
    view_midplanes(vols[0][sz,sy,sx], ax = ax)
    ax[0].text(-64,48, "before", fontsize = 20, rotation = "vertical")
    fig.tight_layout()
    fig, ax = plt.subplots(1,3, figsize = (8,2*len(p_sel.points)))
    view_midplanes(vols[1][sz,sy,sx], ax = ax)    
    ax[0].text(-64,48, "after", fontsize = 20, rotation = "vertical")
    fig.tight_layout()
    return

def show_in_volume(vols, p, sel_idx):
    
    point = p.points[sel_idx]
    center = p.centers()[sel_idx]
    fig, ax = plt.subplots(2,3, figsize = (14,10))
    
    for tstep in [0,1]:
        for ia in range(3):
            img = vols[tstep].take(center[ia], axis = ia)
            ax[tstep, ia].imshow(img, cmap = 'gray')
            seg_img = np.zeros(img.shape, dtype = np.uint8)
            
            p_sel = p.copy().select_by_plane(ia, center[ia])
            all_slices = p_sel.slices()
            for idx in range(len(p_sel.points)):
                slices = all_slices[idx]
                if  ia == 0: sy, sx = slices[1], slices[2]
                elif ia == 1: sy, sx = slices[0], slices[2]
                elif ia == 2: sy, sx = slices[0], slices[1]
                seg_img[sy, sx] = 1
            ax[tstep, ia].imshow(seg_img, cmap = 'copper', alpha = 0.3)            
# ax[0].scatter([center[2]],[center[1]])
# ax[1].scatter([center[2]],[center[0]])
# ax[2].scatter([center[1]],[center[0]])    

if __name__ == "__main__":
    
    print('just a bunch of functions')
    
 
