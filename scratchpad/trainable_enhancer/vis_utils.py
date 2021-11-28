#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class implementations for real-time 3D feature extraction


"""

import os
import matplotlib.pyplot as plt
import numpy as np
from tomo_encoders.misc.viewer import view_midplanes


plot_out_path = '/home/atekawade/Dropbox/Arg/transfers/runtime_plots/scratchpad_enhancer'
if not os.path.exists(plot_out_path):
    os.makedirs(plot_out_path)

def show_planes(vol, filetag = None):
    
    fig, ax = plt.subplots(1,3, figsize = (14,6))
    ax[0].imshow(vol[int(vol.shape[0]*0.2)], cmap = 'gray')
    ax[1].imshow(vol[int(vol.shape[0]*0.5)], cmap = 'gray')
    ax[2].imshow(vol[int(vol.shape[0]*0.8)], cmap = 'gray')                
    
    
    import pdb; pdb.set_trace()
    if filetag is None:
        plt.show()
    else:
        plt.savefig(os.path.join(plot_out_path, filetag + ".png"))
    plt.close()
    return

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

            
if __name__ == "__main__":
    
    print('just a bunch of functions')
    
 
