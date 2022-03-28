#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 

from tomo_encoders import DataFile


dir_path = 'path/to/some/'
rec_fpaths = ["tiff-folder-l1", "tiff-folder-l2"]

nzl, ny, nx = 1000, 2000, 2000
n_layers = 3



if __name__ == "__main__":

    pass
    # do stuff
    # STEP 1
    # make a big volume that stitches together all layers in one volume; Vx_full.shape will be (tot_ht, ny, nx)

    Vx_layers = [np.zeros(1200, 3000, 3000), np.zeros(1000, 3000, 3000), np.zeros(1400, 3000, 3000)]

    usable_layer_ht = 900
    z_height = usable_layer_ht*3
    Vx_full = np.zeros((z_height, 3000, 3000))
    stitching_starts = [val1, val2, val3]
    stitching_ends = [val4, val5, val6]

    for ix, Vx in enumerate(Vx_layers): # ix goes from 0, 1, 2 or the number of layers essentially
        Vx_full[ix*usable_layer_ht:(ix+1)*usable_layer_ht,...] = Vx[stitching_starts[ix]: stitching_ends[ix]]

    
    # for rec_fpath in rec_fpaths:
    #     # read reconstructed volume from given path
    #     ds = DataFile(rec_fpath, tiff = True)
    #     Vx = ds.read_full()

    # STEP 2
    # Process Vx_full into Vy_full where Vy_full contains only ones (inside void) and zeros (inside metal)



    # STEP 3
    # Process Vy_full into void_vols where void_vols is a list of many ndarrays with different shapes (pz, py, px) representing each void
    # Also output cz, cy, cx for each void_vol in void_vols giving the center of the void volume w.r.t. the coordinates in Vy_full



    # STEP 4
    # Process all void_vols into void_surfs in the form of a single .ply file and save
