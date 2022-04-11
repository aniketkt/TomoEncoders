#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import numpy as np
from tomo_encoders.reconstruction.recon import recon_all
from tomo_encoders import DataFile
import dxchange

fname = '/data02/MyArchive/tomo_datasets/Eaton_beamtime/April2nd_2022/Chou_Plate_049.h5'

if __name__ == "__main__":

    # ds = DataFile()
    projs, flat, dark, theta = dxchange.read_aps_32id(fname, sino = (400,432))
    print(f"shape of projeciton array: {projs.shape}")
    flat = np.median(flat, axis = 0)
    dark = np.median(dark, axis = 0)
    center = projs.shape[-1]/2.0

    
    vol = recon_all(projs, theta, center, 32, flat, dark)