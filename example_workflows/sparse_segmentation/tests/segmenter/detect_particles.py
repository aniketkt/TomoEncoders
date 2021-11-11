#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 



if __name__ == "__main__":

    ds = DataFile(fpath, data_tag = 'data', tiff = False, VERBOSITY = 0)
    vol = ds.read_full().astype(np.float32)
    
    sbin = tuple([slice(None,None,binning)]*3) 
    vol = vol[sbin].copy()
    
    print(vol.shape)
    
    

    
