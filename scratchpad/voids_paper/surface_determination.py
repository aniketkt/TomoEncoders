#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
from tomo_encoders.reconstruction.recon import recon_binning, recon_patches_3d
import cupy as cp
import numpy as np
from skimage.filters import threshold_otsu
from tomo_encoders import Grid
from cupyx.scipy.ndimage import label
from scipy.ndimage import label as label_np
from scipy.ndimage import find_objects
import vedo
from tomo_encoders.mesh_processing.vox2mesh import save_ply

class Surface(dict):
    def __init__(self, vertices, faces, texture = None):
        self["vertices"] = vertices
        self["faces"] = faces
        self["texture"] = texture
        return

    def __len__(self):
        return(len(self["vertices"]))
    
    def write_ply(self, filename):
        save_ply(filename, self["vertices"], self["faces"])
        return

class Voids(dict):

    def __init__(self):

        self["sizes"] = []
        self["cents"] = []
        self["cpts"] = []
        self["s_voids"] = []
        self["x_voids"] = []
        self.vol_shape = (None,None,None)

        return
        
    def __len__(self):
        return len(self["x_voids"])

    def export_surface(self, rescale_fac = 1.0, decimate_fac = 1.0):

        st_chkpt = cp.cuda.Event(); end_chkpt = cp.cuda.Event(); st_chkpt.record()    
        Vb = np.zeros(self.vol_shape, dtype = np.uint8)
        for iv, s_void in enumerate(self["s_voids"]):
            Vb[s_void] = self["x_voids"][iv]
            
        pt = np.asarray(np.where(Vb == 1)).T
        spt = np.min(pt, axis = 0)
        ept = np.max(pt, axis = 0)
        s_full = tuple([slice(spt[i3], ept[i3]) for i3 in range(3)])
        surf = vedo.Volume(Vb[s_full]).isosurface(0.5)
        if decimate_fac < 1.0:
            surf = surf.decimate(decimate_fac)
        verts = surf.points()
        for i3 in range(3):
            verts[:,i3] += spt[::-1][i3] # z, y, x to x, y, z

        surf = Surface(verts*rescale_fac if rescale_fac > 1.0 else verts, surf.faces())

        end_chkpt.record(); end_chkpt.synchronize(); t_chkpt = cp.cuda.get_elapsed_time(st_chkpt,end_chkpt)
        print(f"\tTIME: export surface {t_chkpt/1000.0:.2f} secs")

        return surf


    def import_from_grid(self, voids_b, b, x_grid, p_grid):

        Vp = np.zeros(p_grid.vol_shape)
        p_grid.fill_patches_in_volume(x_grid, Vp)

        self["sizes"] = []
        self["cents"] = []
        self["cpts"] = []
        self["s_voids"] = []
        self["x_voids"] = []
        self.vol_shape = p_grid.vol_shape

        for s_b in voids_b["s_voids"]:
            s = tuple([slice(s_b[i].start*b, s_b[i].stop*b) for i in range(3)])
            void = Vp[s]
            
            
            # make sure no other voids fall inside the bounding box
            # void, n_ = label_np(void)
            # if n_ == 0: # turns out high-res segmentation shows no void in it
            #     continue
            # idxs_ = np.arange(n_)
            # idx_max = np.argmax([np.sum(void == idx_ + 1) for idx_ in idxs_])
            # void = (void == idx_max + 1).astype(np.uint8)
            
            self["s_voids"].append(s)
            self["sizes"].append(np.sum(void))
            self["cents"].append([int((s[i].start + s[i].stop)//2) for i in range(3)])
            self["cpts"].append([int((s[i].start)) for i in range(3)])
            self["x_voids"].append(void)

        return self

    def guess_voids(self, projs, theta, center, b, b_K):

        # reconstruct and segment
        V_bin = coarse_segmentation(projs, theta, center, b_K, b, 0.5)
        Vl, n_det = label(V_bin)
        Vl = Vl.get()
        self["s_voids"] = find_objects(Vl)
        print(f"voids found: {n_det}")

        self["sizes"] = []
        self["cents"] = []
        self["cpts"] = []
        self["x_voids"] = []
        for idx, s in enumerate(self["s_voids"]):
            void = (Vl[s] == idx+1)
            self["sizes"].append(np.sum(void))
            self["cents"].append([int((s[i].start + s[i].stop)//2) for i in range(3)])
            self["cpts"].append([int((s[i].start)) for i in range(3)])
            self["x_voids"].append(void)

        self["sizes"] = np.asarray(self["sizes"])
        self["cents"] = np.asarray(self["cents"])
        self["cpts"] = np.asarray(self["cpts"])
        self.vol_shape = V_bin.shape
        
        return self

    def select_by_indices(self, idxs):

        self["x_voids"] = [self["x_voids"][ii] for ii in idxs]
        self["sizes"] = self["sizes"][idxs]
        self["cents"] = self["cents"][idxs]
        self["cpts"] = self["cpts"][idxs]
        self["s_voids"] = [self["s_voids"][ii] for ii in idxs]
        return

    def select_by_size(self, size_thresh):
        idxs = np.arange(len(self["sizes"]))
        cond_list = np.asarray([1 if self["sizes"][idx] > size_thresh**3 else 0 for idx in idxs])
        cond_list[0] = 0 # skip the sample boundary
        idxs = idxs[cond_list == 1]
        print(f'\tSTAT: size thres: {size_thresh:.2f} pixel length')          
        self.select_by_indices(idxs)
        return
    

    def select_around_void(self, void_id, radius):
        idxs = np.arange(len(self["sizes"]))
        
        dist = np.linalg.norm((self["cents"] - self["cents"][void_id]), \
               ord = 2, axis = 1)
        cond_list = dist < radius
        cond_list[0] = 0 # skip the sample boundary
        idxs = idxs[cond_list == 1]
        self.select_by_indices(idxs)
        return

    def export_grid(self, wd):

        V_bin = np.zeros(self.vol_shape, dtype = np.uint8)
        for ii, s_void in enumerate(self["s_voids"]):
            V_bin[s_void] = self["x_voids"][ii]
        
        # find patches on surface
        p3d = Grid(V_bin.shape, width = wd)
        x = p3d.extract(V_bin)
        is_sel = (np.sum(x, axis = (1,2,3)) > 0.0)

        p3d_sel = p3d.filter_by_condition(is_sel)
        r_fac = len(p3d_sel)*(wd**3)/np.prod(p3d_sel.vol_shape)
        print(f"\tSTAT: r value: {r_fac*100.0:.2f}")
        return p3d_sel, r_fac


def coarse_segmentation(projs, theta, center, b_K, b, blur_sigma):
    '''
    coarse reconstruct and thresholding
    input numpy projection data
    return cupy array
    '''    
    st_rec = cp.cuda.Event(); end_rec = cp.cuda.Event(); st_rec.record()
    V_bin = recon_binning(projs, theta, center, b_K, b, blur_sigma = blur_sigma)    
    end_rec.record(); end_rec.synchronize(); t_rec = cp.cuda.get_elapsed_time(st_rec,end_rec)
    print(f"\tTIME reconstructing with binning - {t_rec/1000.0:.2f} secs")
    
    # segmentation
    thresh = cp.float32(threshold_otsu(V_bin[::4,::4,::4].reshape(-1).get()))
    V_bin = (V_bin < thresh).astype(cp.uint8)
    return V_bin    


def guess_surface(projs, theta, center, b, b_K, wd):
    ## P-GUESS ##
    # reconstruction

    V_bin = coarse_segmentation(projs, theta, center, b_K, b, 0.5).get()
    
    # find patches on surface
    wdb = int(wd//b)
    p3d = Grid(V_bin.shape, width = wdb)
    
    x = p3d.extract(V_bin)
    is_surf = (np.std(x, axis = (1,2,3)) > 0.0)
    is_ones = (np.sum(x, axis = (1,2,3))/(wdb**3) == 1)
    is_zeros = (np.sum(x, axis = (1,2,3))/(wdb**3) == 0)
    
    p3d = p3d.rescale(b)
    p3d_surf = p3d.filter_by_condition(is_surf)
    p3d_ones = p3d.filter_by_condition(is_ones)
    p3d_zeros = p3d.filter_by_condition(is_zeros)
    eff = len(p3d_surf)*(wd**3)/np.prod(p3d_surf.vol_shape)
    print(f"\tSTAT: r value: {eff*100.0:.2f}")        
    return p3d_surf, p3d_ones, p3d_zeros

def determine_surface(projs, theta, center, fe, p_surf, Vp = None):

    # SCHEME 1: integrate reconstruction and segmention (segments data on gpu itself)
    # st_proc = cp.cuda.Event(); end_proc = cp.cuda.Event(); st_proc.record()
    # x_surf, p_surf = recon_patches_3d(projs, theta, center, p_surf, \
    #                                   apply_fbp = True, segmenter = fe, \
    #                                   segmenter_batch_size = 256)
    # end_proc.record(); end_proc.synchronize(); t_surf = cp.cuda.get_elapsed_time(st_proc,end_proc)
    
    
    # SCHEME 2: reconstruct and segment separately (copies rec data from gpu to cpu)
    st_rec = cp.cuda.Event(); end_rec = cp.cuda.Event(); st_rec.record()
    x_surf, p_surf = recon_patches_3d(projs, theta, center, p_surf, \
                                      apply_fbp =True)
    end_rec.record(); end_rec.synchronize(); t_rec = cp.cuda.get_elapsed_time(st_rec,end_rec)
    st_seg = cp.cuda.Event(); end_seg = cp.cuda.Event(); st_seg.record()
    min_max = x_surf[:,::4,::4,::4].min(), x_surf[:,::4,::4,::4].max()
    x_surf = fe.predict_patches("segmenter", x_surf[...,np.newaxis], 256, None, min_max = min_max)[...,0]
    end_seg.record(); end_seg.synchronize(); t_seg = cp.cuda.get_elapsed_time(st_seg,end_seg)
    t_surf = t_rec + t_seg
    
    print(f'\tTIME: reconstruction only - {t_rec/1000.0:.2f} secs')    
    print(f'\tTIME: segmentation only - {t_seg/1000.0:.2f} secs')
    print(f"\tTIME: reconstruction + segmentation - {t_surf/1000.0:.2f} secs")
    print(f'\tSTAT: total patches in neighborhood: {len(p_surf)}')    
    # fill segmented patches into volume
    if Vp is not None:
        p_surf.fill_patches_in_volume(x_surf, Vp)    
        return Vp
    else:
        return x_surf, p_surf
    
    
    

