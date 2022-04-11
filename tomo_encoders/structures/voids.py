#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
""" 
import cupy as cp
import numpy as np
from tomo_encoders import Grid
from cupyx.scipy.ndimage import label
from scipy.ndimage import label as label_np
from scipy.ndimage import find_objects
import vedo
from tomo_encoders.mesh_processing.vox2mesh import save_ply
from tomo_encoders.tasks.surface_determination import coarse_segmentation
import os
from tifffile import imsave, imread
import h5py

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
        self["x_boundary"] = []
        self.vol_shape = (None,None,None)

        return
        
    def __len__(self):
        return len(self["x_voids"])

    
    def write_to_disk(self, fpath, overwrite = True):

        '''
        '''

        if os.path.exists(fpath):
            assert overwrite, "overwrite not permitted"
            from shutil import rmtree
            rmtree(fpath)
        
        os.makedirs(fpath)
        
        # write void data to tiff
        void_dir = os.path.join(fpath, "voids")
        os.makedirs(void_dir)
        tot_num = len(str(len(self)))
        for iv, void in enumerate(self["x_voids"]):
            void_id_str = str(iv).zfill(tot_num)
            imsave(os.path.join(void_dir, "void_id_" + void_id_str + ".tiff"), void)
        
        with h5py.File(os.path.join(fpath, "meta.hdf5"), 'w') as hf:
            hf.create_dataset("vol_shape", data = self.vol_shape)
            hf.create_dataset("sizes", data = self["sizes"])
            hf.create_dataset("cents", data = self["cents"])
            hf.create_dataset("cpts", data = self["cpts"])
            s = []
            for s_void in self["s_voids"]:
                sz, sy, sx = s_void
                s.append([sz.start, sz.stop, sy.start, sy.stop, sx.start, sx.stop])
            hf.create_dataset("s_voids", data = np.asarray(s))
            if np.any(self["x_boundary"]):
                hf.create_dataset("x_boundary", data = np.asarray(self["x_boundary"]))
            
        return

    
    def import_from_disk(self, fpath):

        import glob
        assert os.path.exists(fpath), "path not found to import voids data from"
        with h5py.File(os.path.join(fpath, "meta.hdf5"), 'r') as hf:
            self.vol_shape = tuple(np.asarray(hf["vol_shape"]))
            self["sizes"] = np.asarray(hf["sizes"][:])
            self["cents"] = np.asarray(hf["cents"][:])
            self["cpts"] = np.asarray(hf["cpts"][:])

            self["s_voids"] = []
            for s_void in np.asarray(hf["s_voids"]):
                self["s_voids"].append((slice(s_void[0], s_void[1]), slice(s_void[2], s_void[3]), slice(s_void[4], s_void[5])))

            if "x_boundary" in hf.keys():
                self["x_boundary"] = np.asarray(hf["x_boundary"][:])                
            else:
                self["x_boundary"] = []

        flist = sorted(glob.glob(os.path.join(fpath,"voids", "*.tiff")))
        self["x_voids"] = []
        for f_ in flist:
            self["x_voids"].append(imread(f_))

        return self

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
            void, n_objs = label_np(void)
            objs = find_objects(void)
            counts = [np.sum(void[objs[i]] == i+1) for i in range(n_objs)]
            if len(counts) > 1:
                i_main = np.argmax(counts)    
                void = (void == (i_main+1)).astype(np.uint8)            
            else:
                void = (void > 0).astype(np.uint8)

            self["s_voids"].append(s)
            self["sizes"].append(np.sum(void))
            self["cents"].append([int((s[i].start + s[i].stop)//2) for i in range(3)])
            self["cpts"].append([int((s[i].start)) for i in range(3)])
            self["x_voids"].append(void)

        return self

    def guess_voids(self, projs, theta, center, b, b_K, boundary_loc = (0,0,0)):

        # reconstruct and segment
        V_bin = coarse_segmentation(projs, theta, center, b_K, b, 0.5)
        Vl, n_det = label(V_bin)
        Vl = Vl.get()
        boundary_id = Vl[boundary_loc]
        slices = find_objects(Vl)
        print(f"voids found: {n_det}")

        self["sizes"] = []
        self["cents"] = []
        self["cpts"] = []
        self["x_voids"] = []
        self["s_voids"] = []
        for idx, s in enumerate(slices):
            void = (Vl[s] == idx+1)
            if idx + 1 == boundary_id:
                self["x_boundary"] = void.copy()
                continue
            self["s_voids"].append(s)
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

    def select_by_size(self, size_thresh, sel_type = "geq"):
        
        '''
        selection type could be geq or leq.  

        '''
        idxs = np.arange(len(self["sizes"]))
        cond_list = np.asarray([1 if self["sizes"][idx] >= size_thresh**3 else 0 for idx in idxs])
        if sel_type == "leq":
            cond_list = cond_list^1

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

    def export_void_mesh(self, rescale_fac = 1.0, decimate_fac = 1.0):

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
