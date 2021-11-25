
# SparseSegmenter for Metrology and Void Analysis  

<p align="justify">Description goes here  </p>  

Code contributors: Aniket Tekawade and Viktor Nikitin, Argonne National Lab  
Data contributors: Xuan Zhang, Meimei Li, Argonne National Lab  


**Working notes:**

**Step 1, Particle detection:  **

1. reconstructs full scan at some binning = (2, 4, 8, 16, etc.)  
2. segments, labels all voids and returns the bounding boxes around the 'n' largest voids (ref: particle_saver.ipynb)  

**Step 2, Reconstruction and Segmentation around detected particles:  **
1. reconstructs the bounding boxes, then segments those voids  
2. convert 'n' voids to point cloud, then measure 2 size parameters (ref: export_particles.py)  


```
SparseSegmenter(projs,...)

    # bin the projections by some amount to obtain a fast, coarse reconstruction
    def recon_binned(self, yx_binning, proj_binning):
        return binned_vol

    def detect_particles(self,max_particles,binning_particle_counting,...): 
        # detect, then rescale. The particles one draws boxes around the particle. Must contain scalar value associated with particle id
        return p3d, p3d_particles

    def recon_patches_3d(self,projs, p3d,...):
        # pass patches object with the selected 3d patches representing the ROI (or ROIs). Patches must be organized on a regular (non-overlapping) grid.

    def _segment_patches(self, sub_vols,...):
        # segment the patches that were reconstructed

    def measure_sizes(self, p3d_particles):
        # measure sizes of the particles enclosed within the patches. Must ensure other particles coming into the FOV are neglected.

    def show_point_cloud(self, p3d_particles):
        # how to do this?        
```



<p align="center">atekawade [at] anl [dot] gov</p>  
