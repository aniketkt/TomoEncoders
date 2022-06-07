'''

wrapper for change detector that is called by 3D solver based on tomostream
'''


from roi_utils.roi import roi_search, load_seg_nn, roi_search_subtraction, rescale_vol_for_NN
from roi_utils.patches import Patches
from roi_utils.voxel_processing import modified_autocontrast
from roi_utils.ADet4RoI import roi_search_ADet
import os
import numpy as np
detect_flag = True
p_fpath = '/data/2021-12/streaming_rois'
from datetime import datetime as dt

N_SELECT_MAX = 1
CYLINDER_WIDTH = 0.6
CYLINDER_HEIGHT = 0.7
SEARCH_DOWNSAMPLING = 2


#CHECK# this volume passed is a numpy array
def change_detector(vol_t1, vol_t2, model, mode_flag = 1):
    
    # some things to keep track of
    orig_shape = vol_t2
    upsample_fac = SEARCH_DOWNSAMPLING
    
    sbin = tuple([slice(None, None, SEARCH_DOWNSAMPLING)]*3)
    vol_t1 = vol_t1[sbin]
    vol_t2 = vol_t2[sbin]

    #to-do: if projection shape changes within gui, this will happen often.
    # Should we return without error?
    print_out = "%s not identical to %s"%(str(vol_t1.shape), str(vol_t2.shape))
    assert vol_t1.shape == vol_t2.shape, print_out

    # adjust contrast
    h = modified_autocontrast(vol_t1, s = 0.01, normalize_sampling_factor = 4) 
    vol_t1 = np.clip(vol_t1, *h)
    h = modified_autocontrast(vol_t2, s = 0.01, normalize_sampling_factor = 4) 
    vol_t2 = np.clip(vol_t2, *h)

    min_, max_ = vol_t1[::4,::4,::4].min(), vol_t1[::4,::4,::4].max()
    vol_t1 = 255.0*(vol_t1 - min_) / (max_ - min_ + 1.e-12)
    min_, max_ = vol_t2[::4,::4,::4].min(), vol_t2[::4,::4,::4].max()
    vol_t2 = 255.0*(vol_t2 - min_) / (max_ - min_ + 1.e-12)
    
    if mode_flag == 0:
        centers, importance, bbox_start, bbox_width, RoIs = roi_search_subtraction(vol_t1, vol_t2) # subtraction based
    elif mode_flag == 1:
        centers, importance, bbox_start, bbox_width, RoIs = roi_search(vol_t1, vol_t2, model=model, mbsz=8) # segmentation based
    elif mode_flag == 2:
        centers, importance, bbox_start, bbox_width, RoIs = roi_search_ADet(vol_t2) # model has been hard coded
    else:
        pass
    # Zliu - add line here to dump csv for trouble-shooting
    RoIs.to_csv(os.path.join(p_fpath, dt.now().strftime("%Y%m%d-%H%M%S") + ".csv"), index=False)

    if len(centers) == 0:
        return np.asarray([[np.asarray(vol_t2.shape)//2]])

    p = Patches(vol_t1.shape, initialize_by = "data", points = bbox_start, widths = bbox_width)
    p.add_features(importance, names = ["importance"])
    p.add_features(centers*upsample_fac, names = ["cent_z", "cent_y", "cent_x"])
    
    # rescale coordinates back to original shape
    #p = p.rescale(upsample_fac, orig_shape)
    p = p.filter_by_cylindrical_mask(mask_ratio = CYLINDER_WIDTH, height_ratio = CYLINDER_HEIGHT) 
    change_locations = p.select_by_feature(N_SELECT_MAX, ife = 0)

    patches_fname = os.path.join(p_fpath, dt.now().strftime("%Y%m%d-%H%M%S") + ".hdf5")
    p.dump(patches_fname)
    print("Done saving roi file to: %s" % patches_fname)
    
    # Nothing below this line should be edited!    
    #bboxes = p.slices()
    #for bbox in bboxes:
    #    vol_t2[tuple(bbox)] = 0.3*vol_t2[tuple(bbox)]
    change_locations = p.centers()
    
    return np.asarray(change_locations)*upsample_fac

if __name__ == "__main__":

    import pdb; pdb.set_trace()
#    rec = np.zeros((250, 245, 245)) 
   
    rec = np.random.normal(0, 1.0, (753, 511, 511))



