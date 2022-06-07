'''

Authors:
Aniket Tekawade, ANL
Arshad Zahangir Chowdhury, ANL

'''



# Arshad's open-tsne change detector (intensity based)
from tomo_encoders import Patches
from openTSNE import TSNE            
import numpy as np
detect_flag = True
psize = 16
thresh1 = 0.2
thresh2 = 0.3
std_thresh = 0.1
n_iter = 100

def tsne_func(f_arr, verbosity = False):
    # do some things
    tsne = TSNE(n_components = 1, perplexity=30, metric="euclidean", n_jobs=8, random_state=42, verbose=verbosity, n_iter=n_iter)
    embeddings = tsne.fit(f_arr)
    return embeddings

def change_detector(rec, n_pts = 5, verbosity = False):

    
    patch_size = tuple([psize]*3)
    p = Patches(rec.shape, initialize_by = 'regular-grid', patch_size = patch_size, n_points = None)
    sub_vols = p.extract(rec, patch_size)

    f1 = np.mean(sub_vols, axis = (1,2,3))
    f2 = np.std(sub_vols, axis = (1,2,3))
    f3 = (f1 > thresh1).astype(np.float32)
    f4 = (f1 > thresh2).astype(np.float32)
    f5 = (f2 > std_thresh).astype(np.float32)

    f_array = np.asarray([f1, f2, f3, f4, f5]).T
    embedding = tsne_func(f_array, verbosity = verbosity)
    
    p.add_features(embedding, names = ["embedding"])
    p = p.select_by_feature(n_pts, ife = 0, selection_by = "lowest")
    
    bboxes = p.slices()
    for bbox in bboxes:
        rec[tuple(bbox)] = 0.3*rec[tuple(bbox)]
    change_locations = p.centers()[:n_pts]
    
    return rec, change_locations

if __name__ == "__main__":

    import pdb; pdb.set_trace()
#    rec = np.zeros((250, 245, 245)) 
   
    rec = np.random.normal(0, 1.0, (753, 511, 511))

    change_locations = change_detector(rec, n_pts = 5, verbosity = True)


