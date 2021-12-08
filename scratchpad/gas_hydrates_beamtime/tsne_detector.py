'''

Authors:
Aniket Tekawade, ANL
Arshad Zahangir Chowdhury, ANL

'''



# Arshad's open-tsne change detector (intensity based)
from tomo_encoders import Patches
from openTSNE import TSNE            
import numpy as np
n_iter = 100

params = {'psize' : 16, 'thresh1' : 0.2, 'thresh2' : 0.3, 'std_thresh' : 0.1, 'n_iter' : 100}

def tsne_func(f_arr, verbosity = False):
    # do some things
    tsne = TSNE(n_components = 1, perplexity=30, metric="euclidean", n_jobs=8, random_state=42, verbose=verbosity, n_iter=n_iter)
    embeddings = tsne.fit(f_arr)
    return embeddings


def get_features_somehow(vol_pair, p, **kwargs):
    
    n_features = 5
    return np.random.normal(0, 1, (len(p), n_features))

def change_detector(vol_pair, patch_size, n_pts, verbosity = False, **kwargs):

    p = Patches(vol_pair[-1].shape, initialize_by = 'regular-grid', patch_size = patch_size, n_points = None)

    # feature extraction
    f_array = get_features_somehow(vol_pair, p, **kwargs)
    
    # feature reduction
    embedding = tsne_func(f_array, verbosity = verbosity)
    p.add_features(embedding, names = ["embedding"])
    
    # ROI selection
    p_sel = p.select_by_feature(n_pts, ife = 0, selection_by = "lowest")
    
#     # for visualizing only
#     rec = vol_pair[-1]
#     bboxes = p_sel.slices()
#     for bbox in bboxes:
#         rec[tuple(bbox)] = 0.3*rec[tuple(bbox)]
#     change_locations = p_sel.centers()[:n_pts]
    
    return p_sel

if __name__ == "__main__":

    import pdb; pdb.set_trace()
#    rec = np.zeros((250, 245, 245)) 
   
    rec = np.random.normal(0, 1.0, (753, 511, 511))

    change_locations = change_detector(rec, n_pts = 5, verbosity = True)


