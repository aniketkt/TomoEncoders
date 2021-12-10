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

def compare_volumes(vol_pair, input_size, n_pts, verbosity = False, sort_by = "lowest"):

    
    #### possible arguments? #########
    
    model_key = 'leaky_re_lu_3'
    
    ##################################
    
    # feature extraction
    f0 = fe.get_features(vol_pair[0], \
                         'regular-grid-tomo', \
                         model_key, \
                         input_size, \
                         feature_binning)
    
    f1 = fe.get_features(vol_pair[1], \
                         'regular-grid-tomo', \
                         model_key, \
                         input_size, \
                         feature_binning)
    
    # feature reduction
    change_signal = tsne_func(f1-f0, verbosity = verbosity)
    p.add_features(change_signal, names = ["change_signal"])
    
    # ROI selection
    p_sel = p.select_by_feature(n_pts, ife = 0, selection_by = sort_by)
    
    return p_sel

def search_volume(vol, input_size, n_pts, verbosity = False, sort_by = "lowest"):

    
    #### possible arguments? #########
    model_key = 'leaky_re_lu_3'
    ##################################
    
    # feature extraction
    p = fe.get_features(vol, \
                         'regular-grid-tomo', \
                         model_key, \
                         input_size, \
                         feature_binning)
    
    # feature reduction
    unique_signal = tsne_func(f_array, verbosity = verbosity)
    p.add_features(unique_signal, names = ["unique_signal"])
    
    # ROI selection
    p_sel = p.select_by_feature(n_pts, ife = -1, selection_by = sort_by)

    return p_sel


if __name__ == "__main__":

    import pdb; pdb.set_trace()
#    rec = np.zeros((250, 245, 245)) 
   
    rec = np.random.normal(0, 1.0, (753, 511, 511))

    change_locations = change_detector(rec, n_pts = 5, verbosity = True)


