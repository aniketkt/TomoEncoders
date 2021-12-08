'''
Feature extractor based on fCNN architecture that is known to respond to porosity changes.  
Authors:
Aniket Tekawade, ANL

'''

from tomo_encoders import Patches
import numpy as np


def get_features_somehow(vol_pair, p, **kwargs):
    
    n_features = 5
    
    x0 = p.extract(vol_pair[0])
    x1 = p.extract(vol_pair[1])
    emb_vec = fe.predict_embeddings(x1[...,np.newaxis] - x0[...,np.newaxis], n_features)
    
    # emb_vec shape must be (len_patches, n_features)
    
    return np.random.normal(0, 1, (len(p), n_features))

from tomo_encoders.neural_nets.enhancers import Enhancer_fCNN
class FeatureExtraction_fCNN(Enhancer_fCNN):
    def __init__(self, **kwargs):
        
        '''
        models : dict of tf.keras.Models.model 
            dict contains some models with model keys as string descriptors of them.

        '''
        super().__init__(**kwargs)
        
        return

    def print_layers(self, modelkey):
        print("#"*80)
        
        txt_out = []
        for ii in range(len(self.models[modelkey].layers)):
            lshape = str(self.models[modelkey].layers[ii].output_shape)
            lname = str(self.models[modelkey].layers[ii].name)
            txt_out.append(lshape + "    ::    "  + lname)
        print('\n'.join(txt_out))
        return
    
    def predict_embeddings(self, x, latent_dims):
        
        '''
        Here, we break the model into layers, and extract conv. feature maps at some layer, then flatten to form embedding vector.  
        '''
        raise NotImplementedError("not implemented yet for FeatureExtraction_fCNN")


if __name__ == "__main__":

    print("nothing here")

