'''
Feature extractor based on fCNN architecture that is known to respond to porosity changes.  
Authors:
Aniket Tekawade, ANL

'''
import time
from tomo_encoders import Patches
import numpy as np
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tomo_encoders.neural_nets.enhancers import Enhancer_fCNN


def get_features_somehow(vol_pair, p, **kwargs):
    
    n_features = 5
    
    x0 = p.extract(vol_pair[0])
    x1 = p.extract(vol_pair[1])
    emb_vec = fe.predict_embeddings(x1[...,np.newaxis] - x0[...,np.newaxis], n_features)
    
    # emb_vec shape must be (len_patches, n_features)
    
    return np.random.normal(0, 1, (len(p), n_features))


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
    
    def get_interim_layer(self, model, layer_id):

        if type(layer_id) == str:
            layer_names = [layer.name for layer in model.layers]
            layer_idx = layer_names.index(layer_id)

        inp = model.input
        out = model.layers[layer_idx].output
        print(out.name)

        return Model(inputs = inp, outputs = out)
    
    
    def predict_embeddings(self, x, layer_name, chunk_size, TIMEIT = False):
        
        '''
        Here, we break the model into layers, and extract conv. feature maps at some layer, then flatten to form embedding vector.  
        '''
        
        if layer_name not in self.models.keys():
            self.models[layer_name] = self.get_interim_layer(self.models['enhancer'], layer_name)
            
        assert x.ndim == 5, "x must be 5-dimensional (batch_size, nz, ny, nx, 1)."
        
        t0 = time.time()
        print("call to keras predict, len(x) = %i, shape = %s, chunk_size = %i"%(len(x), str(x.shape[1:-1]), chunk_size))
        nb = len(x)
        nchunks = int(np.ceil(nb/chunk_size))
        nb_padded = nchunks*chunk_size
        padding = nb_padded - nb

        for k in range(nchunks):

            sb = slice(k*chunk_size , min((k+1)*chunk_size, nb))
            x_in = x[sb,...]

            if padding != 0:
                if k == nchunks - 1:
                    x_in = np.pad(x_in, \
                                  ((0,padding), (0,0), \
                                   (0,0), (0,0), (0,0)), mode = 'edge')
                x_out = self.models[layer_name].predict(x_in).reshape(len(x_in), -1)

                if k == nchunks -1:
                    x_out = x_out[:-padding,...]
            else:
                x_out = self.models[layer_name].predict(x_in).reshape(len(x_in), -1)
                
            if k == 0:
                latent_dims = x_out.shape[-1]
                out_arr = np.zeros((nb, latent_dims), dtype = np.float32) # use numpy since return from predict is numpy            
            
            out_arr[sb,...] = x_out
        
        print("shape of output array: ", out_arr.shape)
        t_unit = (time.time() - t0)*1000.0/nb
        
        if TIMEIT:
            print("inf. time p. input patch size %s = %.2f ms, nb = %i"%(str(x[0,...,0].shape), t_unit, nb))
            print("\n")
        return out_arr
            
    def get_features(self, vol, grid_key, model_key, input_size, feature_binning):
        
        if grid_key == "regular-grid-tomo":
            grid_key = 'regular-grid'
            cyl_flag = True
        else:
            cyl_flag = False
            
        p = Patches(vol_pair[-1].shape, initialize_by = grid_key, patch_size = input_size, n_points = None)
        if cyl_flag:
            p = p.filter_by_cylindrical_mask()
        
        x = p.extract(vol, input_size)
        emb_vec = self.predict_embeddings(x[0][...,np.newaxis], \
                                        model_key, 32, \
                                        TIMEIT = True)  

        # downsample features?
        emb_vec = emb_vec[:,::feature_binning]
        
        p.add_features(emb_vec)
        return p
        
if __name__ == "__main__":

    print("nothing here")

