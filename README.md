[![DOCS](https://readthedocs.org/projects/docs/badge/?version=latest)](https://tomotwin.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/aniketkt/TomoTwin/master?filepath=notebooks%2Fsynthesize_voids_and_inclusions.ipynb)

# TomoEncoders

<p align="justify">This code (1) trains a convolutional auto-encoder (CAE) against pairs of grayscale and binarized data (2) extracts the latent space of the CAE and projects it to 2D space. For certain configurations of the architecture it is possible to order the clusters based on porosity metrics. More details are in our upcoming paper at IEEE-ICIP 2021. See the jupyter notebooks section for results. Data will be available shortly.  </p>  

The architecture is defined with synthesis and analysis blocks inspired by the 3D U-net (note the skip connections).  

<p align="center">
  <img width="800" src="imgs/autoencoder_architecture.png">
</p>  

The encoder-decoder (segmenter or denoiser) is trained by sampling patches of data from the 3D image pairs (grayscale and binarized image) around random coordinates to generate training data. Then, the encoder part is separated, and latent vectors are projected to 2D by PCA. Once trained, patches can be sampled from a given list of coordinates in the grayscale volume to identify morphological similarities.  

<p align="center">
  <img width="800" src="imgs/representation_learning.png">
</p>  


## Installation  
To install using pip do this in your python environment:

```  
git clone git+https://github.com/aniketkt/TomoEncoders.git
cd TomoEncoders
pip install .
```  

<p align="center">atekawade [at] anl [dot] gov</p>  
