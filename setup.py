#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: atekawade
"""

from setuptools import setup, find_packages

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='tomo_encoders',
    url='https://github.com/aniketkt/TomoEncoders',
    author='Aniket Tekawade',
    author_email='atekawade@anl.gov',
    # Needed to actually package something
    packages= ['tomo_encoders'],
    # Needed for dependencies
    install_requires=['numpy', 'pandas', 'scipy', 'h5py', 'matplotlib', \
                      'opencv-python', 'porespy', \
                      'ConfigArgParse', 'tqdm', 'ipython'],
    version=open('VERSION').read().strip(),
    license='BSD',
    description='Latent encoding of pore morphology in 3D images',
#     long_description=open('README.md').read(),
)


