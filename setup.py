#!/usr/bin/env python
from setuptools import find_packages, setup

import VERSION

setup(
    name='wavelet-ec',
    version=VERSION.__version__,
    url='https://github.com/pedrohenriquecoimbra/wavelete-ec',
    description=(
        "Wavelet-based Eddy Covariance "
        "Written by pedrohenriquecoimbra"),
    long_description=open('README.md').read(),
    keywords="EC, partitionning, wavelet",
    license='BSD',
    platforms=['linux'],
    packages=find_packages(exclude=['sample*', 'deprecated*']),
    include_package_data=True,
    install_requires=[
        'pandas>=2',
        'matplotlib',
        'numpy>=1.24',
        'PyWavelets>=1.6.0',
        'scipy>=1.13.0'
        'PyYAML',
    ],
    #extras_require={
    #    'oscar': ['matplotlib>=2.0,<4.0']
    #},
    # See http://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Other Environment',
        'Framework :: Jupyter',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',        
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.12',
        'Topic :: Other/Nonlisted Topic'],
)