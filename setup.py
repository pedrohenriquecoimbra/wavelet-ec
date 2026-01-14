#!/usr/bin/env python
from setuptools import find_packages, setup

from waveletec import version

setup(
    name='waveletec',
    version=version.__version__,
    url='https://github.com/pedrohenriquecoimbra/wavelete-ec',
    description=(
        "Wavelet-based Eddy Covariance "
        "Written by pedrohenriquecoimbra"),
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author='Pedro Henrique Coimbra',
    author_email='pedro-henrique.herig-coimbra@inrae.fr',
    keywords="EC, partitionning, wavelet",
    license='MIT',
    platforms=['any'],
    packages=find_packages(exclude=['sample*', 'deprecated*']),
    include_package_data=True,
    install_requires=[
        'pandas>=2.0.0',
        'matplotlib>=3.1.0',
        'numpy>=1.24',
        'PyWavelets>=1.4.0',
        'scipy>=1.10.0',
        'scikit-learn',
        'PyYAML',
    ],
    extras_require={
        'cwt': ['pycwt'],
        'fcwt': ['pycwt', 'fcwt'],
    },
    # See http://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Other Environment',
        'Framework :: Jupyter',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8', 
        'Programming Language :: Python :: 3.9', 
        'Programming Language :: Python :: 3.10', 
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Topic :: Other/Nonlisted Topic'],
    python_requires='>=3.8',
    entry_points={
        "console_scripts": [
            "waveletEC-eddypro_run=waveletec.main:eddypro_run",
            "waveletEC-partition=waveletec.main:partition",
            "waveletEC-integrate=waveletec.main:integrate",
            "waveletEC-exec=waveletec.main:exec",
        ],
    },
)