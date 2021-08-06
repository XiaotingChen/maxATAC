#! /usr/bin/env python3
import os
from setuptools import setup, find_packages

VERSION = "0.1.3"
HERE = os.path.abspath(os.path.dirname(__file__))


def get_description():
    README = os.path.join(HERE, 'README.md')
    with open(README, 'r') as f:
        return f.read()


setup(name="maxatac",
      description="Neural networks for predicting TF binding using ATAC-seq",
      long_description=get_description(),
      long_description_content_type="text/markdown",
      version=VERSION,
      url="",
      download_url="",
      author="",
      author_email="",
      license="Apache-2.0",
      include_package_data=True,
      packages=find_packages(),
      install_requires=["tensorflow==2.5.0",
                        "tensorboard",
                        "biopython",
                        "py2bit==0.3.0",
                        "numpy==1.19.5",
                        "pyBigWig==0.3.17",
                        "pydot==1.4.1",
                        "matplotlib",
                        "scikit-learn==0.24.2",
                        "pybedtools==0.8.1",
                        "pandas==1.1.5",
                        "pyfiglet",
                        "h5py==3.1.0",
                        "grpcio==1.34.0",
                        "deeplift",
                        "seaborn",
                        "pyyaml",
                        "graphviz",
                        "shap @ git+https://github.com/AvantiShri/shap.git@master#egg=shap",
                        "modisco @ git+https://github.com/XiaotingChen/tfmodisco.git@0.5.9.2#egg-modisco"
                        ],
      zip_safe=False,
      scripts=["maxatac/bin/maxatac"],
      classifiers=[]
      )
