#!/usr/bin/env python

"""
Setup file for CS6320 Project 4 - Depth Estimation using Stereo
Adapted from Argoverse API setup template.
"""

import sys
from codecs import open
from os import path
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Read README.md
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="CS6320-Proj4",
    version="1.0.0",
    description="Depth Estimation using Stereo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="computer-vision stereo-depth disparity",
    packages=find_packages(include=["proj4_code", "proj4_code.*"]),
    python_requires=">=3.8",
    install_requires=[
        "pytest",
        "torch",
        "numpy",
        "matplotlib",
        "Pillow"
    ],
)