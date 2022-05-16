#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import glob
import subprocess
try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup


setup(
    name="uberMS",
    url="https://github.com/pcargile/uberMS",
    version="0.0",
    author="Phillip Cargile",
    author_email="pcargile@cfa.harvard.edu",
    packages=["uberMS",],
    license="LICENSE",
    description="Optimized MINESweeper",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    install_requires=["Payne", "misty", "astropy", "numpyro", "jax"],
)
