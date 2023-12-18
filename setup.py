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
    packages=["uberMS",
              "uberMS.spots",
              "uberMS.dva",
              "uberMS.smes",
              "uberMS.utils"],
    license="LICENSE",
    description="Optimized MINESweeper",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    install_requires=["Payne", "misty", "astropy", "numpyro", "jax", "optax"],
)

# write top level __init__.py file with the correct absolute path to package repo
toplevelstr = ("""try:
    from ._version import __version__
except(ImportError):
    pass

from jax.config import config
config.update('jax_enable_x64', True)

from . import spots
from . import dva
from . import utils"""
)

with open('uberMS/__init__.py','w') as ff:
  ff.write(toplevelstr)
  ff.write('\n')
  ff.write("""__abspath__ = '{0}/'\n""".format(os.getcwd()))