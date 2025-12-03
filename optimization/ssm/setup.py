import numpy as np

from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("cstats.pyx"),
    include_dirs=[np.get_include()]
)