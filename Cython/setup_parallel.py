from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("bcr_cython_parallel",
              ["bcr_cython_parallel.pyx"],
              include_dirs=[np.get_include()],
              language="c++")
]

setup(
    ext_modules = cythonize(extensions, compiler_directives={'language_level': "3"})
)

# RUN: python -m setup_parallel build_ext --inplace