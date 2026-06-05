from setuptools import find_packages, setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="qsimov",
    ext_modules=cythonize(
        ["qsimov/paths/*.pyx", "qsimov/*.pyx"], build_dir="build"
    ),
    packages=find_packages(),
    include_dirs=[np.get_include()],
)
