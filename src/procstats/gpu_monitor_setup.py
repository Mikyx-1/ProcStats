import os

import pybind11
from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Extension, setup

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "procstats_pid",
        [
            "monitor.cpp",  # Your C++ file
        ],
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_include(),
            # CUDA/NVML headers - adjust path as needed
            "/usr/local/cuda/include",
            "/usr/include/nvidia/gdk",  # Alternative NVML location
        ],
        libraries=["nvidia-ml"],  # Link against NVML
        library_dirs=[
            "/usr/local/cuda/lib64",
            "/usr/lib/x86_64-linux-gnu",  # Common location for libnvidia-ml.so
        ],
        cxx_std=11,
        define_macros=[("VERSION_INFO", '"dev"')],
    ),
]

setup(
    name="procstats_pid",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)
