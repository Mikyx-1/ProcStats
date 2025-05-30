import os

import pybind11
from setuptools import Extension, setup

# Find CUDA installation
cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
if not os.path.exists(cuda_home):
    cuda_home = "/usr/local/cuda-12.6"  # Your specific CUDA version

ext_modules = [
    Extension(
        name="procstats_pid",
        sources=["monitor.cpp"],
        include_dirs=[
            pybind11.get_include(),
            f"{cuda_home}/include",
            f"{cuda_home}/targets/x86_64-linux/include",
            "/usr/local/cuda/extras/CUPTI/include",  # CUPTI headers
        ],
        library_dirs=[
            "/usr/lib/x86_64-linux-gnu",
            f"{cuda_home}/lib64",
            f"{cuda_home}/targets/x86_64-linux/lib",
            "/usr/local/cuda/extras/CUPTI/lib64",  # CUPTI libraries
        ],
        libraries=[
            "nvidia-ml",
            "cudart",  # CUDA Runtime
            "cupti",  # CUPTI for profiling
            "cuda",  # CUDA Driver
        ],
        language="c++",
        extra_compile_args=[
            "-std=c++17",
            "-O3",
            "-Wall",
            "-DWITH_CUPTI",  # Enable CUPTI features
        ],
    )
]

setup(
    name="procstats_pid",
    version="0.2",
    ext_modules=ext_modules,
    zip_safe=False,
)
