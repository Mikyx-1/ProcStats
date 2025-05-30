import pybind11
from setuptools import Extension, setup

ext_modules = [
    Extension(
        name="procstats_pid",
        sources=["monitor.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "/usr/local/cuda-12.6/targets/x86_64-linux/include",  # <- your nvml.h path
        ],
        library_dirs=[
            "/usr/lib/x86_64-linux-gnu",  # <- your real libnvidia-ml.so
        ],
        libraries=["nvidia-ml"],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3", "-Wall"],
    )
]

setup(
    name="procstats_pid",
    version="0.1",
    ext_modules=ext_modules,
)
