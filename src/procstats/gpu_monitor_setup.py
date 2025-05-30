from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        name="procstats_gpu",
        sources=["monitor.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "/usr/local/cuda-12.6/targets/x86_64-linux/include",  # nvml.h path
        ],
        library_dirs=[
            "/usr/lib/x86_64-linux-gnu",  # real libnvidia-ml.so
        ],
        libraries=["nvidia-ml"],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3", "-Wall"],
    ),
]

setup(
    name="procstats_gpu",
    version="0.1",
    ext_modules=ext_modules,
)
