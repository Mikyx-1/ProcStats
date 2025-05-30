import pybind11
from setuptools import Extension, setup

ext_modules = [
    Extension(
        "procstats",
        ["monitor.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    ),
]

setup(
    name="procstats",
    version="0.1",
    ext_modules=ext_modules,
)
