from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "procstats.combined_procstats",
        ["src/procstats/cpp/combined_monitoring.cpp"],
        include_dirs=[
            "/usr/local/cuda/include",
            "/usr/include/nvidia/gdk",
            "/usr/include",
            "/usr/include/x86_64-linux-gnu",
        ],
        libraries=["nvidia-ml"],
        library_dirs=[
            "/usr/lib/x86_64-linux-gnu",
            "/usr/local/cuda/lib64",
            "/usr/lib/nvidia-535",
        ],
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"0.1.0"')],
        extra_compile_args=['-O3'],
    ),
    Pybind11Extension(
        "procstats.cpu_ram_monitoring",
        ["src/procstats/cpp/cpu_ram_monitoring.cpp"],
        include_dirs=[
            "/usr/include",
            "/usr/include/x86_64-linux-gnu",
        ],
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"0.1.0"')],
        extra_compile_args=['-O3'],
    ),
    Pybind11Extension(
        "procstats.gpu_monitoring",
        ["src/procstats/cpp/gpu_monitoring.cpp"],
        include_dirs=[
            "/usr/local/cuda/include",
            "/usr/include/nvidia/gdk",
            "/usr/include",
            "/usr/include/x86_64-linux-gnu",
        ],
        libraries=["nvidia-ml"],
        library_dirs=[
            "/usr/lib/x86_64-linux-gnu",
            "/usr/local/cuda/lib64",
            "/usr/lib/nvidia-535",
        ],
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"0.1.0"')],
        extra_compile_args=['-O3'],
    ),
]

setup(
    name="procstats",
    version="0.1.0",
    description="Combined CPU, RAM, and GPU resource monitoring with child process tracking",
    author="Le Hoang Viet",
    author_email="your.email@example.com",  # Update with your email
    url="https://github.com/yourusername/procstats",  # Update with your repo
    packages=["procstats", "procstats.scripts", "procstats.tests"],
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=[
        "pybind11>=2.6.0",
        "cloudpickle",
    ],
    zip_safe=False,
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: MIT License",  # Adjust if LICENSE differs
    ],
)