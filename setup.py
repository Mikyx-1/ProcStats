from setuptools import find_packages, setup

setup(
    name="procstats",
    version="0.2.0",
    description="A Python package for monitoring CPU, RAM, and GPU resources with adaptive sampling",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Le Hoang Viet",
    author_email="lehoangviet2k@gmail.com",
    url="https://github.com/Mikyx-1/ProcStats",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "psutil>=5.9.0",
        "dill>=0.3.0"
    ],
    # REMOVE the entry_points section - it's handled by pyproject.toml
    extras_require={
        "gpu": [
            "pynvml>=11.0.0",  # Optional for NVIDIA GPU monitoring
        ],
        "test": ["pytest>=7.0.0"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)