import os
import sys

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, get_cxx_compiler


if sys.platform.startswith("linux"):
    # Use PyTorch's active C++ driver even when Python was built on a different host.
    os.environ.setdefault("CXX", get_cxx_compiler())

compile_flags = ["/O2"] if sys.platform == "win32" else ["-O3", "-g0"]
link_flags = []
if torch.backends.openmp.is_available():
    if sys.platform.startswith("linux"):
        compile_flags.append("-fopenmp")
        link_flags.append("-fopenmp")
    elif sys.platform == "win32":
        compile_flags.append("/openmp")

setup(
    name="empool",
    packages=["EmpoolHead"],
    install_requires=["torch"],
    ext_modules=[
        CppExtension(
            "EmpoolHead._wilson_cpu",
            ["EmpoolHead/csrc/wilson.cpp"],
            extra_compile_args=compile_flags,
            extra_link_args=link_flags,
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
