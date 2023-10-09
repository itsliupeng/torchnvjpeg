from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='torchnvjpeg',
    version="0.1.0",
    description="Using gpu decode jpeg image.",
    author="itsliupeng",
    classifiers=[
        "Development Status :: 4 - Beta", "Environment :: GPU :: NVIDIA CUDA",
        "License :: OSI Approved :: BSD License", "Intended Audience :: Developers",
        "Intended Audience :: Science/Research", "Operating System :: POSIX :: Linux", "Programming Language :: C++",
        "Programming Language :: Python", "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering", "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development", "Topic :: Software Development :: Libraries"
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.7.0',
    ],
    ext_modules=[
        CUDAExtension(name='torchnvjpeg',
                      sources=['torchnvjpeg.cpp'],
                      # extra_compile_args=['-g', '-std=c++14', '-fopenmp'],
                      extra_compile_args=['-std=c++17'],
                      libraries=['nvjpeg'],
                      define_macros=[('PYBIND', None)]),
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=True)},
)
