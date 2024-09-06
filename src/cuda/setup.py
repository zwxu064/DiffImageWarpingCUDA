# -------------------------------------------------------------------------------------------------------------
# File: setup.py
# Project: Differentiable Image Warping (CUDA)
# Contributors:
#     Zhiwei Xu <zwxu064@gmail.com>
# 
# Copyright (c) 2024 Zhiwei Xu
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# -------------------------------------------------------------------------------------------------------------

import shutil, glob, os

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# =============================================================================================================

if os.path.exists('build'):
    shutil.rmtree('build')

setup(
    name='DiffImageWarping',
    ext_modules=[
        CUDAExtension(
            name='DiffImageWarping',
            sources=['cuda/DiffImageWarping.cpp', 'cuda/DiffImageWarpingCUDA.cu'],
            extra_compile_args={
                'nvcc': ['-O3', '-arch=sm_35', '--expt-relaxed-constexpr'],
                'cxx': ['-g', '-std=c++14', '-Wno-deprecated-declarations', '-fopenmp']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    include_dirs=['.']
)

target_dir = 'cuda/lib_diffimagewarping'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)

for file in glob.glob('*.so') + glob.glob('*.egg-info'):
    file_path = os.path.join(target_dir, file)

    if os.path.exists(file_path):
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    shutil.move(file, target_dir)

if os.path.exists('build'):
    shutil.rmtree('build')