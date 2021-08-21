#
# Zhiwei Xu <zhiwei.xu@anu.edu.au>
#

import shutil, glob, os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


if os.path.exists('build'):
    shutil.rmtree('build')

setup(name='DualPixel',
      ext_modules=[CUDAExtension(name='DualPixel',
                                 sources=['cuda/DualPixel.cpp',
                                          'cuda/DualPixelCUDA.cu'],
                                 extra_compile_args={'nvcc': ['-O3',
                                                              '-arch=sm_35',
                                                              '--expt-relaxed-constexpr'],
                                                     'cxx': ['-g',
                                                             '-std=c++11',
                                                             '-Wno-deprecated-declarations',
                                                             '-fopenmp']})],
      cmdclass={'build_ext': BuildExtension},
      include_dirs=['.'])

target_dir = 'cuda/lib_dualpixel'
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
