# DualPixelCUDA

## Environment
  - PyTorch 1.1.0
  - Python 3.6.8
  - CUDAtoolkit 10.0.130
  - CUDA 10.0
  - MatLab (to generate test data)

## CUDA Library
  - Add library develop path in bashrc as
      ```
      [open bashrc] vim ~/.bashrc
      export PYTHONPATH=[your path]/Install/DualPixel:$PYTHONPATH
      [activate bashrc] source ~/.bashrc
      ```
  - Ensure "[your path]/Install/DualPixel" is created, then add it to library compiling setup as
  
      ```
      cd [your code root]/DualPixelCUDA/src
      vim .compile.sh
      replace "~/Install/DualPixel" by "[your path]/Install/DualPixel" above
      ```
  
  - Compile CUDA library in "[your code root]/DualPixelCUDA/src" as below.
  Re-compiling will rewrite the library.
      ```
      ./compile.sh
      library will be stored in "[your code root]/DualPixelCUDA/src/cuda/lib_dualpixel"
      ```

## Differentiable PyTorch Module
  - Forward method is on weighted pixel assigning to 4 neighbors by converting depth (disparity)
  to intensity weight (details are in
  [doc/CUDA_implementation.pdf](doc/CUDA_implementation.pdf)
  and [MatLab demo](src/matlab/simulator_dp_extrapol.m)
  in contrast to [non-differentiable nearest version](src/matlab/simulator_dp.m)), see below.
  
    ![alt text](doc/dual_pixel.jpg)
  
  - Module is in [src/dp_module.py](src/dp_module.py) with a demo in [src/unittest/test_dp_module.py](src/unittest/test_dp_module.py).
      ```
      cd src/unittest
      python test_dp_module.py
      ```

  - Time comparison with MatLab script [src/matlab/simulator_dp_extrapol.m](src/matlab/simulator_dp_extrapol.m)
  and Python script [src/python/simulator_dp.py](src/python/simulator_dp.py).
  
    a. Real data with size: (batch, channel, height, width)=(1, 3, 480, 640).
    To dump data for PyTorch and CUDA versions, run [script](src/matlab/simulator_dp_extrapol.m)
    which will generate "matlab.mat" in /data.
    Max value is 20.93.
    
    | | MatLab  | PyTorch GPU | CUDA
    --- | --- | --- | ---
    forward time| 170s | 2257s | 1.11ms
    max error | - | 0 | ~9.5e-6
  
    b. Random data with size: (batch, channel, height, width)=(8, 3, 32, 64).
    Our CUDA version is averaged by **1000 iterations**.
    
    Corresponding max values in forward is 107.02 and
    in backward 73565.67.
    This clarify the following max errors are caused by
    **round error of single-floating point operation**.
    For instance, with say around 8-digit precision,
    73565.67 has 3-digit
    tolerance in decimal; 107.02 has 5-digit tolerance in decimal.
    This is mainly caused by the order of sum operations in CUDA.
    
     | | PyTorch GPU | CUDA
     --- | --- | ---
     forward time| 75s| 0.41ms
     forward max error | - | ~1.5e-4
     backward time| 219s | 0.08ms
     backward max error | - | ~3.9e-3

## Note
  If you have any questions, please contact zhiwei.xu@anu.edu.au.
