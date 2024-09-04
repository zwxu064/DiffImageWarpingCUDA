# DualPixelCUDA

This is the official implementation of "the reblur module" in "[Weakly-supervised Depth Estimation and Image Deblurring via Dual-Pixel Sensors]()".

## Requirements
```
PyTorch 1.1.0
Python 3.6.8
CUDAtoolkit 10.0.130
CUDA 10.0
MatLab (to generate test data)
```

## Compile for CUDA Library
- Add library develop path in bashrc as
    ```bash
    # Add library path to bashrc
    vim ~/.bashrc
    export PYTHONPATH=[path]/Install/DualPixel:$PYTHONPATH
    source ~/.bashrc
    ```
- Ensure "[path]/Install/DualPixel" is created, then add it to library compiling setup as

    ```bash
    cd [repository root]/src
    vim .compile.sh

    # Optional.
    replace "~/Install/DualPixel" by "[path]/Install/DualPixel"
    ```

- Compile CUDA library in "[repository root]/src" as below.
Re-compiling will rewrite the library.
    ```bash
    # Library will be stored in "[repository root]/src/cuda/lib_dualpixel"
    ./compile.sh
    ```

## Differentiable Layer in PyTorch
- Forward propagation is performed by assigning weighted intensity of a pixel to its 4 neighbors through converting depth (disparity)
to intensity weight (refer to
[implementation notes](doc/CUDA_implementation.pdf)
and [demo](src/matlab/simulator_dp_extrapol.m)
in contrast to [non-differentiable discrete version](src/matlab/simulator_dp.m)).

  <!-- ![alt text](doc/dual_pixel.jpg) -->
  <div style="text-align: center"><img src="doc/dual_pixel.jpg" width="300" /></div>

- Module is in [src/dp_module.py](src/dp_module.py) with a demo in [src/unittest/test_dp_module.py](src/unittest/test_dp_module.py).
    ```bash
    cd src/unittest
    python test_dp_module.py
    ```

- Time comparison with MatLab script [src/matlab/simulator_dp_extrapol.m](src/matlab/simulator_dp_extrapol.m)
and Python script [src/python/simulator_dp.py](src/python/simulator_dp.py).

  a) Real data with size: (batch, channel, height, width)=(1, 3, 480, 640).
  To dump data for PyTorch and CUDA versions, run [script](src/matlab/simulator_dp_extrapol.m) to generate "matlab.mat" in /data.
  Max value is 20.93.
  
  <center>

  | | MatLab  | PyTorch GPU | CUDA
  --- | --- | --- | ---
  Forward Time| 170s | 2257s | 1.11ms
  Forward Max Error | - | 0 | ~9.5e-6

  </center>

  b) Random data with size: (batch, channel, height, width)=(8, 3, 32, 64).
  Our CUDA version is averaged by **1000 iterations**.
  
  Corresponding max values in forward is 107.02 and
  in backward 73565.67.
  This clarify the following max errors are caused by
  **round error of single-floating point operation**.
  For instance, with say around 8-digit precision,
  73565.67 has 3-digit
  tolerance in decimal; 107.02 has 5-digit tolerance in decimal.
  This is mainly caused by the order of sum operations in CUDA.
  
  <center>

    | | PyTorch GPU | CUDA
    --- | --- | ---
    Forward Time| 75s| 0.41ms
    Forward Max Error | - | ~1.5e-4
    Backward Time| 219s | 0.08ms
    Backward Max Error | - | ~3.9e-3
  
  </center>

## Reference
If this repository is useful for you, please cite the paper below.
```bibtex
@misc{Pan2024,
    title        = {Weakly-supervised Depth Estimation and Image Deblurring via Dual-Pixel Sensors},
    author       = {Liyuan Pan and Richard Hartley and Liu Liu and Zhiwei Xu and Shah Chowdhury and Yan Yang and Hongguang Zhang and Hongdong Li and Miaomiao Liu},
    year         = {2024},
    howpublished = {IEEE Transactions on Pattern Analysis and Machine Intelligence}
}
```

## License
This code is distributed under the MIT License.

## Contact
For technical support of this library, please contact [Zhiwei Xu](mailto:zhiwei.xu@anu.edu.au).
For project support of the paper, please contact [Liyuan Pan](mailto:liyuan.pan@bit.edu.cn).