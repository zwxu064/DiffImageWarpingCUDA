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
    vim ~/.compile.sh
    replace "~/Install/DualPixel" by "[your path]/Install/DualPixel" above
  ```
  
  - Compile CUDA library in "[your code root]/DualPixelCUDA/src" as below.
  Re-compiling will rewrite the library.
  ```
    ./compile.sh
    library will be stored in "[your code root]/DualPixelCUDA/src/CUDA/lib_dualpixel"
  ```

## Note
  If you have any questions, please contact zhiwei.xu@anu.edu.au.
  
  Cheers~