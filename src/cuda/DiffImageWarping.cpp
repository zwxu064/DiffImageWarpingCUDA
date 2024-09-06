/* -------------------------------------------------------------------------------------------------------------
// File: DiffImageWarping.cpp
// Project: Differentiable Image Warping (CUDA)
// Contributors:
//     Zhiwei Xu <zwxu064@gmail.com>
// 
// Copyright (c) 2024 Zhiwei Xu
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without
// limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
// conditions:
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial
// portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
// LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------------------------------------- */

#include <pybind11/pybind11.h>
#include "DiffImageWarping.h"

namespace py = pybind11;


#ifdef __cplusplus
  extern "C" {
#endif

// =============================================================================================================

void ImageWarping(
    const at::Tensor image,
    const at::Tensor depth,
    at::Tensor left_count,
    at::Tensor right_count,
    at::Tensor left_image_stack,
    at::Tensor right_image_stack
) {
      ImageWarpingCUDA(
          image,
          depth,
          left_count,
          right_count,
          left_image_stack,
          right_image_stack
      );
}

// =============================================================================================================

void ImageWarpingBack(
    const at::Tensor image,
    const at::Tensor depth,
    const at::Tensor dleft_count,
    const at::Tensor dright_count,
    const at::Tensor dleft_image_stack,
    const at::Tensor dright_image_stack,
    at::Tensor ddepth,
    at::Tensor dimage
) {
    ImageWarpingBackCUDA(
        image,
        depth,
        dleft_count,
        dright_count,
        dleft_image_stack,
        dright_image_stack,
        ddepth,
        dimage
    );
}

#ifdef __cplusplus
    }
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ImageWarping", &ImageWarping, "Image Warping (CUDA)");
    m.def("ImageWarpingBack", &ImageWarpingBack, "Image Warping Back (CUDA)");
}