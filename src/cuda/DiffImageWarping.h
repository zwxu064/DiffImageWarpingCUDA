/* -------------------------------------------------------------------------------------------------------------
// File: DiffImageWarping.h
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

#ifndef __DIFFIMAGEWARPING_H__
#define __DIFFIMAGEWARPING_H__

#include <math.h>
#include <stdio.h>
#include <iostream>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


#ifdef __cplusplus
    extern "C" {
#endif

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCKS_PER_GRID 65535

typedef unsigned int uint;
typedef unsigned char uchar;

// =============================================================================================================

void ImageWarpingCUDA(
    const at::Tensor image,
    const at::Tensor depth,
    at::Tensor left_count,
    at::Tensor right_count,
    at::Tensor left_image_stack,
    at::Tensor right_image_stack
);

// =============================================================================================================

void ImageWarpingBackCUDA(
    const at::Tensor image,
    const at::Tensor depth,
    const at::Tensor dleft_count,
    const at::Tensor dright_count,
    const at::Tensor dleft_image_stack,
    const at::Tensor dright_image_stack,
    at::Tensor ddepth,
    at::Tensor dimage
);

/*
    Params: image: (batch,height,width,channel)
    Params: depth: (batch,height,width)
    Returns: *_count: (batch,height,width)
    Returns: *_image_stack: (batch,height,width,channel)

    If *_count has 0 size, *_image_stack will be averaged;
    otherwise, original *_image_stack and *_count will be return.

    If *_image_stack has 0 size, no *_image will be generated;
    this is to control either or both of left and right images are required.

    Note: should require all tensors to be contiguous.
    Supported max image size:
        for grey, batch*height*width <= 1024*65535
        for RGB + inbuilt average, batch*height*width*channel <= 1024*65535
        for RGB + return count, batch*height*width <= 1024*65535
*/

// =============================================================================================================

void ImageWarping(
    const at::Tensor image,
    const at::Tensor depth,
    at::Tensor left_count,
    at::Tensor right_count,
    at::Tensor left_image_stack,
    at::Tensor right_image_stack
);

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
);

#ifdef __cplusplus
    }
#endif

#endif // __DIFFIMAGEWARPING_H__