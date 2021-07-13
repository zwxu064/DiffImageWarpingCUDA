#ifndef __DUALPIXEL_H__
#define __DUALPIXEL_H__

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

void DepthMergeCUDA(const at::Tensor image,
                    const at::Tensor depth,
                    at::Tensor left_image_count,
                    at::Tensor right_image_count,
                    at::Tensor left_image_stack,
                    at::Tensor right_image_stack);

/*
    Params: image: (batch,height,width,channel)
    Params: depth: (batch,height,width)
    Returns: *_image_count: (batch,height,width)
    Returns: *_image_stack: (batch,height,width,channel)

    If *_image_count has 0 size, *_image_stack will be averaged;
    otherwise, original *_image_stack and *_image_count will be return.

    If *_image_stack has 0 size, no *_image will be generated;
    this is to control either or both of left and right images are required.

    Note: should require all tensors to be contiguous.
    Supported max image size: for grey, batch*height*width <= 1024*65535
                              for RGB + inbulit average, batch*height*width*channel <= 1024*65535
                              for RGB + return count, batch*height*width <= 1024*65535
*/
void DepthMerge(const at::Tensor image,
                const at::Tensor depth,
                at::Tensor left_image_count,
                at::Tensor right_image_count,
                at::Tensor left_image_stack,
                at::Tensor right_image_stack);

#ifdef __cplusplus
  }
#endif

#endif // __DUALPIXEL_H__