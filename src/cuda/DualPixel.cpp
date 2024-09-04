//
// Zhiwei Xu <zhiwei.xu@anu.edu.au>
//

#include <pybind11/pybind11.h>
#include "DualPixel.h"

namespace py = pybind11;


#ifdef __cplusplus
  extern "C" {
#endif

void DepthMerge(
    const at::Tensor image,
    const at::Tensor depth,
    at::Tensor left_count,
    at::Tensor right_count,
    at::Tensor left_image_stack,
    at::Tensor right_image_stack
) {
      DepthMergeCUDA(
          image,
          depth,
          left_count,
          right_count,
          left_image_stack,
          right_image_stack
      );
}

void DepthMergeBack(
    const at::Tensor image,
    const at::Tensor depth,
    const at::Tensor dleft_count,
    const at::Tensor dright_count,
    const at::Tensor dleft_image_stack,
    const at::Tensor dright_image_stack,
    at::Tensor ddepth,
    at::Tensor dimage
) {
    DepthMergeBackCUDA(
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
    m.def("DepthMerge", &DepthMerge, "Depth Merge (CUDA)");
    m.def("DepthMergeBack", &DepthMergeBack, "Depth Merge Back (CUDA)");
}