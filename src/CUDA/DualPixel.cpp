#include <pybind11/pybind11.h>
#include "DualPixel.h"

namespace py = pybind11;


#ifdef __cplusplus
  extern "C" {
#endif

// image: (batch,channel,height,width)
// depth_stack: (batch,channel,height,width,num_stack)
void DepthMerge(const at::Tensor image,
                const at::Tensor depth,
                at::Tensor left_image_count,
                at::Tensor right_image_count,
                at::Tensor left_image_stack,
                at::Tensor right_image_stack) {
    DepthMergeCUDA(image, depth, left_image_count, right_image_count,
                   left_image_stack, right_image_stack);
}

#ifdef __cplusplus
  }
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("DepthMerge", &DepthMerge, "Depth Merge (CUDA)");}