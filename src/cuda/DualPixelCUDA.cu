//
// Zhiwei Xu <zhiwei.xu@anu.edu.au>
//

#include "DualPixel.h"

#ifdef __cplusplus
  extern "C" {
#endif

// ==== Forward
__device__ void AssignRGB2Neighbour(const int height,
                                    const int width,
                                    const int channel,
                                    const int b,
                                    const int i,
                                    const int j,
                                    const float y,
                                    const float z,
                                    const bool is_positive,
                                    const float* image,
                                    float* image_count,
                                    float* image_stack) {
    int src_rgb_offset = b * height * width * channel + i * width * channel + j * channel;
    int y_floor = floor(y);
    int y_ceil = ceil(y);
    int z_floor = floor(z);
    int z_ceil = ceil(z);

    for (int idx_y = 0; idx_y <= 1; idx_y++) {
        int y_int = y_floor;

        if (idx_y == 1)
            if (y_ceil == y_floor)  // dst_i is integral
                continue;
            else
                y_int = y_ceil;

        float y_weight = 1 - abs(y - y_int);
        int y_int_valid = max(0, min(y_int, height - 1));

        for (int idx_z = 0; idx_z <= 1; idx_z++) {
            int z_int = z_floor;

            if (idx_z == 1)
                if (z_ceil == z_floor)
                    continue;
                else
                    z_int = z_ceil;

            float z_weight = 1 - abs(z - z_int);
            int z_int_valid = max(0, min(z_int, width - 1));

            int dst_rgb_offset = b * height * width * channel + y_int_valid * width * channel + z_int_valid * channel;
            int count_index = b * height * width + y_int_valid * width + z_int_valid;
            float weight = y_weight * z_weight;
            if (!is_positive) weight = -weight;

            for (int k = 0; k < channel; k++)
                atomicAdd(&image_stack[dst_rgb_offset + k], weight * image[src_rgb_offset + k]);

            atomicAdd(&image_count[count_index], weight);
        }
    }

    __syncthreads();
}

__global__ void AverageDepthStack(const int batch,
                                  const int height,
                                  const int width,
                                  const int channel,
                                  const float* image_count,
                                  float* image_stack) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int b = tid / (height * width * channel);
    int tid_mod = tid % (height * width * channel);
    int i = tid_mod / (width * channel);
    int tid_mod_mod = tid_mod % (width * channel);
    int j = tid_mod_mod / channel;
    int pixel_index = b * height * width + i * width + j;
    if (tid >= batch * height * width * channel || image_count[pixel_index] == 0) return;

    image_stack[tid] /= image_count[pixel_index];
    __syncthreads();
}

__global__ void DepthMergeKernel(const int batch,
                                 const int height,
                                 const int width,
                                 const int channel,
                                 const float* image,
                                 const float* depth,
                                 float* left_count,
                                 float* right_count,
                                 float* left_image_stack,
                                 float* right_image_stack) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch * height * width) return;

    float current_depth = depth[tid];
    int image_size = height * width;
    int b = tid / image_size;
    int tid_mod = tid % image_size;
    int i = tid_mod / width;
    int j = tid_mod % width;
    float y_1 = i - current_depth;
    float y_2 = i + current_depth;
    float z_1 = j;
    float z_2 = j + current_depth;

    // Sync left image
    if (left_image_stack != nullptr) {
        // bottom->up->right-bottom->right-up
        AssignRGB2Neighbour(height, width, channel, b, i, j, y_1, z_1, true, image,
                            left_count, left_image_stack);
        AssignRGB2Neighbour(height, width, channel, b, i, j, y_2, z_1, false, image,
                            left_count, left_image_stack);
        AssignRGB2Neighbour(height, width, channel, b, i, j, y_1, z_2, false, image,
                            left_count, left_image_stack);
        AssignRGB2Neighbour(height, width, channel, b, i, j, y_2, z_2, true, image,
                            left_count, left_image_stack);
    }

    // Sync right image
    if (right_image_stack != nullptr) {
        z_1 = j;
        z_2 = j - current_depth;

        AssignRGB2Neighbour(height, width, channel, b, i, j, y_1, z_1, true, image,
                            right_count, right_image_stack);
        AssignRGB2Neighbour(height, width, channel, b, i, j, y_2, z_1, false, image,
                            right_count, right_image_stack);
        AssignRGB2Neighbour(height, width, channel, b, i, j, y_1, z_2, false, image,
                            right_count, right_image_stack);
        AssignRGB2Neighbour(height, width, channel, b, i, j, y_2, z_2, true, image,
                            right_count, right_image_stack);
    }
}

/*
    Note: Ensure all tensors are contiguous()
    Features: if *_image_stack has 0 size, then no *_image will be generated
    Params: image: (batch,height,width,channel)
    Params: depth: (batch,height,width)
    Returns: *_image_stack: (batch,height,width,channel)
*/
void DepthMergeCUDA(const at::Tensor image,
                    const at::Tensor depth,
                    at::Tensor left_count,
                    at::Tensor right_count,
                    at::Tensor left_image_stack,
                    at::Tensor right_image_stack) {
    const int batch = image.size(0);
    const int height = image.size(1);
    const int width = image.size(2);
    const int channel = image.size(3);
    const float* image_ptr = image.data<float>();
    const float* depth_ptr = depth.data<float>();
    float* left_count_ptr = nullptr;
    float* right_count_ptr = nullptr;
    float* left_image_stack_ptr = nullptr;
    float* right_image_stack_ptr = nullptr;
    float* left_count_inbuilt = nullptr;
    float* right_count_inbuilt = nullptr;

    // Assign pixel values to neighbours
    int n_threads = batch * height * width;
    int n_blocks = int(n_threads + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

    if (left_image_stack.size(0) != 0) {
        left_image_stack_ptr = left_image_stack.data<float>();
        cudaMalloc((void**)&left_count_inbuilt, n_threads * sizeof(float));
        cudaMemset(left_count_inbuilt, 0, n_threads * sizeof(float));
    }

    if (right_image_stack.size(0) != 0) {
        right_image_stack_ptr = right_image_stack.data<float>();
        cudaMalloc((void**)&right_count_inbuilt, n_threads * sizeof(float));
        cudaMemset(right_count_inbuilt, 0, n_threads * sizeof(float));
    }

    DepthMergeKernel<<<n_blocks, MAX_THREADS_PER_BLOCK>>>(batch,
                                                          height,
                                                          width,
                                                          channel,
                                                          image_ptr,
                                                          depth_ptr,
                                                          left_count_inbuilt,
                                                          right_count_inbuilt,
                                                          left_image_stack_ptr,
                                                          right_image_stack_ptr);

    // Count maps
    if (left_count.size(0) != 0) {
        left_count_ptr = left_count.data<float>();
        cudaMemcpy((void**)left_count_ptr, left_count_inbuilt, n_threads * sizeof(float),
                   cudaMemcpyDeviceToHost);
    } else {
        n_threads = batch * height * width * channel;
        n_blocks = int(n_threads + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    
        if (left_image_stack_ptr != nullptr)
            AverageDepthStack<<<n_blocks, MAX_THREADS_PER_BLOCK>>>(batch,
                                                                   height,
                                                                   width,
                                                                   channel,
                                                                   left_count_inbuilt,
                                                                   left_image_stack_ptr);
    }

    if (right_count.size(0) != 0) {
        right_count_ptr = right_count.data<float>();
        cudaMemcpy((void**)right_count_ptr, right_count_inbuilt, n_threads * sizeof(float),
                   cudaMemcpyDeviceToHost);
    } else {
        n_threads = batch * height * width * channel;
        n_blocks = int(n_threads + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

        if (right_image_stack_ptr != nullptr)
            AverageDepthStack<<<n_blocks, MAX_THREADS_PER_BLOCK>>>(batch,
                                                                   height,
                                                                   width,
                                                                   channel,
                                                                   right_count_inbuilt,
                                                                   right_image_stack_ptr);
    }

    if (left_count_inbuilt != nullptr) cudaFree(left_count_inbuilt);
    if (right_count_inbuilt != nullptr) cudaFree(right_count_inbuilt);
}

// ==== Backward
__device__ void AssignRGB2NeighbourBack(const int height,
                                        const int width,
                                        const int channel,
                                        const int b,
                                        const int i,
                                        const int j,
                                        const float y,
                                        const float z,
                                        const float dy_sign,
                                        const float dz_sign,
                                        const bool is_positive,
                                        const float* image,
                                        const float* dimage_count,
                                        const float* dimage_stack,
                                        float* ddepth) {
    int rgb_offset = b * height * width * channel + i * width * channel + j * channel;
    int ddepth_dst = b * height * width + i * width + j;
    int y_floor = floor(y);
    int y_ceil = ceil(y);
    int z_floor = floor(z);
    int z_ceil = ceil(z);
    float dy = 0;
    float dz = 0;

    for (int idx_y = 0; idx_y <= 1; idx_y++) {
        int y_int = y_floor;

        if (idx_y == 1)
            if (y_ceil == y_floor)
                continue;
            else
                y_int = y_ceil;

        float y_weight = 1 - abs(y - y_int);
        int y_int_valid = max(0, min(y_int, height - 1));  // rename not rewrite, y_int will be used below

        for (int idx_z = 0; idx_z <= 1; idx_z++) {
            int z_int = z_floor;

            if (idx_z == 1)
                if (z_ceil == z_floor)
                    continue;
                else
                    z_int = z_ceil;

            float z_weight = 1 - abs(z - z_int);
            int z_int_valid = max(0, min(z_int, width - 1));

            int dcount_index = b * height * width + y_int_valid * width + z_int_valid;
            int drgb_offset = b * height * width * channel + y_int_valid * width * channel + z_int_valid * channel;
            float dweight = dimage_count[dcount_index];

            for (int k = 0; k < channel; k++)
                dweight += image[rgb_offset + k] * dimage_stack[drgb_offset + k];

            if (!is_positive) dweight = -dweight;
            float dy_weight = dweight * z_weight;
            float dz_weight = dweight * y_weight;

            if (y_int > y)
                dy += dy_weight;
            else if (y_int < y)
                dy -= dy_weight;

            if (z_int > z)
                dz += dz_weight;
            else if (z_int < z)
                dz -= dz_weight;
        }
    }

    atomicAdd(&ddepth[ddepth_dst], dy_sign * dy + dz_sign * dz);
    __syncthreads();
}

__global__ void DepthMergeBackKernel(const int batch,
                                     const int height,
                                     const int width,
                                     const int channel,
                                     const float* image,
                                     const float* depth,
                                     const float* dleft_count,
                                     const float* dright_count,
                                     const float* dleft_image_stack,
                                     const float* dright_image_stack,
                                     float* ddepth) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch * height * width) return;

    float current_depth = depth[tid];
    int image_size = height * width;
    int b = tid / image_size;
    int tid_mod = tid % image_size;
    int i = tid_mod / width;
    int j = tid_mod % width;
    float y_1 = i - current_depth;
    float y_2 = i + current_depth;
    float z_1 = j;
    float z_2 = j + current_depth;

    // Sync left image back
    if (dleft_image_stack != nullptr) {
        AssignRGB2NeighbourBack(height, width, channel, b, i, j, y_1, z_1, -1, 0,
                                true, image, dleft_count, dleft_image_stack, ddepth);
        AssignRGB2NeighbourBack(height, width, channel, b, i, j, y_2, z_1, 1, 0,
                                false, image, dleft_count, dleft_image_stack, ddepth);
        AssignRGB2NeighbourBack(height, width, channel, b, i, j, y_1, z_2, -1, 1,
                                false, image, dleft_count, dleft_image_stack, ddepth);
        AssignRGB2NeighbourBack(height, width, channel, b, i, j, y_2, z_2, 1, 1,
                                true, image, dleft_count, dleft_image_stack, ddepth);
    }

    // Sync right image back
    if (dright_image_stack != nullptr) {
        z_1 = j;
        z_2 = j - current_depth;

        AssignRGB2NeighbourBack(height, width, channel, b, i, j, y_1, z_1, -1, 0,
                                true, image, dright_count, dright_image_stack, ddepth);
        AssignRGB2NeighbourBack(height, width, channel, b, i, j, y_2, z_1, 1, 0,
                                false, image, dright_count, dright_image_stack, ddepth);
        AssignRGB2NeighbourBack(height, width, channel, b, i, j, y_1, z_2, -1, -1,
                                false, image, dright_count, dright_image_stack, ddepth);
        AssignRGB2NeighbourBack(height, width, channel, b, i, j, y_2, z_2, 1, -1,
                                true, image, dright_count, dright_image_stack, ddepth);
    }
}

void DepthMergeBackCUDA(const at::Tensor image,
                        const at::Tensor depth,
                        const at::Tensor dleft_count,
                        const at::Tensor dright_count,
                        const at::Tensor dleft_image_stack,
                        const at::Tensor dright_image_stack,
                        at::Tensor ddepth) {
    const int batch = image.size(0);
    const int height = image.size(1);
    const int width = image.size(2);
    const int channel = image.size(3);
    const float* image_ptr = image.data<float>();
    const float* depth_ptr = depth.data<float>();
    float* ddepth_ptr = ddepth.data<float>();
    float* dleft_count_ptr = nullptr;
    float* dright_count_ptr = nullptr;
    float* dleft_image_stack_ptr = nullptr;
    float* dright_image_stack_ptr = nullptr;

    // Assign pixel values to neighbours back
    int n_threads = batch * height * width;
    int n_blocks = int(n_threads + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

    if (dleft_image_stack.size(0) != 0) {
        dleft_image_stack_ptr = dleft_image_stack.data<float>();
        dleft_count_ptr = dleft_count.data<float>();
    }

    if (dright_image_stack.size(0) != 0) {
        dright_image_stack_ptr = dright_image_stack.data<float>();
        dright_count_ptr = dright_count.data<float>();
    }

    DepthMergeBackKernel<<<n_blocks, MAX_THREADS_PER_BLOCK>>>(batch,
                                                              height,
                                                              width,
                                                              channel,
                                                              image_ptr,
                                                              depth_ptr,
                                                              dleft_count_ptr,
                                                              dright_count_ptr,
                                                              dleft_image_stack_ptr,
                                                              dright_image_stack_ptr,
                                                              ddepth_ptr);
}

#ifdef __cplusplus
  }
#endif
