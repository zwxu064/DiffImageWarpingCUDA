#include "DualPixel.h"


#ifdef __cplusplus
  extern "C" {
#endif

__device__ void AssignRGB2Neighbour(const int height,
                                    const int width,
                                    const int channel,
                                    const int b,
                                    const int src_i,
                                    const int src_j,
                                    const int dst_i,
                                    const int dst_j,
                                    const bool enable_minus,
                                    const float* image,
                                    int* map_count,
                                    float* image_stack) {
    int src_rgb_offset = b * height * width * channel + src_i * width * channel + src_j * channel;
    int dst_rgb_offset = b * height * width * channel + dst_i * width * channel + dst_j * channel;
    int dst_pixel_index = b * height * width + dst_i * width + dst_j;

    for (int i = 0; i < channel; i++) {
        if (enable_minus)
            atomicAdd(&image_stack[dst_rgb_offset + i], -image[src_rgb_offset + i]);
        else
            atomicAdd(&image_stack[dst_rgb_offset + i], image[src_rgb_offset + i]);
    }

    if (enable_minus)
        atomicAdd(&map_count[dst_pixel_index], -1);
    else
        atomicAdd(&map_count[dst_pixel_index], 1);

    __syncthreads();
}

__global__ void AverageDepthStack(const int batch,
                                  const int height,
                                  const int width,
                                  const int channel,
                                  const int* map_count,
                                  float* image_stack) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int b = tid / (height * width * channel);
    int tid_mod = tid % (height * width * channel);
    int i = tid_mod / (width * channel);
    int tid_mod_mod = tid_mod % (width * channel);
    int j = tid_mod_mod / channel;
    int pixel_index = b * height * width + i * width + j;
    if (tid >= batch * height * width * channel || map_count[pixel_index] == 0) return;

    image_stack[tid] /= map_count[pixel_index];
    __syncthreads();
}

__global__ void DepthMergeKernel(const int batch,
                                 const int height,
                                 const int width,
                                 const int channel,
                                 const float* image,
                                 const float* depth,
                                 int* left_image_count,
                                 int* right_image_count,
                                 float* left_image_stack,
                                 float* right_image_stack) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch * height * width) return;

    int current_depth = depth[tid] / 2;
    int image_size = height * width;
    int b = tid / image_size;
    int tid_mod = tid % image_size;
    int i = tid_mod / width;
    int j = tid_mod % width;
    int y_1 = i - current_depth;
    int y_2 = i + current_depth;
    int z_1 = j;
    int z_2 = j + current_depth;

    y_1 = max(0, min(y_1, height - 1));
    y_2 = max(0, min(y_2, height - 1));

    if (y_1 == y_2)
        y_2 ++;

    if (z_1 == z_2)
        z_2 ++;

    z_1 = max(0, min(z_1, width - 1));
    z_2 = max(0, min(z_2, width - 1));
    __syncthreads();

    // Sync left image
    if (left_image_stack != nullptr) {
        // bottom->up->right-bottom->right-up
        AssignRGB2Neighbour(height, width, channel, b, i, j, y_1, z_1, false, image,
                            left_image_count, left_image_stack);
        AssignRGB2Neighbour(height, width, channel, b, i, j, y_2, z_1, true, image,
                            left_image_count, left_image_stack);
        AssignRGB2Neighbour(height, width, channel, b, i, j, y_1, z_2, true, image,
                            left_image_count, left_image_stack);
        AssignRGB2Neighbour(height, width, channel, b, i, j, y_2, z_2, false, image,
                            left_image_count, left_image_stack);
    }

    // Sync right image
    if (right_image_stack != nullptr) {
        z_1 = j;
        z_2 = j - current_depth;

        if (z_1 == z_2)
            z_2 --;

        z_1 = max(0, min(z_1, width - 1));
        z_2 = max(0, min(z_2, width - 1));

        AssignRGB2Neighbour(height, width, channel, b, i, j, y_1, z_1, false, image,
                            right_image_count, right_image_stack);
        AssignRGB2Neighbour(height, width, channel, b, i, j, y_2, z_1, true, image,
                            right_image_count, right_image_stack);
        AssignRGB2Neighbour(height, width, channel, b, i, j, y_1, z_2, true, image,
                            right_image_count, right_image_stack);
        AssignRGB2Neighbour(height, width, channel, b, i, j, y_2, z_2, false, image,
                            right_image_count, right_image_stack);
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
                    at::Tensor left_image_count,
                    at::Tensor right_image_count,
                    at::Tensor left_image_stack,
                    at::Tensor right_image_stack) {
    const int batch = image.size(0);
    const int height = image.size(1);
    const int width = image.size(2);
    const int channel = image.size(3);
    const float* image_ptr = image.data<float>();
    const float* depth_ptr = depth.data<float>();
    int* left_image_count_ptr = nullptr;
    int* right_image_count_ptr = nullptr;
    float* left_image_stack_ptr = nullptr;
    float* right_image_stack_ptr = nullptr;

    int* left_image_count_inbuilt = nullptr;
    int* right_image_count_inbuilt = nullptr;

    // Assign pixel values to neighbours
    int n_threads = batch * height * width;
    int n_blocks = int(n_threads + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

    if (left_image_stack.size(0) != 0) {
        left_image_stack_ptr = left_image_stack.data<float>();
        cudaMalloc((void**)&left_image_count_inbuilt, n_threads * sizeof(int));
        cudaMemset(left_image_count_inbuilt, 0, n_threads * sizeof(int));
    }

    if (right_image_stack.size(0) != 0) {
        right_image_stack_ptr = right_image_stack.data<float>();
        cudaMalloc((void**)&right_image_count_inbuilt, n_threads * sizeof(int));
        cudaMemset(right_image_count_inbuilt, 0, n_threads * sizeof(int));
    }

    DepthMergeKernel<<<n_blocks, MAX_THREADS_PER_BLOCK>>>(batch,
                                                          height,
                                                          width,
                                                          channel,
                                                          image_ptr,
                                                          depth_ptr,
                                                          left_image_count_inbuilt,
                                                          right_image_count_inbuilt,
                                                          left_image_stack_ptr,
                                                          right_image_stack_ptr);

    // Count maps
    if (left_image_count.size(0) != 0) {
        left_image_count_ptr = left_image_count.data<int>();
        cudaMemcpy((void**)left_image_count_ptr, left_image_count_inbuilt, n_threads * sizeof(int),
                   cudaMemcpyDeviceToHost);
    } else {
        n_threads = batch * height * width * channel;
        n_blocks = int(n_threads + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    
        if (left_image_stack_ptr != nullptr)
            AverageDepthStack<<<n_blocks, MAX_THREADS_PER_BLOCK>>>(batch,
                                                                   height,
                                                                   width,
                                                                   channel,
                                                                   left_image_count_inbuilt,
                                                                   left_image_stack_ptr);
    }

    if (right_image_count.size(0) != 0) {
        right_image_count_ptr = right_image_count.data<int>();
        cudaMemcpy((void**)right_image_count_ptr, right_image_count_inbuilt, n_threads * sizeof(int),
                   cudaMemcpyDeviceToHost);
    } else {
        n_threads = batch * height * width * channel;
        n_blocks = int(n_threads + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

        if (right_image_stack_ptr != nullptr)
            AverageDepthStack<<<n_blocks, MAX_THREADS_PER_BLOCK>>>(batch,
                                                                   height,
                                                                   width,
                                                                   channel,
                                                                   right_image_count_inbuilt,
                                                                   right_image_stack_ptr);
    }

    if (left_image_count_inbuilt != nullptr) cudaFree(left_image_count_inbuilt);
    if (right_image_count_inbuilt != nullptr) cudaFree(right_image_count_inbuilt);
}

#ifdef __cplusplus
  }
#endif