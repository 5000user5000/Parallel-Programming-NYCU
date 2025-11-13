#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#define PIXELS_PER_THREAD 8
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8

__device__ inline bool in_main_cardioid_or_bulb(float c_re, float c_im)
{
    float q = (c_re - 0.25f) * (c_re - 0.25f) + c_im * c_im;
    if (q * (q + (c_re - 0.25f)) < 0.25f * c_im * c_im)
        return true;

    if ((c_re + 1.0f) * (c_re + 1.0f) + c_im * c_im < 0.0625f)
        return true;

    return false;
}

__device__ int mandel_gpu(float c_re, float c_im, int count)
{
    if (in_main_cardioid_or_bulb(c_re, c_im))
        return count;

    float z_re = c_re, z_im = c_im;
    int i;

    #pragma unroll 8
    for (i = 0; i < count; ++i)
    {
        float z_re2 = z_re * z_re;
        float z_im2 = z_im * z_im;

        if (z_re2 + z_im2 > 4.0f)
            break;

        float new_im = 2.0f * z_re * z_im + c_im;
        z_re = z_re2 - z_im2 + c_re;
        z_im = new_im;
    }
    return i;
}


__global__ void mandel_kernel(int* d_img, size_t pitch, int count, float lowerX, float lowerY, float stepX, float stepY, int resX, int resY)
{
    int base_x = (blockIdx.x * blockDim.x + threadIdx.x) * PIXELS_PER_THREAD;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    if (thisY >= resY) return;

    float y = lowerY + thisY * stepY;
    int* row = (int*)((char*)d_img + thisY * pitch);

    #pragma unroll
    for (int px = 0; px < PIXELS_PER_THREAD; px++)
    {
        int thisX = base_x + px;
        if (thisX >= resX) break;

        float x = lowerX + thisX * stepX;
        int i = mandel_gpu(x, y, count);
        row[thisX] = i;
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void host_fe(float upper_x,
             float upper_y,
             float lower_x,
             float lower_y,
             int *img, // Output image
             int res_x, // width
             int res_y, // height
             int max_iterations)
{
    float step_x = (upper_x - lower_x) / (float)res_x;
    float step_y = (upper_y - lower_y) / (float)res_y;

    int *h_img;
    cudaHostAlloc(&h_img, res_x * res_y * sizeof(int), cudaHostAllocDefault); // host 用 cudaHostAlloc, pinned memory

    int *d_img;
    size_t pitch;
    cudaMallocPitch(&d_img, &pitch, res_x * sizeof(int), res_y); // device 用 cudaMallocPitch, pitch 對齊

    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    int threads_x = (res_x + PIXELS_PER_THREAD - 1) / PIXELS_PER_THREAD;
    dim3 gridSize((threads_x + blockSize.x - 1) / blockSize.x, (res_y + blockSize.y - 1) / blockSize.y);
    mandel_kernel<<<gridSize, blockSize>>>(d_img, pitch, max_iterations, lower_x, lower_y, step_x, step_y, res_x, res_y);

    cudaMemcpy2D(h_img, res_x * sizeof(int), d_img, pitch, res_x * sizeof(int), res_y, cudaMemcpyDeviceToHost); // 使用 cudaMemcpy2D 進行 2D 複製
    for (int i = 0; i < res_x * res_y; i++)
    {
        img[i] = h_img[i];
    }
    cudaFreeHost(h_img); // 釋放 pinned memory
    cudaFree(d_img);
}
