#include <cstdio>
#include <cstdlib>
#include <cuda.h>

__device__ int mandel_gpu(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i)
    {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = (z_re * z_re) - (z_im * z_im);
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    return i;
}


__global__ void mandel_kernel(int* d_img, int count, float lowerX, float lowerY, float stepX, float stepY, int resX, int resY)
{
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    float x = lowerX + thisX * stepX;
    float y = lowerY + thisY * stepY;

    if (thisX >= resX || thisY >= resY) return;
    int idx = thisY * resX + thisX;
    int i = mandel_gpu(x, y, count);
    d_img[idx] = i;
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

    int *h_img = new int[res_x * res_y];

    int *d_img;
    cudaMalloc((void **)&d_img, res_x * res_y * sizeof(int));

    dim3 blockSize(16, 16);
    dim3 gridSize((res_x + blockSize.x - 1) / blockSize.x, (res_y + blockSize.y - 1) / blockSize.y);
    mandel_kernel<<<gridSize, blockSize>>>(d_img, max_iterations, lower_x, lower_y, step_x, step_y, res_x, res_y);

    cudaMemcpy(h_img, d_img, res_x * res_y * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < res_x * res_y; i++)
    {
        img[i] = h_img[i];
    }
    delete[] h_img;
    cudaFree(d_img);
}
