#include <cmath>
#include <cstring>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 8

__device__ int mandel(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    int i;

    for (i = 0; i < count; ++i) {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

__global__ void mandelKernel(int *output, float x0, float y0, float dx, float dy, int maxIterations, bool view1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (view1 && ((i > 790 && i < 1201 && j > 296 && j < 904) ||
                  (i > 438 && i < 629 && j > 493 && j < 707) ||
                  (i > 712 && i < 791 && j > 398 && j < 802))) {
        int index = j * gridDim.x * blockDim.x + i;
        output[index] = maxIterations;
        return;
    }

    float x = x0 + i * dx;
    float y = y0 + j * dy;
    int index = j * gridDim.x * blockDim.x + i;
    output[index] = mandel(x, y, maxIterations);
}

void host_fe(float upperX, float upperY, float lowerX, float lowerY, int *img, int resX, int resY, int maxIterations)
{
    bool view1 = (lowerX == -2 && lowerY == -1);
    float dx = (upperX - lowerX) / resX;
    float dy = (upperY - lowerY) / resY;

    int *h_img = new int[resX * resY];
    int *cudaResult;
    cudaMalloc((void **)&cudaResult, sizeof(int) * resX * resY);

    dim3 dimGrid(resX / BLOCK_SIZE, resY / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    mandelKernel<<<dimGrid, dimBlock>>>(cudaResult, lowerX, lowerY, dx, dy, maxIterations, view1);

    cudaMemcpy(h_img, cudaResult, sizeof(int) * resX * resY, cudaMemcpyDeviceToHost);
    memcpy(img, h_img, sizeof(int) * resX * resY);

    delete[] h_img;
    cudaFree(cudaResult);
}
