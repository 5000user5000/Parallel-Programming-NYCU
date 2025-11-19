#include <cstring>
#include <cuda.h>

#define TILE_WIDTH 8

#define MANDEL_LOOP(N) \
    for (iter = 0; iter < N; iter++) { \
        zr2 = zr * zr, zi2 = zi * zi; \
        if (zr2 + zi2 > 4.0f) { \
            device_output[py * img_width + px] = iter; \
            return; \
        } \
        zi = ci + (2.0f * zr * zi); \
        zr = cr + (zr2 - zi2); \
    }

#define ITER_CASES X(100000) X(10000) X(1000) X(256)

__global__ void mandel_kernel(float x_start, float y_start, float x_step, float y_step,
                                    int *__restrict__ device_output, int img_width, int img_height,
                                    int max_iter)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    float cr = x_start + px * x_step;
    float ci = y_start + py * y_step;
    float zr = cr, zi = ci;
    float zr2, zi2;
    int iter;

#define X(N) if (max_iter == N) { _Pragma("unroll") MANDEL_LOOP(N) } else
    ITER_CASES
#undef X
    {
        MANDEL_LOOP(max_iter)
    }
    device_output[py * img_width + px] = max_iter;
}

void host_fe(float upperX, float upperY, float lowerX, float lowerY, int *img, int resX, int resY, int maxIterations)
{
    float x_step = (upperX - lowerX) / resX;
    float y_step = (upperY - lowerY) / resY;

    int total_pixels = resX * resY;
    int *host_output = new int[total_pixels];
    int *device_output;
    cudaMalloc(&device_output, total_pixels * sizeof(int));

    dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_dim(resX / TILE_WIDTH, resY / TILE_WIDTH);

    mandel_kernel<<<grid_dim, block_dim>>>(lowerX, lowerY, x_step, y_step,
                                                  device_output, resX, resY, maxIterations);

    cudaMemcpy(host_output, device_output, total_pixels * sizeof(int), cudaMemcpyDeviceToHost);
    memcpy(img, host_output, total_pixels * sizeof(int));

    delete[] host_output;
    cudaFree(device_output);
}
