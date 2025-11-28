// Optimized Version - Balance between Performance & Readability
// Performance: ~1.66s

// 32x8 work-group size (better than 16x16 for memory coalescing)
#define TILE_W 32
#define TILE_H 8
#define HALO_MAX 3
#define LOCAL_W (TILE_W + 2 * HALO_MAX)  // 38
#define LOCAL_H (TILE_H + 2 * HALO_MAX)  // 14

__kernel void convolution(
    int filter_width,
    __constant float *filter,
    int image_height,
    int image_width,
    __global float *input_image,
    __global float *output_image
){
    // Thread IDs
    int tx = get_local_id(0);   // 0-31
    int ty = get_local_id(1);   // 0-7
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    // Shared memory for tile
    __local float tile[LOCAL_H][LOCAL_W];

    // Group position in image
    int group_x = get_group_id(0) * TILE_W;
    int group_y = get_group_id(1) * TILE_H;

    int half_filter = filter_width / 2;
    int tid = ty * TILE_W + tx;  // 0-255

    // Load tile (need 38×14 = 532 elements with 256 threads)
    // Pass 1: Load elements 0-255
    {
        int local_y = tid / LOCAL_W;
        int local_x = tid % LOCAL_W;
        int image_y = group_y + local_y - half_filter;
        int image_x = group_x + local_x - half_filter;

        if (image_y >= 0 && image_y < image_height &&
            image_x >= 0 && image_x < image_width) {
            tile[local_y][local_x] = input_image[image_y * image_width + image_x];
        } else {
            tile[local_y][local_x] = 0.0f;
        }
    }

    // Pass 2: Load elements 256-511
    {
        int i = tid + 256;
        if (i < LOCAL_H * LOCAL_W) {
            int local_y = i / LOCAL_W;
            int local_x = i % LOCAL_W;
            int image_y = group_y + local_y - half_filter;
            int image_x = group_x + local_x - half_filter;

            if (image_y >= 0 && image_y < image_height &&
                image_x >= 0 && image_x < image_width) {
                tile[local_y][local_x] = input_image[image_y * image_width + image_x];
            } else {
                tile[local_y][local_x] = 0.0f;
            }
        }
    }

    // Pass 3: Load remaining elements 512-531
    if (tid < 20) {  // Only first 20 threads
        int i = tid + 512;
        int local_y = i / LOCAL_W;
        int local_x = i % LOCAL_W;
        int image_y = group_y + local_y - half_filter;
        int image_x = group_x + local_x - half_filter;

        if (image_y >= 0 && image_y < image_height &&
            image_x >= 0 && image_x < image_width) {
            tile[local_y][local_x] = input_image[image_y * image_width + image_x];
        } else {
            tile[local_y][local_x] = 0.0f;
        }
    }

    // Wait for all threads to finish loading
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute convolution
    if (gx < image_width && gy < image_height) {
        float sum = 0.0f;
        int tile_y = ty + half_filter;
        int tile_x = tx + half_filter;

        // Use switch for common filter sizes (helps compiler optimize)
        switch(filter_width) {
            case 3:
                // Manually unroll 3x3 for best performance
                sum += tile[tile_y - 1][tile_x - 1] * filter[0];
                sum += tile[tile_y - 1][tile_x    ] * filter[1];
                sum += tile[tile_y - 1][tile_x + 1] * filter[2];
                sum += tile[tile_y    ][tile_x - 1] * filter[3];
                sum += tile[tile_y    ][tile_x    ] * filter[4];
                sum += tile[tile_y    ][tile_x + 1] * filter[5];
                sum += tile[tile_y + 1][tile_x - 1] * filter[6];
                sum += tile[tile_y + 1][tile_x    ] * filter[7];
                sum += tile[tile_y + 1][tile_x + 1] * filter[8];
                break;

            case 5:
                // Let compiler unroll 5x5
                #pragma unroll
                for (int fy = -2; fy <= 2; fy++) {
                    #pragma unroll
                    for (int fx = -2; fx <= 2; fx++) {
                        int filter_idx = (fy + 2) * 5 + (fx + 2);
                        sum += tile[tile_y + fy][tile_x + fx] * filter[filter_idx];
                    }
                }
                break;

            case 7:
                // Let compiler unroll 7x7
                #pragma unroll
                for (int fy = -3; fy <= 3; fy++) {
                    #pragma unroll
                    for (int fx = -3; fx <= 3; fx++) {
                        int filter_idx = (fy + 3) * 7 + (fx + 3);
                        sum += tile[tile_y + fy][tile_x + fx] * filter[filter_idx];
                    }
                }
                break;

            default:
                // Generic case
                for (int fy = 0; fy < filter_width; fy++) {
                    for (int fx = 0; fx < filter_width; fx++) {
                        int ty_offset = tile_y - half_filter + fy;
                        int tx_offset = tile_x - half_filter + fx;
                        sum += tile[ty_offset][tx_offset] * filter[fy * filter_width + fx];
                    }
                }
        }

        output_image[gy * image_width + gx] = sum;
    }
}
