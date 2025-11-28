// EXTREME SPARSE OPTIMIZATION - Only for filter1, filter2, filter3
// Performance Target: Beat 1.26s!

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
    __global float *output_image,
    int filter_type
){
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    __local float tile[LOCAL_H][LOCAL_W];

    int group_x = get_group_id(0) * TILE_W;
    int group_y = get_group_id(1) * TILE_H;

    int half_filter = filter_width / 2;
    int tid = ty * TILE_W + tx;

    // Load tile - 3 passes
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

    if (tid < 20) {
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

    barrier(CLK_LOCAL_MEM_FENCE);

    // === EXTREME SPARSE OPTIMIZATION ===
    if (gx < image_width && gy < image_height) {
        int tile_y = ty + half_filter;
        int tile_x = tx + half_filter;

        float sum;

        // Direct computation - no intermediate variables, let compiler optimize
        if (filter_type == 1) {
            // Filter 1 (7x7): 6 values - pattern is symmetric
            // Values: 1, 1, 2, 2, 1, 1
            float t1 = tile[tile_y - 1][tile_x - 1];
            float t2 = tile[tile_y - 1][tile_x + 1];
            float t3 = tile[tile_y    ][tile_x - 1];
            float t4 = tile[tile_y    ][tile_x + 1];
            float t5 = tile[tile_y + 1][tile_x - 1];
            float t6 = tile[tile_y + 1][tile_x + 1];
            // Optimize: *2.0f = add twice
            sum = t1 + t2 + t3 + t3 + t4 + t4 + t5 + t6;

        } else if (filter_type == 2) {
            // Filter 2 (3x3): 3 values - all coefficient 1
            sum = tile[tile_y - 1][tile_x + 1]
                + tile[tile_y    ][tile_x    ]
                + tile[tile_y + 1][tile_x + 1];

        } else {  // filter_type == 3
            // Filter 3 (5x5): 8 values - all coefficient 1
            sum = tile[tile_y - 1][tile_x - 1]
                + tile[tile_y - 1][tile_x + 1]
                + tile[tile_y    ][tile_x - 1]
                + tile[tile_y    ][tile_x    ]
                + tile[tile_y    ][tile_x + 1]
                + tile[tile_y + 1][tile_x - 1]
                + tile[tile_y + 1][tile_x    ]
                + tile[tile_y + 1][tile_x + 1];
        }

        output_image[gy * image_width + gx] = sum;
    }
}
