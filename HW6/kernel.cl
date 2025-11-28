__kernel void convolution(
    int filter_width,
    __constant float *filter,
    int image_height,
    int image_width,
    __global float *input_image,
    __global float *output_image
){
   
    // get global thread IDs
    int x = get_global_id(0); // cols
    int y = get_global_id(1); // rows

    // avoid out-of-bounds threads
    if (x >= image_width || y >= image_height)
        return;

    float sum = 0.0f;
    int half_filter_width = filter_width / 2;

    // Apply filter with zero-padding
    for (int fy = -half_filter_width; fy <= half_filter_width; fy++) {
        for (int fx = -half_filter_width; fx <= half_filter_width; fx++) {
            int image_y = y + fy;
            int image_x = x + fx;

            // Zero-padding: only add if within bounds
            if (image_y >= 0 && image_y < image_height &&
                image_x >= 0 && image_x < image_width) {
                float image_value = input_image[image_y * image_width + image_x];
                float filter_value = filter[(fy + half_filter_width) * filter_width +
                                           (fx + half_filter_width)];
                sum += image_value * filter_value;
            }
        }
    }

    output_image[y * image_width + x] = sum;
}
