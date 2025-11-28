#include "host_fe.h"
#include "helper.h"
#include <stdio.h>
#include <stdlib.h>

void host_fe(int filter_width,
             float *filter,
             int image_height,
             int image_width,
             float *input_image,
             float *output_image,
             cl_device_id *device,
             cl_context *context,
             cl_program *program)
{
    cl_int status;
    int filter_size = filter_width * filter_width;
    int image_size = image_height * image_width;

    // Create command queue
    cl_command_queue command_queue = clCreateCommandQueue(*context, *device, 0, &status);
    CHECK(status, "clCreateCommandQueue");

    // Create memory buffers
    cl_mem filter_buffer = clCreateBuffer(*context, CL_MEM_READ_ONLY,
                                         filter_size * sizeof(float), NULL, &status);
    CHECK(status, "clCreateBuffer filter");

    cl_mem input_buffer = clCreateBuffer(*context, CL_MEM_READ_ONLY,
                                        image_size * sizeof(float), NULL, &status);
    CHECK(status, "clCreateBuffer input");

    cl_mem output_buffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY,
                                         image_size * sizeof(float), NULL, &status);
    CHECK(status, "clCreateBuffer output");

    // Write data to device
    status = clEnqueueWriteBuffer(command_queue, filter_buffer, CL_TRUE, 0,
                                 filter_size * sizeof(float), filter, 0, NULL, NULL);
    CHECK(status, "clEnqueueWriteBuffer filter");

    status = clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE, 0,
                                 image_size * sizeof(float), input_image, 0, NULL, NULL);
    CHECK(status, "clEnqueueWriteBuffer input");

    // Create kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
    CHECK(status, "clCreateKernel");

    // Set kernel arguments
    status = clSetKernelArg(kernel, 0, sizeof(int), &filter_width);
    CHECK(status, "clSetKernelArg 0");

    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &filter_buffer);
    CHECK(status, "clSetKernelArg 1");

    status = clSetKernelArg(kernel, 2, sizeof(int), &image_height);
    CHECK(status, "clSetKernelArg 2");

    status = clSetKernelArg(kernel, 3, sizeof(int), &image_width);
    CHECK(status, "clSetKernelArg 3");

    status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &input_buffer);
    CHECK(status, "clSetKernelArg 4");

    status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &output_buffer);
    CHECK(status, "clSetKernelArg 5");

    // Execute kernel
    size_t global_work_size[2] = {image_width, image_height};
    status = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size,
                                   NULL, 0, NULL, NULL);
    CHECK(status, "clEnqueueNDRangeKernel");

    // Read result back
    status = clEnqueueReadBuffer(command_queue, output_buffer, CL_TRUE, 0,
                                image_size * sizeof(float), output_image, 0, NULL, NULL);
    CHECK(status, "clEnqueueReadBuffer");

    // Cleanup
    clReleaseMemObject(filter_buffer);
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
}
