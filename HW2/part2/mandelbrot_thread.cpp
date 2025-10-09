#include <array>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "cycle_timer.h"

// GCC Vector Extensions
typedef float v4sf __attribute__ ((vector_size (16)));
typedef int v4si __attribute__ ((vector_size (16)));

struct WorkerArgs
{
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
    double executionTime;
};


void mandelbrot_opt(float x0,
                       float y0,
                       float x1,
                       float y1,
                       int width,
                       int height,
                       int start_row,
                       int total_rows,
                       int max_iterations,
                       int *output,
                       int thread_id = 0,
                       int num_threads = 1)
{
    float dx = (x1 - x0) / (float)width;
    float dy = (y1 - y0) / (float)height;

    int end_row = start_row + total_rows;

    // 使用交錯分配：thread_id 處理第 thread_id, thread_id+num_threads, ... 行
    for (int j = start_row + thread_id; j < end_row; j += num_threads)
    {
        float y = y0 + ((float)j * dy);
        int row_base = j * width;

        int i = 0;
        // 向量化處理：每次 4 個點
        for (; i + 3 < width; i += 4)
        {
            // 準備 4 個點的初始值
            v4sf c_re = {
                x0 + ((float)i * dx),
                x0 + ((float)(i+1) * dx),
                x0 + ((float)(i+2) * dx),
                x0 + ((float)(i+3) * dx)
            };
            v4sf c_im = {y, y, y, y};
            v4sf z_re = c_re;
            v4sf z_im = c_im;

            int counts[4] = {0, 0, 0, 0};
            int active[4] = {1, 1, 1, 1};

            for (int iter = 0; iter < max_iterations; ++iter)
            {
                // 逐個檢查發散條件（與 serial 完全一致）
                int any_active = 0;
                for (int k = 0; k < 4; k++)
                {
                    if (active[k] && z_re[k] * z_re[k] + z_im[k] * z_im[k] > 4.0f)
                    {
                        active[k] = 0;
                    }
                    if (active[k])
                        any_active = 1;
                }

                if (!any_active)
                    break;

                // 向量運算更新 z
                v4sf new_re = (z_re * z_re) - (z_im * z_im);
                v4sf two = {2.0f, 2.0f, 2.0f, 2.0f};
                v4sf new_im = two * z_re * z_im;
                z_re = c_re + new_re;
                z_im = c_im + new_im;

                // 更新計數
                for (int k = 0; k < 4; k++)
                {
                    if (active[k])
                        counts[k]++;
                }
            }

            output[row_base + i]     = counts[0];
            output[row_base + i + 1] = counts[1];
            output[row_base + i + 2] = counts[2];
            output[row_base + i + 3] = counts[3];
        }

        // 處理剩餘的點
        for (; i < width; i++)
        {
            float x = x0 + ((float)i * dx);
            float c_re = x;
            float c_im = y;
            float z_re = c_re;
            float z_im = c_im;
            int count = 0;

            for (int iter = 0; iter < max_iterations; ++iter)
            {
                if (z_re * z_re + z_im * z_im > 4.0f)
                    break;

                float new_re = (z_re * z_re) - (z_im * z_im);
                float new_im = 2.0f * z_re * z_im;
                z_re = c_re + new_re;
                z_im = c_im + new_im;
                count++;
            }

            output[row_base + i] = count;
        }
    }
}



//
// worker_thread_start --
//
// Thread entrypoint.
void worker_thread_start(WorkerArgs *const args)
{
    double start_time = CycleTimer::current_seconds();

    // 直接調用 mandelbrot_opt，傳入 thread 資訊讓它自己處理交錯分配
    mandelbrot_opt(
        args->x0, args->y0,
        args->x1, args->y1,
        args->width, args->height,
        0, args->height,  // 處理所有行，但會根據 thread_id 做交錯分配
        args->maxIterations,
        args->output,
        args->threadId,
        args->numThreads
    );

    double end_time = CycleTimer::current_seconds();
    args->executionTime = end_time - start_time;
}

//
// mandelbrot_thread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrot_thread(int num_threads,
                       float x0,
                       float y0,
                       float x1,
                       float y1,
                       int width,
                       int height,
                       int max_iterations,
                       int *output)
{
    static constexpr int max_threads = 32;

    if (num_threads > max_threads)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", max_threads);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::array<std::thread, max_threads> workers;
    std::array<WorkerArgs, max_threads> args = {};

    for (int i = 0; i < num_threads; i++)
    {
        // TODO FOR PP STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = max_iterations;
        args[i].numThreads = num_threads;
        args[i].output = output;

        args[i].threadId = i;
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i = 1; i < num_threads; i++)
    {
        workers[i] = std::thread(worker_thread_start, &args[i]);
    }

    worker_thread_start(&args[0]);

    // join worker threads
    for (int i = 1; i < num_threads; i++)
    {
        workers[i].join();
    }

    // Print execution time for each thread
    printf("\nThread execution times:\n");
    for (int i = 0; i < num_threads; i++)
    {
        printf("  Thread %d: [%.3f] ms\n", i, args[i].executionTime * 1000);
    }
}
