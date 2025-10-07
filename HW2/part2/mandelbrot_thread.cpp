#include <array>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "cycle_timer.h"

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

extern void mandelbrot_serial(float x0,
                              float y0,
                              float x1,
                              float y1,
                              int width,
                              int height,
                              int start_row,
                              int num_rows,
                              int max_iterations,
                              int *output);

//
// worker_thread_start --
//
// Thread entrypoint.
void worker_thread_start(WorkerArgs *const args)
{
    double start_time = CycleTimer::current_seconds();

    // 計算每個執行緒負責的行數
    int rows_per_thread = args->height / args->numThreads;
    int start_row = args->threadId * rows_per_thread;
    int num_rows = rows_per_thread;

    // 最後一個執行緒處理剩餘的行
    if (args->threadId == args->numThreads - 1) {
        num_rows = args->height - start_row;
    }

    // 呼叫 mandelbrot_serial 計算這個執行緒負責的部分
    mandelbrot_serial(
        args->x0, args->y0,
        args->x1, args->y1,
        args->width, args->height,
        start_row, num_rows,
        args->maxIterations,
        args->output
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
