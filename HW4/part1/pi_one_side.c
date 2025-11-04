#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define SEED 1234

long long int compute_pi(long long int tosses, int world_rank, int world_size){
    // Calculate local count for this process
    long long int number_of_tosses = tosses / world_size;
    long long int count = 0;
    unsigned int seed = world_rank * SEED;

    if(world_rank == world_size - 1) {
        // Last process takes the remainder tosses
        number_of_tosses += tosses % world_size;
    }
    for (long long int toss = 0; toss < number_of_tosses; toss++)
    {
        double x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        if (x * x + y * y <= 1.0)
        {
            count++;
        }
    }
    return count;
}

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Each process computes its local count
    long long int local_count = compute_pi(tosses, world_rank, world_size);

    // Create window: each process exposes its own local_count
    MPI_Win_create(&local_count, sizeof(long long int), sizeof(long long int),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    long long int total_count = 0;

    if (world_rank == 0)
    {
        // Rank 0 actively fetches data from all processes using MPI_Get
        total_count = local_count;  // Include rank 0's own count

        for (int i = 1; i < world_size; i++)
        {
            long long int temp_count = 0;
            MPI_Win_lock(MPI_LOCK_SHARED, i, 0, win);
            MPI_Get(&temp_count, 1, MPI_LONG_LONG, i, 0, 1, MPI_LONG_LONG, win);
            MPI_Win_unlock(i, win);
            total_count += temp_count;
        }
    }

    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
        pi_result = 4.0 * (double)total_count / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
