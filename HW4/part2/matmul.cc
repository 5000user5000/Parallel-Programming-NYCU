#include <mpi.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// Compiler optimization hints
#pragma GCC optimize("O3", "unroll-loops", "omit-frame-pointer", "inline")
#pragma GCC target("avx2", "fma")

static int g_n, g_m, g_l;
static int g_rank, g_size;
static int g_local_rows;
static int *g_sendcounts = nullptr;
static int *g_displs = nullptr;
static int *g_local_out = nullptr; // Pre-allocated output buffer

void construct_matrices(
    int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr)
{
    g_n = n;
    g_m = m;
    g_l = l;

    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

    // Calculate rows per process (load balancing)
    int base_rows = n / g_size;
    int extra_rows = n % g_size;

    // Each process gets base_rows, and first 'extra_rows' processes get one more
    g_local_rows = base_rows + (g_rank < extra_rows ? 1 : 0);

    // Allocate aligned memory for better performance
    *a_mat_ptr = new int[g_local_rows * m];
    *b_mat_ptr = new int[m * l];

    // Pre-allocate output buffer to avoid allocation in multiply
    g_local_out = new int[g_local_rows * l];

    // Prepare sendcounts and displacements for Scatterv
    if (g_rank == 0) {
        g_sendcounts = new int[g_size];
        g_displs = new int[g_size];

        int offset = 0;
        for (int i = 0; i < g_size; i++) {
            int rows = base_rows + (i < extra_rows ? 1 : 0);
            g_sendcounts[i] = rows * m;
            g_displs[i] = offset;
            offset += rows * m;
        }
    }

    // Distribute rows of A to all processes
    int recvcount = g_local_rows * m;
    MPI_Scatterv(a_mat, g_sendcounts, g_displs, MPI_INT,
                 *a_mat_ptr, recvcount, MPI_INT,
                 0, MPI_COMM_WORLD);

    // Broadcast entire matrix B to all processes
    // Optimize: avoid unnecessary copy on rank 0
    if (g_rank == 0) {
        memcpy(*b_mat_ptr, b_mat, m * l * sizeof(int));
    }
    MPI_Bcast(*b_mat_ptr, m * l, MPI_INT, 0, MPI_COMM_WORLD);
}

void matrix_multiply(
    const int n, const int m, const int l, const int *__restrict__ a_mat,
    const int *__restrict__ b_mat, int *__restrict__ out_mat)
{
    // Use pre-allocated buffer
    int *__restrict__ local_out = g_local_out;

    // Adaptive block size based on matrix dimensions
    // For smaller matrices (dataset1: 300-500), use smaller blocks
    // For larger matrices (dataset2: 1000-2000), use larger blocks
    const int BLOCK_I = (n < 600) ? 32 : 64;
    const int BLOCK_J = (l < 600) ? 32 : 64;
    const int BLOCK_K = (m < 600) ? 32 : 64;

    // Initialize output to zero
    memset(local_out, 0, g_local_rows * l * sizeof(int));

    // Cache-blocked (tiled) matrix multiplication
    // This dramatically improves cache hit rate
    for (int kk = 0; kk < m; kk += BLOCK_K) {
        int k_end = std::min(kk + BLOCK_K, m);

        for (int jj = 0; jj < l; jj += BLOCK_J) {
            int j_end = std::min(jj + BLOCK_J, l);

            for (int ii = 0; ii < g_local_rows; ii += BLOCK_I) {
                int i_end = std::min(ii + BLOCK_I, g_local_rows);

                // Compute block
                for (int i = ii; i < i_end; i++) {
                    const int *__restrict__ a_row = &a_mat[i * m];
                    int *__restrict__ c_row = &local_out[i * l];

                    for (int j = jj; j < j_end; j++) {
                        const int *__restrict__ b_col = &b_mat[j * m];
                        int sum = c_row[j];

                        // Inner loop - most critical for performance
                        // Unroll manually for better performance
                        int k = kk;
                        for (; k + 3 < k_end; k += 4) {
                            sum += a_row[k] * b_col[k];
                            sum += a_row[k+1] * b_col[k+1];
                            sum += a_row[k+2] * b_col[k+2];
                            sum += a_row[k+3] * b_col[k+3];
                        }
                        // Handle remaining elements
                        for (; k < k_end; k++) {
                            sum += a_row[k] * b_col[k];
                        }

                        c_row[j] = sum;
                    }
                }
            }
        }
    }

    // Gather results back to rank 0
    int *recvcounts = nullptr;
    int *displs_out = nullptr;

    if (g_rank == 0) {
        recvcounts = new int[g_size];
        displs_out = new int[g_size];

        int base_rows = n / g_size;
        int extra_rows = n % g_size;
        int offset = 0;

        for (int i = 0; i < g_size; i++) {
            int rows = base_rows + (i < extra_rows ? 1 : 0);
            recvcounts[i] = rows * l;
            displs_out[i] = offset;
            offset += rows * l;
        }
    }

    MPI_Gatherv(local_out, g_local_rows * l, MPI_INT,
                out_mat, recvcounts, displs_out, MPI_INT,
                0, MPI_COMM_WORLD);

    if (g_rank == 0) {
        delete[] recvcounts;
        delete[] displs_out;
    }
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    delete[] a_mat;
    delete[] b_mat;
    delete[] g_local_out;

    if (g_rank == 0) {
        delete[] g_sendcounts;
        delete[] g_displs;
    }
}
