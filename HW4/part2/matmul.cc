#include <mpi.h>
#include <cstdlib>

static int g_n, g_m, g_l;
static int g_rank, g_size;
static int g_local_rows;
static int *g_sendcounts = nullptr;
static int *g_displs = nullptr;

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

    // Allocate memory for local portion of A
    *a_mat_ptr = new int[g_local_rows * m];

    // Allocate memory for full matrix B (all processes need it)
    *b_mat_ptr = new int[m * l];

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
    if (g_rank == 0) {
        for (int i = 0; i < m * l; i++) {
            (*b_mat_ptr)[i] = b_mat[i];
        }
    }
    MPI_Bcast(*b_mat_ptr, m * l, MPI_INT, 0, MPI_COMM_WORLD);
}

void matrix_multiply(
    const int n, const int m, const int l, const int *a_mat, const int *b_mat, int *out_mat)
{
    // Allocate local output buffer
    int *local_out = new int[g_local_rows * l];

    // Perform local matrix multiplication
    // C[i][j] = sum(A[i][k] * B[k][j]) for k = 0..m-1
    // A is row-major: A[i][k] = a_mat[i*m + k]
    // B is column-major: B[k][j] = b_mat[j*m + k]
    // This makes accessing B cache-friendly in the inner loop

    for (int i = 0; i < g_local_rows; i++) {
        for (int j = 0; j < l; j++) {
            int sum = 0;
            for (int k = 0; k < m; k++) {
                sum += a_mat[i * m + k] * b_mat[j * m + k];
            }
            local_out[i * l + j] = sum;
        }
    }

    // Gather results back to rank 0
    // Calculate receive counts and displacements for output
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

    delete[] local_out;
    if (g_rank == 0) {
        delete[] recvcounts;
        delete[] displs_out;
    }
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    delete[] a_mat;
    delete[] b_mat;

    if (g_rank == 0) {
        delete[] g_sendcounts;
        delete[] g_displs;
    }
}
