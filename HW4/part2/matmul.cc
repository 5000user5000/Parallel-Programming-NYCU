#include <mpi.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#pragma GCC optimize("O3", "unroll-loops", "omit-frame-pointer", "inline", "fast-math")

static int g_n, g_m, g_l;
static int g_rank, g_size;
static int g_local_rows;
static int *g_sendcounts = nullptr;
static int *g_displs = nullptr;
static int *g_local_out = nullptr;
static int *g_packed_a = nullptr;  // Packed A blocks
static int *g_packed_b = nullptr;  // Packed B blocks

void construct_matrices(
    int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr)
{
    g_n = n;
    g_m = m;
    g_l = l;

    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

    int base_rows = n / g_size;
    int extra_rows = n % g_size;
    g_local_rows = base_rows + (g_rank < extra_rows ? 1 : 0);

    // Allocate with alignment for better cache performance
    if (posix_memalign((void**)a_mat_ptr, 64, g_local_rows * m * sizeof(int)) != 0 ||
        posix_memalign((void**)b_mat_ptr, 64, m * l * sizeof(int)) != 0 ||
        posix_memalign((void**)&g_local_out, 64, g_local_rows * l * sizeof(int)) != 0) {
        // Fallback to regular allocation if alignment fails
        *a_mat_ptr = new int[g_local_rows * m];
        *b_mat_ptr = new int[m * l];
        g_local_out = new int[g_local_rows * l];
    }

    // Allocate packing buffers
    int max_pack_a = 256 * m;  // Max BLOCK_I * m
    int max_pack_b = m * 256;  // Max m * BLOCK_J
    if (posix_memalign((void**)&g_packed_a, 64, max_pack_a * sizeof(int)) != 0 ||
        posix_memalign((void**)&g_packed_b, 64, max_pack_b * sizeof(int)) != 0) {
        g_packed_a = new int[max_pack_a];
        g_packed_b = new int[max_pack_b];
    }

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

    // Scatter A rows
    MPI_Scatterv(a_mat, g_sendcounts, g_displs, MPI_INT,
                 *a_mat_ptr, g_local_rows * m, MPI_INT, 0, MPI_COMM_WORLD);

    // B is already in column-major format (b_mat[j*m+k] = B[k][j])
    // This is optimal for our access pattern! Just broadcast directly.
    MPI_Request req;
    if (g_rank == 0) {
        memcpy(*b_mat_ptr, b_mat, m * l * sizeof(int));
        MPI_Ibcast(*b_mat_ptr, m * l, MPI_INT, 0, MPI_COMM_WORLD, &req);
    } else {
        MPI_Ibcast(*b_mat_ptr, m * l, MPI_INT, 0, MPI_COMM_WORLD, &req);
    }
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

// Pack A block for better cache locality: pack A[i:i+rows][kk:kk+k_len] into contiguous memory
inline void pack_a_block(const int *a_mat, int *packed, int i_start, int rows, int kk, int k_len, int m) {
    for (int i = 0; i < rows; i++) {
        const int *src = &a_mat[(i_start + i) * m + kk];
        int *dst = &packed[i * k_len];
        memcpy(dst, src, k_len * sizeof(int));
    }
}

// Pack B block: B is column-major, pack B[kk:kk+k_len][jj:jj+cols]
inline void pack_b_block(const int *b_mat, int *packed, int kk, int k_len, int jj, int cols, int m) {
    for (int j = 0; j < cols; j++) {
        const int *src = &b_mat[(jj + j) * m + kk];
        int *dst = &packed[j * k_len];
        memcpy(dst, src, k_len * sizeof(int));
    }
}

// Optimized 6x8 micro-kernel with packed buffers
// Process 6 rows of A and 8 columns of B at once
inline void matmul_kernel_6x8(
    const int *__restrict__ packed_a, const int *__restrict__ packed_b,
    int *__restrict__ c_base, int k_len, int ldc)
{
    // Load existing C values
    int c00=c_base[0], c01=c_base[1], c02=c_base[2], c03=c_base[3];
    int c04=c_base[4], c05=c_base[5], c06=c_base[6], c07=c_base[7];
    int c10=c_base[ldc], c11=c_base[ldc+1], c12=c_base[ldc+2], c13=c_base[ldc+3];
    int c14=c_base[ldc+4], c15=c_base[ldc+5], c16=c_base[ldc+6], c17=c_base[ldc+7];
    int c20=c_base[2*ldc], c21=c_base[2*ldc+1], c22=c_base[2*ldc+2], c23=c_base[2*ldc+3];
    int c24=c_base[2*ldc+4], c25=c_base[2*ldc+5], c26=c_base[2*ldc+6], c27=c_base[2*ldc+7];
    int c30=c_base[3*ldc], c31=c_base[3*ldc+1], c32=c_base[3*ldc+2], c33=c_base[3*ldc+3];
    int c34=c_base[3*ldc+4], c35=c_base[3*ldc+5], c36=c_base[3*ldc+6], c37=c_base[3*ldc+7];
    int c40=c_base[4*ldc], c41=c_base[4*ldc+1], c42=c_base[4*ldc+2], c43=c_base[4*ldc+3];
    int c44=c_base[4*ldc+4], c45=c_base[4*ldc+5], c46=c_base[4*ldc+6], c47=c_base[4*ldc+7];
    int c50=c_base[5*ldc], c51=c_base[5*ldc+1], c52=c_base[5*ldc+2], c53=c_base[5*ldc+3];
    int c54=c_base[5*ldc+4], c55=c_base[5*ldc+5], c56=c_base[5*ldc+6], c57=c_base[5*ldc+7];

    // Packed A: row-major, packed_a[i * k_len + k]
    // Packed B: column-major, packed_b[j * k_len + k]

    for (int k = 0; k < k_len; k++) {
        __builtin_prefetch(&packed_a[k + 64], 0, 3);
        __builtin_prefetch(&packed_b[k + 64], 0, 3);

        int a0 = packed_a[0 * k_len + k];
        int a1 = packed_a[1 * k_len + k];
        int a2 = packed_a[2 * k_len + k];
        int a3 = packed_a[3 * k_len + k];
        int a4 = packed_a[4 * k_len + k];
        int a5 = packed_a[5 * k_len + k];

        int b0 = packed_b[0 * k_len + k];
        int b1 = packed_b[1 * k_len + k];
        int b2 = packed_b[2 * k_len + k];
        int b3 = packed_b[3 * k_len + k];
        int b4 = packed_b[4 * k_len + k];
        int b5 = packed_b[5 * k_len + k];
        int b6 = packed_b[6 * k_len + k];
        int b7 = packed_b[7 * k_len + k];

        c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
        c04 += a0 * b4; c05 += a0 * b5; c06 += a0 * b6; c07 += a0 * b7;

        c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
        c14 += a1 * b4; c15 += a1 * b5; c16 += a1 * b6; c17 += a1 * b7;

        c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
        c24 += a2 * b4; c25 += a2 * b5; c26 += a2 * b6; c27 += a2 * b7;

        c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
        c34 += a3 * b4; c35 += a3 * b5; c36 += a3 * b6; c37 += a3 * b7;

        c40 += a4 * b0; c41 += a4 * b1; c42 += a4 * b2; c43 += a4 * b3;
        c44 += a4 * b4; c45 += a4 * b5; c46 += a4 * b6; c47 += a4 * b7;

        c50 += a5 * b0; c51 += a5 * b1; c52 += a5 * b2; c53 += a5 * b3;
        c54 += a5 * b4; c55 += a5 * b5; c56 += a5 * b6; c57 += a5 * b7;
    }

    // Write back
    c_base[0]=c00; c_base[1]=c01; c_base[2]=c02; c_base[3]=c03;
    c_base[4]=c04; c_base[5]=c05; c_base[6]=c06; c_base[7]=c07;
    c_base[ldc]=c10; c_base[ldc+1]=c11; c_base[ldc+2]=c12; c_base[ldc+3]=c13;
    c_base[ldc+4]=c14; c_base[ldc+5]=c15; c_base[ldc+6]=c16; c_base[ldc+7]=c17;
    c_base[2*ldc]=c20; c_base[2*ldc+1]=c21; c_base[2*ldc+2]=c22; c_base[2*ldc+3]=c23;
    c_base[2*ldc+4]=c24; c_base[2*ldc+5]=c25; c_base[2*ldc+6]=c26; c_base[2*ldc+7]=c27;
    c_base[3*ldc]=c30; c_base[3*ldc+1]=c31; c_base[3*ldc+2]=c32; c_base[3*ldc+3]=c33;
    c_base[3*ldc+4]=c34; c_base[3*ldc+5]=c35; c_base[3*ldc+6]=c36; c_base[3*ldc+7]=c37;
    c_base[4*ldc]=c40; c_base[4*ldc+1]=c41; c_base[4*ldc+2]=c42; c_base[4*ldc+3]=c43;
    c_base[4*ldc+4]=c44; c_base[4*ldc+5]=c45; c_base[4*ldc+6]=c46; c_base[4*ldc+7]=c47;
    c_base[5*ldc]=c50; c_base[5*ldc+1]=c51; c_base[5*ldc+2]=c52; c_base[5*ldc+3]=c53;
    c_base[5*ldc+4]=c54; c_base[5*ldc+5]=c55; c_base[5*ldc+6]=c56; c_base[5*ldc+7]=c57;
}

void matrix_multiply(
    const int n, const int m, const int l, const int *__restrict__ a_mat,
    const int *__restrict__ b_mat, int *__restrict__ out_mat)
{
    int *__restrict__ local_out = g_local_out;

    // Tuned for Intel i5-10500: L1=32KB, L2=256KB, L3=12MB
    const int BLOCK_K = (m < 600) ? 512 : 768;
    const int BLOCK_J = (l < 600) ? 128 : 192;
    const int BLOCK_I = (g_local_rows < 600) ? 128 : 192;
    const int MICRO_I = 6;
    const int MICRO_J = 8;

    memset(local_out, 0, g_local_rows * l * sizeof(int));

    // Outer loop: K-J-I order for best cache reuse
    for (int kk = 0; kk < m; kk += BLOCK_K) {
        int k_end = std::min(kk + BLOCK_K, m);
        int k_len = k_end - kk;

        for (int jj = 0; jj < l; jj += BLOCK_J) {
            int j_end = std::min(jj + BLOCK_J, l);
            int j_len = j_end - jj;

            // Pack B block once for all i iterations
            pack_b_block(b_mat, g_packed_b, kk, k_len, jj, j_len, m);

            for (int ii = 0; ii < g_local_rows; ii += BLOCK_I) {
                int i_end = std::min(ii + BLOCK_I, g_local_rows);
                int i_len = i_end - ii;

                // Pack A block
                pack_a_block(a_mat, g_packed_a, ii, i_len, kk, k_len, m);

                // Process with 6x8 micro-kernels
                int i = 0;
                for (; i + MICRO_I - 1 < i_len; i += MICRO_I) {
                    int j = 0;
                    for (; j + MICRO_J - 1 < j_len; j += MICRO_J) {
                        matmul_kernel_6x8(
                            &g_packed_a[i * k_len],
                            &g_packed_b[j * k_len],
                            &local_out[(ii + i) * l + (jj + j)],
                            k_len, l);
                    }

                    // Handle remaining columns
                    for (; j < j_len; j++) {
                        const int *__restrict__ a_row = &g_packed_a[i * k_len];
                        const int *__restrict__ b_col = &g_packed_b[j * k_len];
                        int sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0;

                        for (int k = 0; k < k_len; k++) {
                            int b_val = b_col[k];
                            sum0 += a_row[k] * b_val;
                            sum1 += a_row[k_len + k] * b_val;
                            sum2 += a_row[2*k_len + k] * b_val;
                            sum3 += a_row[3*k_len + k] * b_val;
                            sum4 += a_row[4*k_len + k] * b_val;
                            sum5 += a_row[5*k_len + k] * b_val;
                        }
                        local_out[(ii+i)*l + (jj+j)] += sum0;
                        local_out[(ii+i+1)*l + (jj+j)] += sum1;
                        local_out[(ii+i+2)*l + (jj+j)] += sum2;
                        local_out[(ii+i+3)*l + (jj+j)] += sum3;
                        local_out[(ii+i+4)*l + (jj+j)] += sum4;
                        local_out[(ii+i+5)*l + (jj+j)] += sum5;
                    }
                }

                // Handle remaining rows
                for (; i < i_len; i++) {
                    const int *__restrict__ a_row = &g_packed_a[i * k_len];
                    for (int j = 0; j < j_len; j++) {
                        const int *__restrict__ b_col = &g_packed_b[j * k_len];
                        int sum = 0;
                        for (int k = 0; k < k_len; k++) {
                            sum += a_row[k] * b_col[k];
                        }
                        local_out[(ii+i)*l + (jj+j)] += sum;
                    }
                }
            }
        }
    }

    // Gather results
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
                out_mat, recvcounts, displs_out, MPI_INT, 0, MPI_COMM_WORLD);

    if (g_rank == 0) {
        delete[] recvcounts;
        delete[] displs_out;
    }
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    free(a_mat);
    free(b_mat);
    free(g_local_out);
    free(g_packed_a);
    free(g_packed_b);
    if (g_rank == 0) {
        delete[] g_sendcounts;
        delete[] g_displs;
    }
}
