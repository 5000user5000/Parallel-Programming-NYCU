#include <mpi.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// Compiler optimization hints (No explicit SIMD, only scalar optimizations)
#pragma GCC optimize("O3", "unroll-loops", "omit-frame-pointer", "inline", "fast-math")

static int g_n, g_m, g_l;
static int g_rank, g_size;
static int g_local_rows;
static int *g_sendcounts = nullptr;
static int *g_displs = nullptr;
static int *g_local_out = nullptr;

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

    *a_mat_ptr = new int[g_local_rows * m];
    *b_mat_ptr = new int[m * l];  // Will store B transposed
    g_local_out = new int[g_local_rows * l];

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

    // Transpose B and broadcast with non-blocking communication
    // Original B[j][k] -> Transposed B_T[k][j] for contiguous column access
    MPI_Request req;
    if (g_rank == 0) {
        // Transpose: B_T[k * l + j] = B[j * m + k]
        for (int j = 0; j < l; j++) {
            for (int k = 0; k < m; k++) {
                (*b_mat_ptr)[k * l + j] = b_mat[j * m + k];
            }
        }
        MPI_Ibcast(*b_mat_ptr, m * l, MPI_INT, 0, MPI_COMM_WORLD, &req);
    } else {
        MPI_Ibcast(*b_mat_ptr, m * l, MPI_INT, 0, MPI_COMM_WORLD, &req);
    }

    // Wait for broadcast to complete
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

// Optimized 4x4 micro-kernel for transposed B
// B_T is stored as B_T[k][j], so column j is contiguous: B_T[kk*l + j, kk*l + j + 1, ...]
inline void matmul_kernel_4x4_transposed(
    const int *__restrict__ a_base, const int *__restrict__ b_T,
    int *__restrict__ c_base, int m, int l, int ldc, int k_len)
{
    int c00 = c_base[0], c01 = c_base[1], c02 = c_base[2], c03 = c_base[3];
    int c10 = c_base[ldc], c11 = c_base[ldc+1], c12 = c_base[ldc+2], c13 = c_base[ldc+3];
    int c20 = c_base[2*ldc], c21 = c_base[2*ldc+1], c22 = c_base[2*ldc+2], c23 = c_base[2*ldc+3];
    int c30 = c_base[3*ldc], c31 = c_base[3*ldc+1], c32 = c_base[3*ldc+2], c33 = c_base[3*ldc+3];

    // 8-way unrolling with prefetching
    int k = 0;
    for (; k + 7 < k_len; k += 8) {
        __builtin_prefetch(&a_base[k + 64], 0, 3);
        __builtin_prefetch(&b_T[k * l + 64], 0, 3);

        // Load A values (4 rows x 8 k-values)
        int a0_0 = a_base[k], a0_1 = a_base[k+1], a0_2 = a_base[k+2], a0_3 = a_base[k+3];
        int a0_4 = a_base[k+4], a0_5 = a_base[k+5], a0_6 = a_base[k+6], a0_7 = a_base[k+7];
        int a1_0 = a_base[m+k], a1_1 = a_base[m+k+1], a1_2 = a_base[m+k+2], a1_3 = a_base[m+k+3];
        int a1_4 = a_base[m+k+4], a1_5 = a_base[m+k+5], a1_6 = a_base[m+k+6], a1_7 = a_base[m+k+7];
        int a2_0 = a_base[2*m+k], a2_1 = a_base[2*m+k+1], a2_2 = a_base[2*m+k+2], a2_3 = a_base[2*m+k+3];
        int a2_4 = a_base[2*m+k+4], a2_5 = a_base[2*m+k+5], a2_6 = a_base[2*m+k+6], a2_7 = a_base[2*m+k+7];
        int a3_0 = a_base[3*m+k], a3_1 = a_base[3*m+k+1], a3_2 = a_base[3*m+k+2], a3_3 = a_base[3*m+k+3];
        int a3_4 = a_base[3*m+k+4], a3_5 = a_base[3*m+k+5], a3_6 = a_base[3*m+k+6], a3_7 = a_base[3*m+k+7];

        // Load B_T values (4 columns x 8 k-values) - NOW CONTIGUOUS!
        // b_T points to B_T[k][j], stride between k is l
        int b0_0 = b_T[0], b0_1 = b_T[l], b0_2 = b_T[2*l], b0_3 = b_T[3*l];
        int b0_4 = b_T[4*l], b0_5 = b_T[5*l], b0_6 = b_T[6*l], b0_7 = b_T[7*l];
        int b1_0 = b_T[1], b1_1 = b_T[l+1], b1_2 = b_T[2*l+1], b1_3 = b_T[3*l+1];
        int b1_4 = b_T[4*l+1], b1_5 = b_T[5*l+1], b1_6 = b_T[6*l+1], b1_7 = b_T[7*l+1];
        int b2_0 = b_T[2], b2_1 = b_T[l+2], b2_2 = b_T[2*l+2], b2_3 = b_T[3*l+2];
        int b2_4 = b_T[4*l+2], b2_5 = b_T[5*l+2], b2_6 = b_T[6*l+2], b2_7 = b_T[7*l+2];
        int b3_0 = b_T[3], b3_1 = b_T[l+3], b3_2 = b_T[2*l+3], b3_3 = b_T[3*l+3];
        int b3_4 = b_T[4*l+3], b3_5 = b_T[5*l+3], b3_6 = b_T[6*l+3], b3_7 = b_T[7*l+3];

        // Compute 4x4 block
        c00 += a0_0*b0_0 + a0_1*b0_1 + a0_2*b0_2 + a0_3*b0_3 + a0_4*b0_4 + a0_5*b0_5 + a0_6*b0_6 + a0_7*b0_7;
        c01 += a0_0*b1_0 + a0_1*b1_1 + a0_2*b1_2 + a0_3*b1_3 + a0_4*b1_4 + a0_5*b1_5 + a0_6*b1_6 + a0_7*b1_7;
        c02 += a0_0*b2_0 + a0_1*b2_1 + a0_2*b2_2 + a0_3*b2_3 + a0_4*b2_4 + a0_5*b2_5 + a0_6*b2_6 + a0_7*b2_7;
        c03 += a0_0*b3_0 + a0_1*b3_1 + a0_2*b3_2 + a0_3*b3_3 + a0_4*b3_4 + a0_5*b3_5 + a0_6*b3_6 + a0_7*b3_7;

        c10 += a1_0*b0_0 + a1_1*b0_1 + a1_2*b0_2 + a1_3*b0_3 + a1_4*b0_4 + a1_5*b0_5 + a1_6*b0_6 + a1_7*b0_7;
        c11 += a1_0*b1_0 + a1_1*b1_1 + a1_2*b1_2 + a1_3*b1_3 + a1_4*b1_4 + a1_5*b1_5 + a1_6*b1_6 + a1_7*b1_7;
        c12 += a1_0*b2_0 + a1_1*b2_1 + a1_2*b2_2 + a1_3*b2_3 + a1_4*b2_4 + a1_5*b2_5 + a1_6*b2_6 + a1_7*b2_7;
        c13 += a1_0*b3_0 + a1_1*b3_1 + a1_2*b3_2 + a1_3*b3_3 + a1_4*b3_4 + a1_5*b3_5 + a1_6*b3_6 + a1_7*b3_7;

        c20 += a2_0*b0_0 + a2_1*b0_1 + a2_2*b0_2 + a2_3*b0_3 + a2_4*b0_4 + a2_5*b0_5 + a2_6*b0_6 + a2_7*b0_7;
        c21 += a2_0*b1_0 + a2_1*b1_1 + a2_2*b1_2 + a2_3*b1_3 + a2_4*b1_4 + a2_5*b1_5 + a2_6*b1_6 + a2_7*b1_7;
        c22 += a2_0*b2_0 + a2_1*b2_1 + a2_2*b2_2 + a2_3*b2_3 + a2_4*b2_4 + a2_5*b2_5 + a2_6*b2_6 + a2_7*b2_7;
        c23 += a2_0*b3_0 + a2_1*b3_1 + a2_2*b3_2 + a2_3*b3_3 + a2_4*b3_4 + a2_5*b3_5 + a2_6*b3_6 + a2_7*b3_7;

        c30 += a3_0*b0_0 + a3_1*b0_1 + a3_2*b0_2 + a3_3*b0_3 + a3_4*b0_4 + a3_5*b0_5 + a3_6*b0_6 + a3_7*b0_7;
        c31 += a3_0*b1_0 + a3_1*b1_1 + a3_2*b1_2 + a3_3*b1_3 + a3_4*b1_4 + a3_5*b1_5 + a3_6*b1_6 + a3_7*b1_7;
        c32 += a3_0*b2_0 + a3_1*b2_1 + a3_2*b2_2 + a3_3*b2_3 + a3_4*b2_4 + a3_5*b2_5 + a3_6*b2_6 + a3_7*b2_7;
        c33 += a3_0*b3_0 + a3_1*b3_1 + a3_2*b3_2 + a3_3*b3_3 + a3_4*b3_4 + a3_5*b3_5 + a3_6*b3_6 + a3_7*b3_7;

        b_T += 8 * l;  // Move to next 8 k-values
    }

    // Handle remaining k
    for (; k < k_len; k++) {
        int a0 = a_base[k], a1 = a_base[m+k], a2 = a_base[2*m+k], a3 = a_base[3*m+k];
        int b0 = b_T[0], b1 = b_T[1], b2 = b_T[2], b3 = b_T[3];
        c00 += a0*b0; c01 += a0*b1; c02 += a0*b2; c03 += a0*b3;
        c10 += a1*b0; c11 += a1*b1; c12 += a1*b2; c13 += a1*b3;
        c20 += a2*b0; c21 += a2*b1; c22 += a2*b2; c23 += a2*b3;
        c30 += a3*b0; c31 += a3*b1; c32 += a3*b2; c33 += a3*b3;
        b_T += l;
    }

    // Write back
    c_base[0] = c00; c_base[1] = c01; c_base[2] = c02; c_base[3] = c03;
    c_base[ldc] = c10; c_base[ldc+1] = c11; c_base[ldc+2] = c12; c_base[ldc+3] = c13;
    c_base[2*ldc] = c20; c_base[2*ldc+1] = c21; c_base[2*ldc+2] = c22; c_base[2*ldc+3] = c23;
    c_base[3*ldc] = c30; c_base[3*ldc+1] = c31; c_base[3*ldc+2] = c32; c_base[3*ldc+3] = c33;
}

void matrix_multiply(
    const int n, const int m, const int l, const int *__restrict__ a_mat,
    const int *__restrict__ b_mat, int *__restrict__ out_mat)
{
    int *__restrict__ local_out = g_local_out;

    // Optimized for Intel i5-10500: L1=32KB, L2=256KB, L3=12MB
    // Cache-aware block sizes
    const int BLOCK_K = (m < 600) ? 384 : 512;  // Maximize k-reuse
    const int BLOCK_J = (l < 600) ? 96 : 128;   // Fit in L2
    const int BLOCK_I = (g_local_rows < 600) ? 96 : 128;
    const int MICRO_I = 4;
    const int MICRO_J = 4;

    memset(local_out, 0, g_local_rows * l * sizeof(int));

    // Three-level cache blocking: K-J-I order for best reuse
    for (int kk = 0; kk < m; kk += BLOCK_K) {
        int k_end = std::min(kk + BLOCK_K, m);
        int k_len = k_end - kk;

        for (int jj = 0; jj < l; jj += BLOCK_J) {
            int j_end = std::min(jj + BLOCK_J, l);

            for (int ii = 0; ii < g_local_rows; ii += BLOCK_I) {
                int i_end = std::min(ii + BLOCK_I, g_local_rows);

                // Process with 4x4 micro-kernels
                int i = ii;
                for (; i + MICRO_I - 1 < i_end; i += MICRO_I) {
                    int j = jj;
                    for (; j + MICRO_J - 1 < j_end; j += MICRO_J) {
                        // B_T is stored as B_T[k][j], access B_T[kk][jj] at b_mat + kk*l + jj
                        matmul_kernel_4x4_transposed(
                            &a_mat[i * m + kk],     // A[i][kk]
                            &b_mat[kk * l + j],     // B_T[kk][j]
                            &local_out[i * l + j],  // C[i][j]
                            m, l, l, k_len);
                    }

                    // Handle remaining columns
                    for (; j < j_end; j++) {
                        const int *__restrict__ a_row = &a_mat[i * m + kk];
                        const int *__restrict__ b_col = &b_mat[kk * l + j];  // B_T[kk][j]
                        int sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;

                        int k = 0;
                        for (; k + 7 < k_len; k += 8) {
                            __builtin_prefetch(&b_col[(k + 64) * l], 0, 3);
                            // B_T column j at different k: b_col[k*l]
                            int b0 = b_col[0], b1 = b_col[l], b2 = b_col[2*l], b3 = b_col[3*l];
                            int b4 = b_col[4*l], b5 = b_col[5*l], b6 = b_col[6*l], b7 = b_col[7*l];

                            sum0 += a_row[k]*b0 + a_row[k+1]*b1 + a_row[k+2]*b2 + a_row[k+3]*b3 +
                                   a_row[k+4]*b4 + a_row[k+5]*b5 + a_row[k+6]*b6 + a_row[k+7]*b7;
                            sum1 += a_row[m+k]*b0 + a_row[m+k+1]*b1 + a_row[m+k+2]*b2 + a_row[m+k+3]*b3 +
                                   a_row[m+k+4]*b4 + a_row[m+k+5]*b5 + a_row[m+k+6]*b6 + a_row[m+k+7]*b7;
                            sum2 += a_row[2*m+k]*b0 + a_row[2*m+k+1]*b1 + a_row[2*m+k+2]*b2 + a_row[2*m+k+3]*b3 +
                                   a_row[2*m+k+4]*b4 + a_row[2*m+k+5]*b5 + a_row[2*m+k+6]*b6 + a_row[2*m+k+7]*b7;
                            sum3 += a_row[3*m+k]*b0 + a_row[3*m+k+1]*b1 + a_row[3*m+k+2]*b2 + a_row[3*m+k+3]*b3 +
                                   a_row[3*m+k+4]*b4 + a_row[3*m+k+5]*b5 + a_row[3*m+k+6]*b6 + a_row[3*m+k+7]*b7;
                            b_col += 8 * l;
                        }
                        for (; k < k_len; k++) {
                            int b_val = b_col[0];
                            sum0 += a_row[k] * b_val;
                            sum1 += a_row[m+k] * b_val;
                            sum2 += a_row[2*m+k] * b_val;
                            sum3 += a_row[3*m+k] * b_val;
                            b_col += l;
                        }
                        local_out[i*l + j] += sum0;
                        local_out[(i+1)*l + j] += sum1;
                        local_out[(i+2)*l + j] += sum2;
                        local_out[(i+3)*l + j] += sum3;
                    }
                }

                // Handle remaining rows
                for (; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        const int *__restrict__ a_row = &a_mat[i * m + kk];
                        const int *__restrict__ b_col = &b_mat[kk * l + j];
                        int sum = 0;

                        int k = 0;
                        for (; k + 7 < k_len; k += 8) {
                            sum += a_row[k]*b_col[0] + a_row[k+1]*b_col[l] +
                                  a_row[k+2]*b_col[2*l] + a_row[k+3]*b_col[3*l] +
                                  a_row[k+4]*b_col[4*l] + a_row[k+5]*b_col[5*l] +
                                  a_row[k+6]*b_col[6*l] + a_row[k+7]*b_col[7*l];
                            b_col += 8 * l;
                        }
                        for (; k < k_len; k++) {
                            sum += a_row[k] * b_col[0];
                            b_col += l;
                        }
                        local_out[i * l + j] += sum;
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
    delete[] a_mat;
    delete[] b_mat;
    delete[] g_local_out;
    if (g_rank == 0) {
        delete[] g_sendcounts;
        delete[] g_displs;
    }
}
