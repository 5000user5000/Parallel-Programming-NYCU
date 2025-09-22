#include "test.h"
#include <stdint.h>

void test1(float *__restrict a, float *__restrict b, float *__restrict c, int N) {
  __builtin_assume(N == 1024);          // assume N%8 ==0 （AVX2: 8 floats/vec）
  __builtin_assume((uintptr_t)a % 32 == 0);
  __builtin_assume((uintptr_t)b % 32 == 0);
  __builtin_assume((uintptr_t)c % 32 == 0);

  a = (float *)__builtin_assume_aligned(a, 32);  // 32-byte for AVX2
  b = (float *)__builtin_assume_aligned(b, 32);
  c = (float *)__builtin_assume_aligned(c, 32);

  for (int i = 0; i < I; i++) {
    for (int j = 0; j < N; j++) {
      c[j] = a[j] + b[j];
    }
  }
}
