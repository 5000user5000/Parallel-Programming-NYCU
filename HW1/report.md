# Q1

## Q1-1:

Does the vector utilization increase, decrease or stay the same as VECTOR_WIDTH changes? Why?

ANS:
As VECTOR_WIDTH increases from 2→4→8→16, vector utilization tends to slightly decrease.
The main reason is per-lane divergence in `clampedExp`: different exponents make some lanes finish early and sit idle while others keep multiplying. Wider vectors amplify this effect. Tail underfill at the last chunk also exists but is negligible for large N. Therefore, utilization typically drops modestly as the width grows.


You can see the data below:
all run by `run -- ./myexp -s 10000`
VECTOR_WIDTH = 2 -> 4 -> 8 -> 16
vector utilization in `clampedExp` = 86.0% -> 79.5% -> 76.2% -> 74.7%
p.s. All above > 60% utilization.
| VECTOR_WIDTH = 2,8 | VECTOR_WIDTH = 4,16 | 
| -------- | -------- |
| ![image](https://hackmd.io/_uploads/HkytRiCjge.png =400x200)   | ![image](https://hackmd.io/_uploads/BkNaasCiel.png =400x200)   |
|![image](https://hackmd.io/_uploads/ry6pCiRjxe.png =400x200) | ![image](https://hackmd.io/_uploads/rJc-y2Rsxx.png =400x200)|



## Bonus

I implemented `arraySumVector`, assuming `N` is a multiple of `VECTOR_WIDTH` and `VECTOR_WIDTH` is even. Using `hadd + interleave`, the reduction completes in `log₂(VECTOR_WIDTH)` steps, followed by a final extraction. Verification passed, and vector utilization is ~100%, satisfying both the >80% requirement and the complexity bound, as confirmed by the four figures in Q1-1.



# Q2

## Q2-1

Fix the code to make sure it uses aligned moves for the best performance. 
Hint: we want to see vmovaps rather than vmovups.


ANS:
You can compile this code below by
`make clean; make test1.o ASSEMBLE=1 VECTORIZE=1 RESTRICT=1 ALIGN=1 AVX2=1`

```c=
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
```

And the compiled result (test1.vec.restr.align.avx2.s) will be this
```asm=
	.text
	.file	"test1.c"
	.globl	test1                           # -- Begin function test1
	.p2align	4, 0x90
	.type	test1,@function
test1:                                  # @test1
	.cfi_startproc
# %bb.0:
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB0_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_2 Depth 2
	xorl	%ecx, %ecx
	.p2align	4, 0x90
.LBB0_2:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vmovaps	(%rdi,%rcx,4), %ymm0
	vmovaps	32(%rdi,%rcx,4), %ymm1
	vmovaps	64(%rdi,%rcx,4), %ymm2
	vmovaps	96(%rdi,%rcx,4), %ymm3
	vaddps	(%rsi,%rcx,4), %ymm0, %ymm0
	vaddps	32(%rsi,%rcx,4), %ymm1, %ymm1
	vaddps	64(%rsi,%rcx,4), %ymm2, %ymm2
	vaddps	96(%rsi,%rcx,4), %ymm3, %ymm3
	vmovaps	%ymm0, (%rdx,%rcx,4)
	vmovaps	%ymm1, 32(%rdx,%rcx,4)
	vmovaps	%ymm2, 64(%rdx,%rcx,4)
	vmovaps	%ymm3, 96(%rdx,%rcx,4)
	addq	$32, %rcx
	cmpq	$1024, %rcx                     # imm = 0x400
	jne	.LBB0_2
# %bb.3:                                #   in Loop: Header=BB0_1 Depth=1
	incl	%eax
	cmpl	$20000000, %eax                 # imm = 0x1312D00
	jne	.LBB0_1
# %bb.4:
	vzeroupper
	retq
.Lfunc_end0:
	.size	test1, .Lfunc_end0-test1
	.cfi_endproc
                                        # -- End function
	.ident	"Ubuntu clang version 18.1.3 (1ubuntu1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig

```


## Q2-2

What speedup does the vectorized code achieve over the unvectorized code? 
What additional speedup does using -mavx2 give (AVX2=1 in the Makefile)? 
You may wish to run this experiment several times and take median elapsed times; you can report answers to the nearest 100% (e.g., 2×, 3×, etc). 
What can you infer about the bit width of the default vector registers on the PP machines? 
What about the bit width of the AVX2 vector registers?

Hint: Aside from speedup and the vectorization report, the most relevant information is that the data type of each array is float.


ANS:

### Experimental Results

On the PP machines, I ran three configurations and took the median elapsed times:

|  Case  | Configuration                  | Time (s)  | Relative Speedup                               |
|:------:| ------------------------------ | --------- | ---------------------------------------------- |
| Case 1 | baseline (no vectorization)    | **6.945** | 1×                                             |
| Case 2 | vectorization (SSE, 128-bit)   | **1.735** | \~**4×** (6.945 / 1.735)                       |
| Case 3 | vectorization + AVX2 (256-bit) | **0.874** | \~**8×** (6.945 / 0.874), \~**2×** over Case 2 |


### Speedup Analysis

1. **Case 2 vs. Case 1**: Vectorization provides a \~4× speedup compared to the scalar baseline.
2. **Case 3 vs. Case 2**: Enabling AVX2 yields an additional \~2× speedup over SSE vectorization.
3. **Case 3 vs. Case 1**: Overall, the AVX2 vectorized version is about 8× faster than the baseline.


### Conclusion and Inference

* From the \~4× speedup in Case 2, we infer that the **default vector registers** are **128-bit SSE registers**, processing 4 floats (4×32 bits) at a time.
* From the additional 2× gain in Case 3, we infer that the **AVX2 vector registers** are **256-bit registers**, processing 8 floats (8×32 bits) at a time.
* The measured performance aligns well with the theoretical throughput of these vector widths.

## Q2-3

Provide a theory for why the compiler is generating dramatically different assemblies.

ANS:
The original version first stores a value and then conditionally overwrites it. 
This looks like two separate stores rather than a simple max, introducing a **store dependency**, so the compiler conservatively **avoids vectorization**.

The rewritten version expresses it as a **conditional assignment**, which the compiler can recognize as a **max pattern**. 
It gets lowered to the SIMD `maxps` operation, so the compiler safely vectorizes it using `movaps` (aligned load) and `maxps` (packed float max).