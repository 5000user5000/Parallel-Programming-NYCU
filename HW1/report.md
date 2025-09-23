# Q1

## Q1-1:

Does the vector utilization increase, decrease or stay the same as VECTOR_WIDTH changes? Why?

**ANS:**
As `VECTOR_WIDTH` increases from 2, 4, 8 to 16, vector utilization generally **decreases**.

Before explaining the reason, let’s first observe how `addLog` in `Logger.cpp` computes vector utilization: for each vector instruction, it increments `utilized_lane` for every `i ∈ [0, N)` where `mask.value[i] == 1`, and increments `total_lane` by `N` (usually equal to `VECTOR_WIDTH`). Therefore,

$$
\text{Utilization} = \frac{\text{utilized\_lane}}{\text{total\_lane}}.
$$

In other words, any masked-off lane reduces the numerator while the denominator remains fixed.

Now for the cause: in `clampedExp`, different elements may have different exponents `y`. Some lanes finish quickly and get masked off, while others continue multiplying, creating **per-lane divergence**. With wider vectors, more lanes are likely to sit idle within the same iteration, so utilization drops. There is also a **tail underfill** effect in the last chunk if the data size is not a multiple of `VECTOR_WIDTH`, but this effect is negligible for large `N`.

You can see the data below:
all run by `run -- ./myexp -s 10000`
VECTOR_WIDTH = 2 -> 4 -> 8 -> 16
vector utilization in `clampedExp` = 77.9% -> 70.7% -> 67.0% -> 65.2%
p.s. All above > 60% utilization.
| VECTOR_WIDTH = 2,8 | VECTOR_WIDTH = 4,16 | 
| -------- | -------- |
| ![image](https://hackmd.io/_uploads/HkgR4s1nge.png =400x200)   | ![image](https://hackmd.io/_uploads/HkL9Voy3ex.png =400x200)   |
|![image](https://hackmd.io/_uploads/SymXroknxx.png =400x200) | ![image](https://hackmd.io/_uploads/SyogBiJ2lg.png =400x200)|



## Bonus

I implemented `arraySumVector`, assuming `N` is a multiple of `VECTOR_WIDTH` and `VECTOR_WIDTH` is even. Using `hadd + interleave`, the reduction completes in `log₂(VECTOR_WIDTH)` steps, followed by a final extraction. Verification passed, and vector utilization is 100%, satisfying both the >80% requirement and the complexity bound, as confirmed by the four figures in Q1-1.



# Q2

## Q2-1

Fix the code to make sure it uses aligned moves for the best performance. 
Hint: we want to see vmovaps rather than vmovups.


**ANS:**
You can modify `test1.c` like below:

```c=
#include "test.h"
#include <stdint.h>

void test1(float *__restrict a, float *__restrict b, float *__restrict c, int N) {
  __builtin_assume(N == 1024);

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

With this change the compiler now emits `vmovaps` instead of `vmovups` (as shown in the assembly below). The reason is that we explicitly told the compiler the arrays are **32-byte aligned** (the alignment requirement for AVX2 YMM registers). When alignment is guaranteed, the optimizer can safely generate the faster aligned  instructions; without that guarantee, it will conservatively use `vmovups` for safety.


```asm=
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

On the PP machines, I ran three configurations on fixed `test1.c` and took the median elapsed times:

|  Case  | Configuration                  | Time (s)  | Relative Speedup                               |
|:------:| ------------------------------ | --------- | ---------------------------------------------- |
| Case 1 | baseline (no vectorization)    | **6.945** | 1×                                             |
| Case 2 | vectorization    | **1.735** | \~**4×** (6.945 / 1.735)                       |
| Case 3 | vectorization + AVX2  | **0.874** | \~**8×** (6.945 / 0.874), \~**2×** over Case 2 |


### Speedup Analysis

1. **Case 2 vs. Case 1**: Vectorization provides a \~4× speedup compared to the scalar baseline.
2. **Case 3 vs. Case 2**: Enabling AVX2 yields an additional \~2× speedup over SSE vectorization.
3. **Case 3 vs. Case 1**: Overall, the AVX2 vectorized version is about 8× faster than the baseline.


### Conclusion and Inference

* From the \~4× speedup in Case 2, I infer that the **default vector registers** are **128-bit SSE registers**, processing 4 floats (4×32 bits) at a time.
* From the additional 2× gain in Case 3, I infer that the **AVX2 vector registers** are **256-bit registers**, processing 8 floats (8×32 bits) at a time.
* The measured performance aligns well with the theoretical throughput of these vector widths.

## Q2-3

Provide a theory for why the compiler is generating dramatically different assemblies.

ANS:
The original version first stores `a[j]` into `c[j]` and may overwrite it later. This introduces an **extra store side effect**, so the compiler cannot safely fold it into a single max operation and thus avoids vectorization.

The rewritten version expresses the logic as a **conditional assignment**, which matches the compiler’s **max idiom**. The optimizer recognizes it can be lowered to a branchless SIMD operation using `maxps`, so the loop is vectorized with `movaps` (aligned load) + `maxps` (packed float max).


Original version of `test2.c` :
```asm=
.LBB0_7:                                #   in Loop: Header=BB0_1 Depth=1
        addl    $1, %r8d
        cmpl    $20000000, %r8d                 # imm = 0x1312D00
        je      .LBB0_8
.LBB0_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_2 Depth 2
        xorl    %ecx, %ecx
        jmp     .LBB0_2
        .p2align        4, 0x90
.LBB0_6:                                #   in Loop: Header=BB0_2 Depth=2
        addq    $2, %rcx
        cmpq    $1024, %rcx                     # imm = 0x400
        je      .LBB0_7
.LBB0_2:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
        movl    (%rdi,%rcx,4), %eax
        movl    %eax, (%rdx,%rcx,4)
        movss   (%rsi,%rcx,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
        movd    %eax, %xmm1
        ucomiss %xmm1, %xmm0
        jbe     .LBB0_4
# %bb.3:                                #   in Loop: Header=BB0_2 Depth=2
        movss   %xmm0, (%rdx,%rcx,4)
.LBB0_4:                                #   in Loop: Header=BB0_2 Depth=2
        movl    4(%rdi,%rcx,4), %eax
        movl    %eax, 4(%rdx,%rcx,4)
        movss   4(%rsi,%rcx,4), %xmm0           # xmm0 = mem[0],zero,zero,zero
        movd    %eax, %xmm1
        ucomiss %xmm1, %xmm0
        jbe     .LBB0_6
# %bb.5:                                #   in Loop: Header=BB0_2 Depth=2
        movss   %xmm0, 4(%rdx,%rcx,4)
        jmp     .LBB0_6
.LBB0_8:
        retq
```


After patching:
```asm=
.LBB0_2:                                #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
        movaps  (%rsi,%rcx,4), %xmm0
        movaps  16(%rsi,%rcx,4), %xmm1
        maxps   (%rdi,%rcx,4), %xmm0
        maxps   16(%rdi,%rcx,4), %xmm1
        movups  %xmm0, (%rdx,%rcx,4)
        movups  %xmm1, 16(%rdx,%rcx,4)
        movaps  32(%rsi,%rcx,4), %xmm0
        movaps  48(%rsi,%rcx,4), %xmm1
        maxps   32(%rdi,%rcx,4), %xmm0
        maxps   48(%rdi,%rcx,4), %xmm1
        movups  %xmm0, 32(%rdx,%rcx,4)
        movups  %xmm1, 48(%rdx,%rcx,4)
        addq    $16, %rcx
        cmpq    $1024, %rcx                     # imm = 0x400
        jne     .LBB0_2
# %bb.3:                                #   in Loop: Header=BB0_1 Depth=1
        addl    $1, %eax
        cmpl    $20000000, %eax                 # imm = 0x1312D00
        jne     .LBB0_1
# %bb.4:
        retq
```