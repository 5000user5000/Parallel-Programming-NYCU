# Programming Assignment I: SIMD Programming

**Parallel Programming by Prof. Yi-Ping You**
Due date: **23:59, October 2 Thursday, 2025**

The purpose of this assignment is to familiarize yourself with (single-instruction, multiple-data) SIMD programming. Most modern processors include vector operations (i.e., SIMD instructions) that you can take advantage of to improve performance through vectorization. Although modern compilers support automatic vectorization optimizations, their capabilities to fully auto-vectorize a given piece of code are limited. Fortunately, most compilers (targeted to processors with SIMD extensions) provide SIMD intrinsics to allow programmers to manually vectorize their code.

---

## Table of contents

1. Programming Assignment I: SIMD Programming

   1. 0\. Using Workstation
   2. 1. Part 1: Vectorizing Code Using Fake SIMD Intrinsics
   3. 2. Part 2: Vectorizing Code with Automatic Vectorization Optimizations

      * 2.1 No auto-vectorization
      * 2.2 Turning on auto-vectorization
      * 2.3 Adding the `__restrict` qualifier
      * 2.4 Adding the `__builtin_assume_aligned` intrinsic
      * 2.5 Turning on AVX2 instructions
      * 2.6 Performance impacts of vectorization
      * 2.7 More examples

        * 2.7.1 Example 2
        * 2.7.2 Example 3
   4. 3. Requirements

      * 3.1 Part 1
      * 3.2 Part 2
   5. 4. Grading Policy
   6. 5. Evaluation Platform
   7. 6. Submission
   8. 7. References

---

We use the workstations provided by NYCU CSIT for this course. You can access the workstations by `ssh` or use your own environment to complete the assignment. (To learn how to use `ssh` and `scp`, you can refer to the video listed in the [references](#7-references))

| Hostname                        | IP                    |
| ------------------------------- | --------------------- |
| hpclogin\[01-03].cs.nycu.edu.tw | 140.113.17.\[101-103] |

Login example:

```shell
ssh <username>@hpclogin[01-03].cs.nycu.edu.tw
```

> **Important**
> Additionally, we have configured the environment for you, which **MUST** be activated before running the assignment code using the following command:
>
> ```shell
> module load pp
> ```
>
> You can leave the environment by typing `module unload pp`.

Please download the assignment code and unzip it:

```shell
wget https://nycu-sslab.github.io/PP-f25/assignments/HW1/HW1.zip
unzip HW1.zip
cd HW1
```

---

## 0. Using Workstation

Our workstations are managed using the Slurm Workload Manager and NFS. After connecting via `ssh`, you’ll be placed on the *login node*. From there, you can submit jobs to the *compute nodes* using the `run` command. Your home directory is shared across all login nodes and compute nodes. Below is an example of how to submit a job to the cluster:

```shell
run -- ./myexp -s 10000
```

> **Important**
>
> * Avoid running your programs directly on the login node… all program execution should occur on the compute nodes.
> * The double dash (`--`) is required because the run command itself accepts arguments.

Run multiple programs in a row:

```shell
run -- bash -c "./myexp -s 10000 && ./myexp -s 10000"
```

If your program gets stuck:

```shell
scancel <your_job_id>
```

> **Note**
>
> * `run` wraps `sbatch`. You may use `sbatch` directly, but it’s outside the scope of this course.
> * `run` is provided by the TA; contact them for issues/suggestions.

See also a Slurm user guide: [https://man.twcc.ai/@twnia3/SJDW5J3Rv](https://man.twcc.ai/@twnia3/SJDW5J3Rv)

---

## 1. Part 1: Vectorizing Code Using Fake SIMD Intrinsics

Open `part1/main.cpp`. The `clampedExpSerial` function raises `values[i]` to the power `exponents[i]` and clamps the result at 9.999999. Implement the SIMD version in `clampedExpVector` inside `part1/vectorOP.cpp`.

Compile:

```shell
cd part1 && make
```

Rather than real SSE/AVX2 intrinsics, **use PP’s “fake vector intrinsics” in `PPintrin.h`**. An example `absVector` is provided in `part1/vectorOP.cpp`. (It’s intentionally not perfect.)

**Hints**

* Read `PPintrin.h` to know available ops.
* Every vector instruction is subject to a mask.
* Use multiple mask registers and operations.
* `_pp_init_ones`, `_pp_cntbits` may help.
* Handle tails when `N` isn’t a multiple of `VECTOR_WIDTH` (`run -- ./myexp -s 3`).
* `run -- ./myexp -l` prints vector-instruction logs; use `addUserLog()` and `PPLogger.printLog()` to debug.
* Change width via `make VECTOR_WIDTH=<number>` (default 4).

If there are incorrect results, the program will print the first one it finds and print out a table of function inputs and outputs. Your function’s output is after “output = “, which should match with the results after “gold = “. The program also prints out the utilization statistics of the PP fake vector units. You should consider the performance of your implementation related to “Total Vector Instructions”. (You can assume every PP fake vector instruction takes one cycle on the PP fake SIMD CPU.) “Vector Utilization” shows the percentage of vector lanes that are enabled.

The program reports mismatches and vector-utilization stats. See [requirements](#3-requirements).

> *Optional practice:* After part 1, try real SIMD intrinsics.

---

## 2. Part 2: Vectorizing Code with Automatic Vectorization Optimizations

Answer **Q2-1**, **Q2-2**, **Q2-3**.

We are going to start from scratch and make the compiler do the brunt of the work. You will notice that this is not a “flip a switch and everything is good” exercise, but it also requires effort from the developer to write the code in a way that the compiler can optimize. The goal of this assignment is to learn how to fully exploit the optimization capabilities of the compiler such that in the future when you write code, you write it in a way that gets you the best performance for the least amount of effort.

Enter folder:

```shell
cd part2
```

### 2.1 No auto-vectorization

`test1.c` (outer loop over `I` to magnify timing):

```c
void test1(float *a, float *b, float *c, int N) {
  __builtin_assume(N == 1024);

  for (int i=0; i<I; i++) {
    for (int j=0; j<N; j++) {
      c[j] = a[j] + b[j];
    }
  }
}
```

Compile and dump asm with vectorization **disabled**:

```shell
make clean; make test1.o ASSEMBLE=1
```

You are recommended to try out Compiler Explorer, a nifty online tool that provides an “interactive compiler”.
A [Compiler Explorer link](https://godbolt.org/z/4roc539fx) is pre-configured for 11.0.1 version of clang and compiler flags from the makefile. To manually configure yourself: select language C, compiler version x86-64 clang 11.0.1 and enter flags `-O3 -std=c11 -Wall -fno-vectorize -fverbose-asm`. A screenshot is shown below.

### 2.2 Turning on auto-vectorization

Let’s turn on the compiler optimizations and see how much the compiler can speed up the program.

Remove `-fno-vectorize` from the compiler option to turn on the compiler optimizations, and add `-Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize` to get more information from clang about why it does or does not optimize code. This was done in the makefile, and you can enable auto-vectorization by typing the following command, which generates `assembly/test1.vec.s`.

```shell
make clean; make test1.o ASSEMBLE=1 VECTORIZE=1
```

You should see the following output, informing you that the loop has been vectorized. Although clang does tell you this, you should always look at the assembly to see exactly how it has been vectorized, since it is not guaranteed to be using the vector registers optimally.

```
test1.c:7:5: remark: vectorized loop (vectorization width: 4, interleaved count: 2) [-Rpass=loop-vectorize]
    for (int j=0; j<N; j++) {
    ^

```

You can observe the difference between `test1.vec.s` and `test1.novec.s` with the following command or by changing the compiler flag on Compiler Explorer.
Compare:

```shell
diff assembly/test1.vec.s assembly/test1.novec.s
```

### 2.3 Adding the `__restrict` qualifier

Now, if you inspect the assembly code—actually, you don’t need to do that, which is out of the scope of this assignment—you will see the code first checks if there is a partial overlap between arrays `a` and `c` or arrays `b` and `c`. If there is an overlap, then it does a simple non-vectorized code. If there is no overlap, it does a vectorized version. The above can, at best, be called partially vectorized.

The problem is that the compiler is constrained by what we tell it about the arrays. If we tell it more, then perhaps it can do more optimization. The most obvious thing is to inform the compiler that no overlap is possible. This is done in standard C by using the `restrict` qualifier for the pointers. By adding this type qualifier, you can hint to the compiler that for the lifetime of the pointer, only the pointer itself or a value directly derived from it (such as `pointer + 1`) will be used to access the object it points to.

C++ does not have standard support for `restrict`, but many compilers have equivalents that work in both C++ and C, such as the GCC’s and clang’s `__restrict__` (or `__restrict)`, and Visual C++’s `__declspec(restrict)`.

The code after adding the `__restrict` qualifier is shown as follows.

```c
void test1(float *__restrict a, float *__restrict b, float *__restrict c, int N) {
  __builtin_assume(N == 1024);

  for (int i=0; i<I; i++) {
    for (int j=0; j<N; j++) {
      c[j] = a[j] + b[j];
    }
  }
}
```

Let’s modify `test1.c` accordingly and recompile it again with the following command, which generates `assembly/test1.vec.restr.s`.

```shell
make clean; make test1.o ASSEMBLE=1 VECTORIZE=1 RESTRICT=1
```

Now you should see the generated code is better—the code for checking possible overlap is gone—but it is assuming the data are **NOT** 16 bytes aligned (`movups` is unaligned move). It also means that the loop above can not assume that the arrays are aligned.

If clang were smart, it could test for the cases where the arrays are either all aligned, or all unaligned, and have a fast inner loop. However, it is unable to do that currently.🙁

### 2.4 Adding the `__builtin_assume_aligned` intrinsic

To get the performance we are looking for, we need to tell clang that the arrays are aligned. There are a couple of ways to do that. The first is to construct a (non-portable) aligned type and use that in the function interface. The second is to add an alignment hint within the function. The second one is easier to implement on older code bases, as other functions calling the one to be vectorized do not have to be modified. The intrinsic for this is called `__builtin_assume_aligned`:

```c
void test1(float *__restrict a, float *__restrict b, float *__restrict c, int N) {
  __builtin_assume(N == 1024);
  a = (float *)__builtin_assume_aligned(a, 16);
  b = (float *)__builtin_assume_aligned(b, 16);
  c = (float *)__builtin_assume_aligned(c, 16);

  for (int i=0; i<I; i++) {
    for (int j=0; j<N; j++) {
      c[j] = a[j] + b[j];
    }
  }
}

```

Let’s modify `test1.c` accordingly and recompile it again with the following command, which generates `assembly/test1.vec.restr.align.s`.

```shell
make clean; make test1.o ASSEMBLE=1 VECTORIZE=1 RESTRICT=1 ALIGN=1
```

Let’s see the difference:

```shell
diff assembly/test1.vec.restr.s assembly/test1.vec.restr.align.s
```

Now finally, we get the nice tight vectorized code (`movaps` is aligned move.) we were looking for, because clang has used packed SSE instructions to add 16 bytes at a time. It also manages `load` and `store` two at a time, which it did not do last time. The question is now that we understand what we need to tell the compiler, how complex can the loop be before auto-vectorization fails?

### 2.5 Turning on AVX2 instructions

Next, we try to turn on AVX2 instructions using the following command, which generates `assembly/test1.vec.restr.align.avx2.s`

```shell
make clean; make test1.o ASSEMBLE=1 VECTORIZE=1 RESTRICT=1 ALIGN=1 AVX2=1
```

Let’s see the difference:

```shell
diff assembly/test1.vec.restr.align.s assembly/test1.vec.restr.align.avx2.s
```

We can see instructions with the prefix `v*`. That’s good. We confirm the compiler uses AVX2 instructions; however, this code is still not aligned when using AVX2 registers.

> **Q2-1:** Fix the code to make sure it uses aligned moves for the best performance.
> Hint: we want to see `vmovaps` rather than `vmovups`.

### 2.6 Performance impacts of vectorization

Let’s see what speedup we get from vectorization. Build and run the program with the following configurations, which run `test1()` many times, and record the elapsed execution time.

```shell
# case 1
make clean && make && run -- ./test_auto_vectorize -t 1
# case 2
make clean && make VECTORIZE=1 && run -- ./test_auto_vectorize -t 1
# case 3
make clean && make VECTORIZE=1 AVX2=1 && run -- ./test_auto_vectorize -t 1
```

Note that you may wish to use the workstations provided by this course, which support AVX2; otherwise, you may get a message like “Illegal instruction (core dumped)”. You can check whether or not a machine supports the AVX2 instructions by looking for `avx2` in the flags section of the output of cat `/proc/cpuinfo`.

```shell
run -- cat /proc/cpuinfo | grep avx2
```

> **Q2-2:** : What speedup does the vectorized code achieve over the unvectorized code? What additional speedup does using `-mavx2` give (`AVX2=1` in the `Makefile`)? You may wish to run this experiment several times and take median elapsed times; you can report answers to the nearest 100% (e.g., 2×, 3×, etc). What can you infer about the bit width of the default vector registers on the PP machines? What about the bit width of the AVX2 vector registers?
> 
> Hint: Aside from speedup and the vectorization report, the most relevant information is that the data type of each array is `float`.

You may also run `test2()` and `test3()` with `run -- ./test_auto_vectorize -t 2` and `run -- ./test_auto_vectorize -t 3`, respectively, before and after fixing the vectorization issues in Section 2.7.

### 2.7 More examples

#### 2.7.1 Example 2 (`test2.c`)

Original:

```c
void test2(float *__restrict a, float *__restrict b, float *__restrict c, int N)
{
  __builtin_assume(N == 1024);
  a = (float *)__builtin_assume_aligned(a, 16);
  b = (float *)__builtin_assume_aligned(b, 16);
  c = (float *)__builtin_assume_aligned(c, 16);

  for (int i = 0; i < I; i++)
  {
    for (int j = 0; j < N; j++)
    {
      /* max() */
      c[j] = a[j];
      if (b[j] > a[j])
        c[j] = b[j];
    }
  }
}
```

Compile the code with the following command:

```shell
make clean; make test2.o ASSEMBLE=1 VECTORIZE=1
```

Apply patch (`patch -i ./test2.c.patch`), which changes to:

```diff
-      c[j] = a[j];
-      if (b[j] > a[j])
-        c[j] = b[j];
+      if (b[j] > a[j]) c[j] = b[j];
+      else c[j] = a[j];
```

Now you should see vectorized asm with `movaps`/`maxps`.

> **Q2-3:** Provide a theory for why the compiler is generating dramatically different assemblies.

#### 2.7.2 Example 3 (`test3.c`)

```c
double test3(double *__restrict a, int N) {
  __builtin_assume(N == 1024);
  a = (double *)__builtin_assume_aligned(a, 16);

  double b = 0;

  for (int i=0; i<I; i++) {
    for (int j=0; j<N; j++) {
      b += a[j];
    }
  }
  return b;
}
```

Compile the code with the following command:

```shell
make clean; make test3.o ASSEMBLE=1 VECTORIZE=1
```

You should see the non-vectorized code with the `addsd` instructions.

Notice that this is not vectorization as the xmm registers are operating only on 8-byte chunks. The problem here is that clang is not allowed to re-order the operations we give it. Even though addition is associative over real numbers, they aren’t associative over floating point numbers. (Consider computers can’t express floating point numbers precisely, leading to errors in calculations.)

Furthermore, we need to tell clang that reordering operations is okay with us. To do this, we need to add another compile-time flag, `-ffast-math`. Compile the program again with the following command:

```shell
make clean; make test3.o ASSEMBLE=1 VECTORIZE=1 FASTMATH=1
```

Now you should see vectorized `addpd`.

---

## 3. Requirements

Use **HackMD** for your **REPORT** and answer questions (**Q1 & Q2**).

### 3.1 Part 1

1. Implement `clampedExpVector` (fake intrinsics). Works for any `N` and `VECTOR_WIDTH`, **utilization > 60%**, **passes verification**. (Assume `N` ≫ `VECTOR_WIDTH`.)
2. Run `run -- ./myexp -s 10000` with `VECTOR_WIDTH = 2, 4, 8, 16`. Record vector utilization.
   **Q1-1:** As `VECTOR_WIDTH` changes, does utilization increase/decrease/same? Why?
3. **Bonus:** Implement `arraySumVector` with span `O(N / VECTOR_WIDTH + log2(VECTOR_WIDTH))`, **utilization > 80%**, pass verification. Assume even `VECTOR_WIDTH` and divides `N`.

### 3.2 Part 2

Answer **Q2-1**, **Q2-2**, **Q2-3**. We don’t test your code, but explain clearly. If you have code for answering the questions, show the code and explain it thoroughly in your report.

---

## 4. Grading Policy

**NO CHEATING!!**

Total 100%:

* **Part 1 (70%)**

  * Correctness (50%): `clampedExpVector` meets **all** requirements (else 0%).
  * Question (10%): Q1-1 graded: excellent (10) / good (7) / normal (3) / terrible (0).
  * Bonus (10%): `arraySumVector` meets **all** bonus requirements (else 0%).
* **Part 2 (30%)**

  * Q2-1 \~ Q2-3: each 10%, same 4-tier rubric.

---

## 5. Evaluation Platform

UNIX-like OS. Workstations: Debian 12.9, Intel® Core™ i5-10500 @ 3.10GHz, GeForce GTX 1060 6GB. Installed: `g++-12`, `clang++-11`, `CUDA 12.8`.

---

## 6. Submission

Zip structure (file named `HW1_xxxxxxx.zip`, where `xxxxxxx` = your student ID):

```
HW1_xxxxxxx.zip
├─ vectorOP.cpp
└─ url.txt
```

Zip the file:

```shell
zip HW1_xxxxxxx.zip vectorOP.cpp url.txt
```

Notice that you just need to provide the URL of your HackMD report in `url.txt`, and enable the write permission for someone who knows the URL so that TAs can give you feedback directly in your report.

**Run `test_hw1 <your_student_id>` on the workstation** (in the directory containing your zip). It verifies structure and runs graders (reference only) and **automatically** uses compute node.

```shell
test_hw1 <your_student_id>
```

> **Notes**
>
> * It may take a few minutes.
> * Upload to the new E3 e-Campus by the due date.
> * **No point** if zip name or hierarchy is wrong.
> * **-5 points** if you include unnecessary files (obj, `.vscode`, `.__MACOSX`, etc.).

---

## 7. References

* [Wikipedia: Analysis of parallel algorithms](https://en.wikipedia.org/wiki/Analysis_of_parallel_algorithms)
* [Wikipedia: SIMD](https://en.wikipedia.org/wiki/SIMD)
* [Clang: built-in functions document](https://clang.llvm.org/docs/LanguageExtensions.html#builtin-functions)
* [Slurm Workload Manager: Documentation](https://slurm.schedmd.com/documentation.html)
* [Video: Markdown 使用教學](https://www.youtube.com/watch?v=Or6adjo3W4E&list=PLCOCSTovXmudP_dZi1T9lNHLOtqpK9e2P&index=19)
* [Video: SSH & SCP 使用教學](https://www.youtube.com/watch?v=PYdM2vN4BpE&list=PLCOCSTovXmudP_dZi1T9lNHLOtqpK9e2P&index=15)
