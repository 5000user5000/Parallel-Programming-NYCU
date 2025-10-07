# HW2 Part1: Monte Carlo Estimation of π with Pthreads

This project is **Homework 2 Part 1** for NYCU Parallel Programming.
The task is to implement a **multi-threaded Monte Carlo simulation** using **pthreads** to estimate the value of π.

## How to Build & Run

### Build
```bash
make          # compile pi.cpp → pi.out
```

### Run

```bash
make run      # run with 4 threads and 1 billion tosses
./pi.out <num_threads> <num_tosses>
```

Example:
```bash
./pi.out 4 1000000000    # 4 threads, 1 billion tosses
```

### Clean

```bash
make clean    # remove executables
```


## Performance Profiling

Measure execution time with different thread counts:

```bash
time ./pi.out 1 1000000000
time ./pi.out 2 1000000000
time ./pi.out 4 1000000000
```

## 📂 Files

* `pi.cpp` – multi-threaded Monte Carlo π estimation using pthreads
* `Makefile` – build commands
* `README.md` – project description
* `include/` – header files for optimized random number generation
  * `Xoshiro256Plus.h` – fast, high-quality PRNG (Xoshiro256+ algorithm)
  * `SplitMix64.h` – seed initialization for Xoshiro256+
  * `SIMDInstructionSet.h` – SIMD instruction set enumeration