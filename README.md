# Parallel Programming @ NYCU

![HW1-Part1](https://github.com/5000user5000/Parallel-Programming-NYCU/actions/workflows/hw1-part1-validation.yml/badge.svg)
![HW2-Part1](https://github.com/5000user5000/Parallel-Programming-NYCU/actions/workflows/hw2-part1-validation.yml/badge.svg)
![HW2-Part2](https://github.com/5000user5000/Parallel-Programming-NYCU/actions/workflows/hw2-part2-validation.yml/badge.svg)
![HW3-Part1](https://github.com/5000user5000/Parallel-Programming-NYCU/actions/workflows/hw3-part1-validation.yml/badge.svg)

**Fall 2025 | National Yang Ming Chiao Tung University**
**Instructor: Prof. Yi-Ping You**

This repository contains assignments from the Parallel Programming course, covering fundamental parallel computing paradigms from SIMD to distributed computing.

## Technologies Covered

| Assignment | Topic | Technology |
|------------|-------|------------|
| HW0 | Monte Carlo π Estimation | Profiling (gprof, perf) |
| HW1 | SIMD Programming | SSE/AVX2 Intrinsics, Auto-vectorization |
| HW2 | Thread-level Parallelism | Pthreads |
| HW3 | Shared Memory Parallelism | OpenMP |
| HW4 | Distributed Computing | MPI |
| HW5 | GPU Computing | CUDA |
| HW6 | Heterogeneous Computing | OpenCL |

## Assignment Overview

### HW0: Warm-up
Monte Carlo simulation to estimate π. Introduces performance profiling tools (`time`, `gprof`, `perf`).

### HW1: SIMD Programming
- **Part 1**: Implement vector operations using fake SIMD intrinsics (educational abstraction)
- **Part 2**: Explore compiler auto-vectorization with `__restrict`, alignment hints, and `-mavx2`

### HW2: Pthreads
- **Part 1**: Multi-threaded Monte Carlo π with pthreads
- **Part 2**: Parallel sorting algorithms

### HW3: OpenMP
- **Part 1**: Parallelize Conjugate Gradient (CG) solver from NAS Parallel Benchmarks
- **Part 2**: Graph algorithms - Breadth-First Search and PageRank

### HW4: MPI
- **Part 1**: Odd-Even Sort with MPI
- **Part 2**: Distributed matrix multiplication (SUMMA algorithm, cache blocking)

### HW5: CUDA
Mandelbrot set computation on GPU. Explores thread/block configuration and memory optimization.

### HW6: OpenCL
Image convolution using OpenCL. Applies filters for image processing on heterogeneous devices.

## Environment

- **Platform**: NYCU CSIT Workstations (Slurm-managed cluster)
- **Hardware**: Intel Core i5-10500, NVIDIA GTX 1060
- **Software**: GCC 12, Clang 11, CUDA 12.8, OpenMPI

## Development Process

Each assignment was developed incrementally with commits documenting the optimization journey. PRs are organized by assignment (e.g., `hw5/cuda`, `hw6-opencl`).

To understand how a specific optimization was implemented, check the [Pull Requests](https://github.com/5000user5000/Parallel-Programming-NYCU/pulls?q=is%3Apr) - each PR contains step-by-step commits showing the evolution from initial implementation to final optimized version.

## CI (Just for Fun)

HW1-HW3 have GitHub Actions workflows for automated testing. This is **not a course requirement** - just a personal experiment with CI/CD. The workflows run on push/PR to validate correctness before merging.

## Quick Start

Each assignment folder contains its own build instructions. General pattern:

```bash
cd HW<n>
make
./executable [args]
```

For workstation submissions, use the Slurm wrapper:
```bash
run -- ./executable [args]
```
