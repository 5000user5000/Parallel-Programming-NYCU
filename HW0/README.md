# HW0: Monte Carlo Estimation of π

This project is the **Homework 0 (warming up)** for NYCU Parallel Programming.  
The task is to implement a simple **Monte Carlo simulation** to estimate the value of π.

## How to Build & Run

### Build
```bash
make          # compile pi.cpp → pi.out
````

### Run

```bash
make run      # run the program
```

### Clean

```bash
make clean    # remove executables and profiling data
```


## Performance Profiling

This project also introduces basic **performance profiling tools**:

* **time** – measure execution time

  ```bash
  time ./pi.out
  ```

* **gprof** – function-level profiling

  ```bash
  make pg
  ```

* **perf** – hardware performance counter profiling

  ```bash
  make perf
  ```

> Note: For `perf`, you may need to adjust kernel permissions:
>
> ```bash
> echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid
> ```


## 📂 Files

* `pi.cpp` – source code for Monte Carlo π estimation
* `Makefile` – build and profiling commands
* `README.md` – project description