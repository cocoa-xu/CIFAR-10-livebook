# CIFAR-10-livebook
A simple example using [Numerical Elixir (Nx)](https://github.com/elixir-nx) and [Elixir livebook](https://github.com/livebook-dev/livebook)!

Dataset: [cifar-10-binary.tar.gz](https://www.cs.toronto.edu/\~kriz/cifar-10-binary.tar.gz)

P.S. The goal of this benchmark is only to evaluate the matrix computation performance, instead of getting a decent (or even acceptable) CIFAR-10 prediction accuracy.

## TL;DR

1. Use C libraries (via NIF) for matrix computation when performance is top priority. Otherwise it is about 10^3 times slower in terms of matrix computation.
2. OTP 25 introduces JIT on ARM64 and it shows 3-4% performance improvement (matrix computation).
3. Almost linear speedup can be achieved when a large computation task can be divided into independent smaller ones.
4. Apple M1 Max performs much better than its x86_64 competitors (Intel Core i9 8950HK and AMD Ryzen 9 3900XT).

For some more details, [https://cocoa-research.works/2021/11/numerical-elixir-benchmark-cifar10-with-3-layer-densenn/](https://cocoa-research.works/2021/11/numerical-elixir-benchmark-cifar10-with-3-layer-densenn/)

## Run this benchmark
```
$ export LIBTORCH_DIR=/path/to/libtorch
$ mix deps.get
$ iex -S mix
iex(1)> # path to the directory that contains CIFAR10 .bin files
iex(2)> datadir = __ENV__.file |> Path.dirname() |> Path.join(["cifar10-dataset"])
...
iex(3)> Benchmark.run(datadir: datadir, backend: Torchx.Backend, batch_size: 300)
...
iex(4)> Benchmark.run(
...(4)>   datadir: datadir, 
...(4)>   backend: Nx.BinaryBackend,
...(4)>   batch_size: 300,
...(4)>   n_jobs: 1
...(4)> )
...
iex(5)> Benchmark.run(
...(5)>   datadir: datadir, 
...(5)>   backend: Nx.BinaryBackend,
...(5)>   batch_size: 250 * System.schedulers_online(),
...(5)>   n_jobs: System.schedulers_online()
...(5)> )
```

### Results

Numbers are in seconds.

| Hardware     | Backend                | OTP | Load Dataset | To Batched Input | Mean Epoch Time |
|--------------|------------------------|-----|-------------:|-----------------:|----------------:|
| RPi 4        | Binary (Single-thread) | 24  |              |                  |                 |
| RPi 4        | Binary (Multi-thread)  | 24  |              |                  |                 |
| RPi 4        | Binary (Single-thread) | 25  | 194.427      | 11.917           | 27336.010       |
| RPi 4        | Binary (Multi-thread)  | 25  |              |                  |                 |
| RPi 4        | LibTorch CPU           | 24  | 15.334       | 4.880            | 17.170          |
| RPi 4        | LibTorch CPU           | 25  | 16.372       | 4.442            | 16.207          |
| Intel 8950HK | Binary (Single-thread) | 24  | 17.994       | 3.036            | 4460.758        |
| Intel 8950HK | Binary (Multi-thread)  | 24  | 17.826       | 2.934            | 1471.090        |
| Intel 8950HK | LibTorch CPU           | 24  | 2.141        | 0.778            | 0.841           |
| Ryzen 3900XT | Binary (Single-thread) | 24  | 6.058        | 2.391            | 3670.930        |
| Ryzen 3900XT | Binary (Multi-thread)  | 24  | 6.034        | 2.536            | 786.443         |
| Ryzen 3900XT | LibTorch CPU           | 24  |              |                  |                 |
| Ryzen 3900XT | LibTorch GPU           | 24  | 1.630        | 0.652            | 0.564           | 
| M1 Max       | Binary (Single-thread) | 24  | 11.090       | 2.135            | 3003.321        |
| M1 Max       | Binary (Multi-thread)  | 24  | 10.925       | 2.154            | 453.536         |
| M1 Max       | Binary (Single-thread) | 25  | 9.458        | 1.548            | 3257.853        |
| M1 Max       | Binary (Multi-thread)  | 25  | 9.949        | 1.527            | 436.385         |
| M1 Max       | LibTorch CPU           | 24  | 1.702        | 1.900            | 0.803           |
| M1 Max       | LibTorch CPU           | 25  | 1.599        | 0.745            | 0.773           | 

RPi 4: Raspberry Pi 4, 8 GB RAM

Intel 8950HK: 6 Cores 12 Threads, MacBook Pro (15-inch, 2018), 32 GB RAM

Ryzen 3900XT: 12 Cores 24 Threads, Desktop PC, 64 GB RAM

M1 Max: 10 Cores (8 Performance + 2 Effiency) 10 Threads, MacBook Pro (14-inch, 2021), 64 GB RAM

Binary (Single-thread): All computation related to training is done in a single erlang thread

Binary (Multi-thread): All computation related to training is done in multi erlang threads (System.schedulers_online())

LibTorch CPU: v1.9.1

LibTorch GPU: v1.9.1, CUDA 11.1, cuDNN 8.2.1, NVIDIA RTX 3090 24 GB

OTP 24: 24.0.6 (installed by asdf)

OTP 25: [b58c66e12](https://github.com/erlang/otp/tree/b58c66e123521bc8f2b2c9332f41ce8093a90dbc)
