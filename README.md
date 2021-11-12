# CIFAR-10-livebook
A simple example using [Numerical Elixir (Nx)](https://github.com/elixir-nx) and [Elixir livebook](https://github.com/livebook-dev/livebook)!

## Run this benchmark
```
$ export LIBTORCH_DIR=/path/to/libtorch
$ mix deps.get
$ iex -S mix
iex(1)> # choose backend
iex(2)> # backend = Nx.BinaryBackend
iex(3)> backend = Torchx.Backend
{Nx.BinaryBackend, []}
iex(4)> # path to the directory that contains CIFAR10 .bin files
iex(5)> datadir = __ENV__.file |> Path.dirname() |> Path.join(["cifar10-dataset"])
...
iex(6)> Benchmark.run(datadir: datadir, backend: backend)
...
```
