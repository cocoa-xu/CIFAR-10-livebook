defmodule Benchmark do
  require CIFAR10
  require DenseNN
  require Helper

  def load_dataset(datadir, backend) do
    {uSec, result} = :timer.tc(fn -> CIFAR10.get_dataset(:grey, datadir, backend) end)
    IO.puts("[Time] load dataset: #{uSec/1000.0} ms")
    result
  end

  def to_batched_input(x_training, y_training, batch_size) do
    unique_classes = 10

    x_training_batched =
      x_training
      # uint8 to float
      |> Nx.as_type({:f, 32})
      |> Nx.divide(255.0)
      # flatten
      |> Nx.reshape({:auto, 1024})
      |> Nx.to_batched_list(batch_size)

    y_training_batched =
      y_training
      |> Helper.to_onehot(unique_classes)
      |> Nx.as_type({:f, 32})
      |> Nx.to_batched_list(batch_size)

    {x_training_batched, y_training_batched}
  end

  def time_to_batched_input(x_training, y_training, batch_size) do
    {uSec, result} = :timer.tc(fn -> to_batched_input(x_training, y_training, batch_size) end)
    IO.puts("[Time] to batched input: #{uSec/1000.0} ms")
    result
  end

  def init_random_params do
    {uSec, result} = :timer.tc(fn -> DenseNN.init_random_params() end)
    IO.puts("[Time] init random params: #{uSec/1000.0} ms")
    result
  end

  def run(opts \\ []) do
    datadir = opts[:datadir] || __ENV__.file |> Path.dirname() |> Path.join(["../", "cifar10-dataset"])
    epochs = opts[:epochs] || 5
    backend = opts[:backend] || Nx.BinaryBackend
    batch_size = opts[:batch_size] || 300
    Nx.default_backend(backend)

    params = init_random_params()
    {x_training, y_training, _x_test, _y_test} = load_dataset(datadir, backend)
    {x_training_batched, y_training_batched} = time_to_batched_input(x_training, y_training, batch_size)

    {_final_params, _history_acc, _history_loss, history_time} = DenseNN.train(
      x_training_batched,
      y_training_batched,
      params,
      epochs: epochs
    )

    mean_epoch_time =
      history_time
      |> Nx.tensor()
      |> Nx.mean()
      |> Nx.to_scalar()
    IO.puts("[Time] mean epoch time: #{mean_epoch_time} secs")
  end
end
