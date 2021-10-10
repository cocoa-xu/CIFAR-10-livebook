# CIFAR-10-livebook
A simple example using [Numerical Elixir (Nx)](https://github.com/elixir-nx) and [Elixir livebook](https://github.com/livebook-dev/livebook)!

## Code
```elixir
# Title: CIFAR-10

# ── Section ──

# optional
# use Torch or XLA will be much faster
Nx.default_backend(Torchx.Backend)

# expected directory layout
# .
# ├── CIFAR-10.livemd
# ├── data_batch_1.bin
# ├── data_batch_2.bin
# ├── data_batch_3.bin
# ├── data_batch_4.bin
# ├── data_batch_5.bin
# └── test_batch.bin
defmodule CIFAR10 do
  defp get_batch_file("data", batch_num) do
    # CIFAR-10 training dataset
    __ENV__.file
    |> Path.dirname()
    |> Path.join(["data_batch_" <> to_string(batch_num) <> ".bin"])
  end

  defp get_batch_file("test", _) do
    # CIFAR-10 test dataset
    __ENV__.file
    |> Path.dirname()
    |> Path.join(["test_batch.bin"])
  end

  defp read_batch(type, batch_num) do
    # get corresponding batch
    filename = get_batch_file(type, batch_num)
    # read batch file
    File.read(filename)
  end

  def rgb_to_grey(chw) do
    # using the NTSC formula: 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue
    chw
    |> Nx.transpose(axes: [1, 2, 0])
    |> Nx.dot(Nx.tensor([0.299, 0.587, 0.114]))
  end

  defp get_dataset() do
    # read training dataset
    {:ok, training_data} = get_dataset("data", Enum.to_list(1..5), <<>>)
    # read test dataset
    {:ok, test_data} = get_dataset("test", [nil], <<>>)
    {:ok, training_data, test_data}
  end

  def get_dataset(colour_type) do
    {:ok, training_data, test_data} = get_dataset()
    # convert to Nx arrays
    {:ok, x_training, y_training} = to_nx_array(training_data, colour_type)
    {:ok, x_test, y_test} = to_nx_array(test_data, colour_type)
    # return greyscale dataset
    {Nx.as_type(x_training, {:f, 32}), Nx.as_type(y_training, {:f, 32}), x_test, y_test}
  end

  def get_dataset(type, [batch_num | rest], all_data) do
    # read current batch file
    {:ok, data} = read_batch(type, batch_num)
    # read rest batch file
    get_dataset(type, rest, all_data <> data)
  end

  def get_dataset(_type, [], all_data) do
    # no more batch file to read
    {:ok, all_data}
  end

  defp to_nx_array(raw_data, colour_type) do
    # get number of bytes
    num_bytes = byte_size(raw_data)
    # according to https://www.cs.toronto.edu/~kriz/cifar.html
    # the layout of each sample is <1 x label><3072 x pixel>
    # therefore, rem(num_bytes, 3073) must be 0
    0 = rem(num_bytes, 3073)
    # get number of samples
    num_samples = trunc(num_bytes / 3073 - 1)

    # read samples
    samples =
      case colour_type do
        :rgb ->
          for(
            i <- Enum.to_list(0..num_samples),
            do:
              :binary.part(raw_data, i * 3073, 3072)
              |> Nx.from_binary({:u, 8})
          )
          |> Nx.concatenate()
          |> Nx.reshape({num_samples + 1, 3, 32, 32}, names: [:n, :c, :h, :w])

        :grey ->
          for(
            i <- Enum.to_list(0..num_samples),
            do:
              :binary.part(raw_data, i * 3073, 3072)
              |> Nx.from_binary({:u, 8})
              |> Nx.reshape({3, 32, 32}, names: [:c, :h, :w])
              |> rgb_to_grey()
          )
          |> Nx.concatenate()
          |> Nx.reshape({num_samples + 1, 32, 32}, names: [:n, :h, :w])
      end

    # get labels
    labels = Nx.tensor(for i <- Enum.to_list(0..num_samples), do: :binary.at(raw_data, i * 3073))
    # :ok
    {:ok, samples, labels}
  end
end

# training code
# based on https://github.com/elixir-nx/nx/blob/e4454423f7be39d3adc9dea76526185fbfaf7a58/exla/examples/mnist.exs
defmodule DenseNN do
  import Nx.Defn

  defn init_random_params do
    # 3 layers
    #  1. Dense(64) with sigmoid
    #  2. Dense(32) with sigmoid
    #  3. Dense(10) with softmax
    w1 = Nx.random_normal({1024, 64}, 0.0, 0.1, names: [:input, :layer1])
    b1 = Nx.random_normal({64}, 0.0, 0.1, names: [:layer1])
    w2 = Nx.random_normal({64, 32}, 0.0, 0.1, names: [:layer1, :layer2])
    b2 = Nx.random_normal({32}, 0.0, 0.1, names: [:layer2])
    w3 = Nx.random_normal({32, 10}, 0.0, 0.1, names: [:layer2, :output])
    b3 = Nx.random_normal({10}, 0.0, 0.1, names: [:output])
    {w1, b1, w2, b2, w3, b3}
  end

  defn softmax(logits) do
    Nx.exp(logits) /
      Nx.sum(Nx.exp(logits), axes: [:output], keep_axes: true)
  end

  defn predict({w1, b1, w2, b2, w3, b3}, batch) do
    batch
    |> Nx.dot(w1)
    |> Nx.add(b1)
    |> Nx.logistic()
    |> Nx.dot(w2)
    |> Nx.add(b2)
    |> Nx.logistic()
    |> Nx.dot(w3)
    |> Nx.add(b3)
    |> softmax()
  end

  defn accuracy({w1, b1, w2, b2, w3, b3}, batch_images, batch_labels) do
    Nx.mean(
      Nx.equal(
        Nx.argmax(batch_labels, axis: :output),
        Nx.argmax(predict({w1, b1, w2, b2, w3, b3}, batch_images), axis: :output)
      )
      |> Nx.as_type({:s, 8})
    )
  end

  defn loss({w1, b1, w2, b2, w3, b3}, batch_images, batch_labels) do
    preds = predict({w1, b1, w2, b2, w3, b3}, batch_images)
    -Nx.sum(Nx.mean(Nx.log(preds) * batch_labels, axes: [:output]))
  end

  defn update({w1, b1, w2, b2, w3, b3} = params, batch_images, batch_labels, step) do
    {grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3} =
      grad(params, &loss(&1, batch_images, batch_labels))

    {
      w1 - grad_w1 * step,
      b1 - grad_b1 * step,
      w2 - grad_w2 * step,
      b2 - grad_b2 * step,
      w3 - grad_w3 * step,
      b3 - grad_b3 * step
    }
  end

  defn update_with_averages(
         {_, _, _, _, _, _} = cur_params,
         imgs,
         tar,
         avg_loss,
         avg_accuracy,
         total
       ) do
    batch_loss = loss(cur_params, imgs, tar)
    batch_accuracy = accuracy(cur_params, imgs, tar)
    avg_loss = avg_loss + batch_loss / total
    avg_accuracy = avg_accuracy + batch_accuracy / total
    {update(cur_params, imgs, tar, 0.01), avg_loss, avg_accuracy}
  end

  def train_epoch(cur_params, x, labels) do
    total_batches = Enum.count(x)

    x
    |> Enum.zip(labels)
    |> Enum.reduce({cur_params, Nx.tensor(0.0), Nx.tensor(0.0)}, fn
      {x, tar}, {cur_params, avg_loss, avg_accuracy} ->
        update_with_averages(cur_params, x, tar, avg_loss, avg_accuracy, total_batches)
    end)
  end

  def train(x, labels, params, opts \\ []) do
    epochs = opts[:epochs] || 5

    for epoch <- 1..epochs, reduce: params do
      cur_params ->
        {time, {new_params, epoch_avg_loss, epoch_avg_acc}} =
          :timer.tc(__MODULE__, :train_epoch, [cur_params, x, labels])

        epoch_avg_loss =
          epoch_avg_loss
          |> Nx.backend_transfer()
          |> Nx.to_scalar()

        epoch_avg_acc =
          epoch_avg_acc
          |> Nx.backend_transfer()
          |> Nx.to_scalar()

        IO.puts("Epoch #{epoch} Time: #{time / 1_000_000}s")
        IO.puts("Epoch #{epoch} average loss: #{inspect(epoch_avg_loss)}")
        IO.puts("Epoch #{epoch} average accuracy: #{inspect(epoch_avg_acc)}")
        IO.puts("\n")
        new_params
    end
  end
end

defmodule Helper do
  def to_onehot_single(0, oh, _pos) do
    oh
  end

  def to_onehot_single(count, oh, pos) do
    cur = count - 1

    case cur == pos do
      true -> to_onehot_single(count - 1, [1] ++ oh, pos)
      _ -> to_onehot_single(count - 1, [0] ++ oh, pos)
    end
  end

  def to_onehot_single(0, _pos) do
    []
  end

  def to_onehot_single(count, pos) do
    to_onehot_single(count, [], pos)
  end

  def to_onehot(labels, unique_classes) do
    for(
      l <- Nx.to_flat_list(labels),
      do: Nx.tensor([to_onehot_single(unique_classes, l)])
    )
    |> Nx.concatenate()
    |> Nx.reshape({:auto, unique_classes}, names: [:batch, :output])
  end
end

{x_training, y_training, x_test, y_test} = CIFAR10.get_dataset(:grey)

batch_size = 300
unique_classes = 10

x_training_batched =
  x_training
  # uint8 to float
  |> Nx.divide(255)
  # flatten
  |> Nx.reshape({:auto, 1024})
  |> Nx.as_type({:f, 32})
  |> Nx.to_batched_list(batch_size)

y_training_batched =
  y_training
  |> Helper.to_onehot(unique_classes)
  |> Nx.as_type({:f, 32})
  |> Nx.to_batched_list(batch_size)

params = DenseNN.init_random_params()

final_params =
  DenseNN.train(
    x_training_batched,
    y_training_batched,
    params,
    epochs: 50
)

```
