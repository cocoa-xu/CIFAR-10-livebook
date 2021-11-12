defmodule CIFAR10 do
  @doc """
  # expected directory layout
  # datadir
  # ├── data_batch_1.bin
  # ├── data_batch_2.bin
  # ├── data_batch_3.bin
  # ├── data_batch_4.bin
  # ├── data_batch_5.bin
  # └── test_batch.bin
  """
  def get_batch_file("data", batch_num, datadir) do
    # CIFAR-10 training dataset
    datadir
    |> Path.join(["data_batch_" <> to_string(batch_num) <> ".bin"])
  end

  def get_batch_file("test", _, datadir) do
    # CIFAR-10 test dataset
    datadir
    |> Path.join(["test_batch.bin"])
  end

  defp read_batch(type, batch_num, datadir) do
    # get corresponding batch
    filename = get_batch_file(type, batch_num, datadir)
    # read batch file
    File.read(filename)
  end

  def rgb_to_grey(chw) do
    # http://support.ptc.com/help/mathcad/en/index.html#page/PTC_Mathcad_Help/example_grayscale_and_color_in_images.html
    # using the NTSC formula: 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue
    chw
    |> Nx.transpose(axes: [1, 2, 0])
    |> Nx.dot(Nx.tensor([0.299, 0.587, 0.114]))
  end

  def get_all_dataset(datadir \\ __ENV__.file |> Path.dirname()) do
    # read training dataset
    {:ok, training_data} = get_dataset("data", Enum.to_list(1..5), <<>>, datadir)
    # read test dataset
    {:ok, test_data} = get_dataset("test", [nil], <<>>, datadir)
    {:ok, training_data, test_data}
  end

  def get_dataset(colour_type, datadir \\ __ENV__.file |> Path.dirname()) do
    {:ok, training_data, test_data} = get_all_dataset(datadir)
    # convert to Nx arrays
    {:ok, x_training, y_training} = to_nx_array(training_data, colour_type)
    {:ok, x_test, y_test} = to_nx_array(test_data, colour_type)
    # return greyscale dataset
    {Nx.as_type(x_training, {:f, 32}), Nx.as_type(y_training, {:f, 32}), x_test, y_test}
  end

  def get_dataset(type, [batch_num | rest], all_data, datadir) do
    # read current batch file
    {:ok, data} = read_batch(type, batch_num, datadir)
    # read rest batch file
    get_dataset(type, rest, all_data <> data, datadir)
  end

  def get_dataset(_type, [], all_data, _datadir) do
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
          Enum.to_list(0..num_samples)
          |> Enum.map(fn i -> Task.async(fn -> :binary.part(raw_data, i * 3073, 3072)
            |> Nx.from_binary({:u, 8}) end) end)
          |> Enum.map(&Task.await(&1, :infinity))
          |> Nx.concatenate()
          |> Nx.reshape({num_samples + 1, 3, 32, 32}, names: [:n, :c, :h, :w])

        :grey ->
          Enum.to_list(0..num_samples)
          |> Enum.map(fn i -> Task.async(fn -> :binary.part(raw_data, i * 3073, 3072)
            |> Nx.from_binary({:u, 8})
            |> Nx.reshape({3, 32, 32}, names: [:c, :h, :w])
            |> rgb_to_grey() end) end)
          |> Enum.map(&Task.await(&1, :infinity))
          |> Nx.concatenate()
          |> Nx.reshape({num_samples + 1, 32, 32}, names: [:n, :h, :w])
      end

    # get labels
    labels =
      Enum.to_list(0..num_samples)
      |> Enum.map(fn i -> Task.async(fn -> :binary.at(raw_data, i * 3073) end) end)
      |> Enum.map(&Task.await(&1, :infinity))
      |> Nx.tensor()

    # :ok
    {:ok, samples, labels}
  end
end
