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
