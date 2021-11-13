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

  defn compute_gradient({_, _, _, _, _, _} = params, batch_images, batch_labels) do
    grad(params, &loss(&1, batch_images, batch_labels))
  end

  defn update({w1, b1, w2, b2, w3, b3} = _params, batch_grad, step) do
    {grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3} = batch_grad

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
         batch_grad,
         batch_loss,
         batch_acc,
         avg_loss,
         avg_accuracy,
         total
       ) do
    avg_loss = avg_loss + batch_loss / total
    avg_accuracy = avg_accuracy + batch_acc / total
    {update(cur_params, batch_grad, 0.01), avg_loss, avg_accuracy}
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
    batch_grad = compute_gradient(cur_params, imgs, tar)
    {update(cur_params, batch_grad, 0.01), avg_loss, avg_accuracy}
  end

  def train_epoch(cur_params, x, labels, Nx.BinaryBackend, n_jobs) when n_jobs >= 1 do
    total_batches = Enum.count(x)

    x
    |> Enum.zip(labels)
    |> Enum.reduce({cur_params, Nx.tensor(0.0), Nx.tensor(0.0)}, fn
      {x, tar}, {cur_params, avg_loss, avg_accuracy} ->
        [n_samples|_] = Nx.shape(x) |> Tuple.to_list()
        split_len =
          n_samples
          |> :erlang.div(n_jobs)
          |> round()

        x_splits = Nx.to_batched_list(x, split_len)
        tar_splits = Nx.to_batched_list(tar, split_len)

        [[first_grad, first_loss, first_acc]|rest_splits] =
          Enum.zip(x_splits, tar_splits)
          |> Enum.map(fn {imgs, labels} ->
            Task.async(fn ->
              Nx.default_backend(Nx.BinaryBackend)
              split_loss = loss(cur_params, imgs, labels)
              split_acc = accuracy(cur_params, imgs, labels)
              [compute_gradient(cur_params, imgs, labels), split_loss, split_acc]
            end)
          end)
          |> Enum.map(&Task.await(&1, :infinity))

        [batch_grad, batch_loss, batch_acc] =
          rest_splits |>
          Enum.reduce([Tuple.to_list(first_grad), first_loss, first_acc],
          fn [grad, loss, acc], [acc_grad, acc_loss, acc_acc] ->
            acc_grad =
              grad
              |> Tuple.to_list()
              |> Enum.zip(acc_grad)
              |> Enum.map(fn {current, total} -> Nx.add(total, current) end)
            [acc_grad, Nx.add(loss, acc_loss), Nx.add(acc, acc_acc)]
          end)
        batch_loss = Nx.divide(batch_loss, n_jobs)
        batch_acc = Nx.divide(batch_acc, n_jobs)

        update_with_averages(cur_params, List.to_tuple(batch_grad), batch_loss, batch_acc, avg_loss, avg_accuracy, total_batches)
    end)
  end

  def train_epoch(cur_params, x, labels, _backend, _n_jobs) do
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
    n_jobs = opts[:n_jobs] || 1
    backend = opts[:backend] || Nx.BinaryBackend

    for epoch <- 1..epochs, reduce: {params, [], [], []} do
      {cur_params, history_acc, history_loss, history_time} ->
        {time, {new_params, epoch_avg_loss, epoch_avg_acc}} =
          :timer.tc(__MODULE__, :train_epoch, [cur_params, x, labels, backend, n_jobs])

        epoch_avg_loss =
          epoch_avg_loss
          |> Nx.backend_transfer()
          |> Nx.to_scalar()

        epoch_avg_acc =
          epoch_avg_acc
          |> Nx.backend_transfer()
          |> Nx.to_scalar()

        epoch_time = time / 1_000_000

        history_acc = history_acc ++ [epoch_avg_acc]
        history_loss = history_loss ++ [epoch_avg_loss]
        history_time = history_time ++ [epoch_time]

        IO.puts("Epoch #{epoch} Time: #{Float.round(epoch_time, 3)}s, loss: #{Float.round(epoch_avg_loss, 3)}, acc: #{Float.round(epoch_avg_acc, 3)}")
        {new_params, history_acc, history_loss, history_time}
    end
  end
end
