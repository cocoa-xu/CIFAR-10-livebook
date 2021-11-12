defmodule MyElixirBenchmark.MixProject do
  use Mix.Project

  def project do
    [
      app: :my_elixir_benchmark,
      version: "0.1.0",
      elixir: "~> 1.12",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:torchx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "torchx"},
      {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true},
    ]
  end
end
