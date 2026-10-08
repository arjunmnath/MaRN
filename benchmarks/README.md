# Mapping Networks Benchmarks

These scripts compare direct training with mapping strategies, plus loss
ablation and add-on experiments. Every MNIST benchmark uses the complete
official training and validation splits (60,000 and 10,000 examples). The
deepfake script uses its complete generated `torchvision.datasets.FakeData`
splits (512 training and 256 validation examples), not a real deepfake dataset.
The LSTM benchmark uses all windows generated from its configured sine-wave
train and validation series.

## Benchmark scripts

- `benchmark_cnn.py`: CNN1 and CNN2 image classification on MNIST.
- `benchmark_deepfake.py`: binary classification on synthetic `FakeData`.
- `benchmark_lstm.py`: next-step forecasting on generated sine-wave data.
- `benchmark_loss_ablation.py`: mapping loss component ablation on MNIST.
- `benchmark_addons.py`: LRD and pruning experiments on MNIST.
- `run_benchmarks.py`: runs the scripts above and writes `benchmark_report.md`.

The runner is single-process. Run it from the repository root with:

```bash
python benchmarks/run_benchmarks.py
```

## Results status

The numbers previously listed in this README are withdrawn. The MNIST
experiments selected the first items of each dataset, which are ordered by
class, so the training and validation subsets were almost entirely digit zero.
The old validation percentages therefore do not measure ten-class
classification. In addition, the direct and mapped CNN/deepfake runs used
different numbers of training epochs, and some peak-memory readings were
reported as a `0.50 MB` floor because CUDA device strings such as `cuda:0` were
not recognized by the profiler.

Those issues are now corrected in the benchmark code: MNIST runs consume the
full splits, comparisons use the configured epoch count, and CUDA profiling
accepts indexed device names. No replacement results are published here
because the training runs have not been rerun. Run the command above on the
intended hardware to generate a new report.
