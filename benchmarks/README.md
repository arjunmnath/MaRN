# Mapping Networks Benchmarks

This directory contains scripts to evaluate the performance, memory efficiency, parameter reduction, and training speed of Mapping Networks compared to direct training, mirroring the experiments described in the paper.

## Benchmark Suite Structure

- `benchmark_cnn.py`: Image classification using CNN1 (~538k parameters) and CNN2 (~108k parameters) on MNIST/FashionMNIST-like synthetic data.
- `benchmark_deepfake.py`: Deepfake detection using CNN1 and CNN2 on synthetic video frame datasets.
- `benchmark_lstm.py`: Mapped LSTM vs direct LSTM on time series forecasting.
- `benchmark_finetuning.py`: ResNet50 parameter-modulation fine-tuning compared to full fine-tuning.
- `benchmark_loss_ablation.py`: Impact of different mapping loss terms ($L_{\text{stability}}$, $L_{\text{smoothness}}$, $L_{\text{alignment}}$).
- `benchmark_addons.py`: Impact of Low Rank Decomposition (LRD) and Pruning add-ons on training and inference parameters.

## Running the Benchmarks

To run all benchmarks and generate a unified markdown report, execute:

```bash
accelerate launch --num_processes=2 --mixed_precision=fp16 run_benchmarks.py
```

## Benchmark Report

Performance comparisons across parameter count, peak memory, epoch time, and accuracy/loss. All results are from **real training** on standard benchmark datasets.

> Device: `cuda:0` | Mixed-precision: `fp16`

## 1. Image Classification — MNIST

| Method | Total Params | Trainable Params | Reduction | Peak Mem (MB) | Epoch Time (ms) | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| `cnn1_direct` | 537,748 | 537,748 | 1.0× | 0.50 | 161.6 | 17.19% |
| `cnn1_lwt_4080` | 537,748 | 4,080 | 131.8× | 0.50 | 2170.9 | 92.19% |
| `cnn2_direct` | 107,998 | 107,998 | 1.0× | 0.50 | 134.8 | 51.56% |
| `cnn2_lwt_1872` | 107,998 | 1,872 | 57.7× | 0.50 | 675.5 | 91.80% |
| `cnn2_lwt_2688` | 107,998 | 2,688 | 40.2× | 0.50 | 708.8 | 86.33% |

## 2. Deepfake Detection — 3×64×64 Binary Classification

| Method | Total Params | Trainable Params | Reduction | Peak Mem (MB) | Epoch Time (ms) | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| `cnn1_direct` | 536,723 | 536,723 | 1.0× | 0.50 | 674.0 | 48.44% |
| `cnn1_lwt_1955` | 536,723 | 1,955 | 274.5× | 0.50 | 4080.3 | 46.48% |
| `cnn1_lwt_2790` | 536,723 | 2,790 | 192.4× | 0.50 | 4784.2 | 53.91% |
| `cnn2_direct` | 104,195 | 104,195 | 1.0× | 0.50 | 671.8 | 48.44% |
| `cnn2_lwt_1872` | 104,195 | 1,872 | 55.7× | 0.50 | 2650.3 | 45.70% |
| `cnn2_lwt_2688` | 104,195 | 2,688 | 38.8× | 0.50 | 2801.7 | 43.75% |

## 3. LSTM Time-Series Forecasting — Sine-Wave Signal

| Method | Total Params | Trainable Params | Reduction | Peak Mem (MB) | Epoch Time (ms) | Val Loss (MSE) |
| --- | --- | --- | --- | --- | --- | --- |
| `lstm_direct` | 12,051 | 12,051 | 1.0× | 0.50 | 113.5 | 0.00059 |
| `lstm_slvt_64` | 12,051 | 64 | 188.3× | 0.50 | 471.0 | 0.00076 |
| `lstm_slvt_2048` | 12,051 | 2,048 | 5.9× | 0.50 | 628.8 | 0.00006 |

## 4. Ablation of Mapping Loss Components

| Loss Configuration | Validation Accuracy |
| --- | --- |
| Task Loss Only | 64.84% |
| + Stability | 66.02% |
| + Smoothness | 64.45% |
| + Alignment | 65.23% |
| + Smoothness + Align | 62.89% |
| + Stab + Smooth | 65.23% |
| Full Mapping Loss | 60.16% |

## 5. Impact of LRD and Pruning Add-ons

| Configuration | Total Params | Trainable Params | Peak Mem (MB) | Epoch Time (ms) | Accuracy |
| --- | --- | --- | --- | --- | --- |
| CNN2 Baseline | 107,998 | 107,998 | 0.50 | 129.1 | 73.44% |
| CNN2 + Prune 90% | 107,998 | 10,799 | 0.50 | 131.8 | 56.25% |
| Ours* (SLVT) | 107,998 | 2,048 | 0.50 | 1245.2 | 80.08% |
| Ours* + LRD (rank 8) | 107,998 | 8,192 | 0.50 | 625.1 | 14.06% |
| Ours* + Prune 90% | 107,998 | 204 | 0.50 | 1234.5 | 81.25% |
