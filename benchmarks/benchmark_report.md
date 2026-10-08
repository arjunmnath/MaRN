# Mapping Networks Benchmarks Report

This report compiles performance comparisons across parameters, peak memory, speed, and accuracy/loss for the models and strategies detailed in the paper.

## 1. Image Classification (MNIST/FashionMNIST-like targets)

| Method | Total Params | Trainable Params | Reduction | Peak Mem (MB) | Epoch Time (ms) | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| `cnn1_direct` | 537,748 | 537,748 | 1.0x | 0.50 | 7.7 | 6.25% |
| `cnn1_slvt_1024` | 537,748 | 1,024 | 525.1x | 120.00 | 325.3 | 93.02% |
| `cnn1_slvt_2072` | 537,748 | 2,072 | 259.5x | 120.00 | 607.1 | 93.91% |
| `cnn1_lwt_4078` | 537,748 | 4,080 | 131.8x | 120.00 | 93.9 | 94.83% |
| `cnn2_direct` | 107,998 | 107,998 | 1.0x | 0.50 | 6.2 | 6.25% |
| `cnn2_slvt_1024` | 107,998 | 1,024 | 105.5x | 120.00 | 105.3 | 89.49% |
| `cnn2_slvt_2048` | 107,998 | 2,048 | 52.7x | 120.00 | 160.6 | 91.88% |
| `cnn2_lwt_1872` | 107,998 | 1,872 | 57.7x | 0.50 | 194.4 | 6.25% |
| `cnn2_lwt_2688` | 107,998 | 2,688 | 40.2x | 72.57 | 59.1 | 93.35% |

## 2. Deepfake Detection (Celeb-DF/FF++ video frame targets)

| Method | Total Params | Trainable Params | Reduction | Peak Mem (MB) | Epoch Time (ms) | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| `cnn1_direct` | 536,723 | 536,723 | 1.0x | 0.50 | 7.4 | 25.00% |
| `cnn1_slvt_1024` | 536,723 | 1,024 | 524.1x | 110.00 | 319.8 | 83.92% |
| `cnn1_slvt_2048` | 536,723 | 2,048 | 262.1x | 110.00 | 594.6 | 88.88% |
| `cnn1_lwt_1956` | 536,723 | 1,955 | 274.5x | 110.00 | 66.0 | 88.78% |
| `cnn1_lwt_2792` | 536,723 | 2,790 | 192.4x | 110.00 | 74.9 | 89.98% |
| `cnn2_direct` | 104,195 | 104,195 | 1.0x | 0.50 | 6.2 | 50.00% |
| `cnn2_slvt_1024` | 104,195 | 1,024 | 101.8x | 110.00 | 98.3 | 78.83% |
| `cnn2_slvt_2048` | 104,195 | 2,048 | 50.9x | 110.00 | 151.7 | 85.90% |
| `cnn2_lwt_1872` | 104,195 | 1,872 | 55.7x | 0.50 | 195.1 | 50.00% |
| `cnn2_lwt_2688` | 104,195 | 2,688 | 38.8x | 70.02 | 53.8 | 86.09% |

## 3. Image Segmentation (U-Net/CNN3 target on Cityscapes-like data)

| Method | Total Params | Trainable Params | Reduction | Peak Mem (MB) | Epoch Time (ms) | Pixel Acc | mIoU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cnn3_direct` | 1,925,667 | 1,925,667 | 1.0x | 0.50 | 19.8 | 33.97% | 0.1371 |
| `cnn3_slvt_8192` | 1,925,667 | 8,192 | 235.1x | 140.00 | 7947.5 | 97.92% | 0.4623 |
| `cnn3_lwt_9126` | 1,925,667 | 9,126 | 211.0x | 140.00 | 168.5 | 97.56% | 0.4823 |

## 4. LSTM Time-Series Forecasting (Air Pollution target)

| Method | Total Params | Trainable Params | Reduction | Peak Mem (MB) | Epoch Time (ms) | Val Loss (MSE) |
| --- | --- | --- | --- | --- | --- | --- |
| `lstm_direct` | 12,451 | 12,451 | 1.0x | 0.50 | 6.0 | 1.29665 |
| `lstm_slvt_64` | 12,451 | 64 | 194.5x | 0.50 | 20.9 | 0.00190 |
| `lstm_slvt_2048` | 12,451 | 2,048 | 6.1x | 0.50 | 77.1 | 0.00061 |

## 5. Fine-Tuning (ResNet target)

| Method | Total Params | Trainable Params | Reduction | Peak Mem (MB) | Epoch Time (ms) | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| `resnet_direct` | 25,557,032 | 25,557,032 | 1.0x | 0.50 | 8.8 | 95.23% |
| `resnet_mapping_1024` | 25,557,032 | 1,024 | 24958.0x | 150.00 | 95.9 | 92.10% |
| `resnet_mapping_2048` | 25,557,032 | 2,048 | 12479.0x | 150.00 | 116.9 | 95.10% |

## 6. Ablation of Mapping Loss Components

| Loss Configuration | Validation Accuracy |
| --- | --- |
| Task Loss | 88.00% |
| + Stab | 89.50% |
| + Smooth | 89.50% |
| + Align | 89.00% |
| + Sm + Ali | 90.50% |
| + Stab + Sm | 91.00% |
| Full Mapping Loss | 92.00% |

## 7. Impact of LRD and Pruning Add-ons

| Configuration | Total Params | Trainable Params | Peak Mem (MB) | Epoch Time (ms) | Accuracy |
| --- | --- | --- | --- | --- | --- |
| CNN2 Baseline | 107,998 | 107,998 | 0.50 | 6.6 | 90.40% |
| CNN2 + LRD | 107,998 | 107,998 | 0.50 | 6.4 | 90.40% |
| CNN2 + Prune 90% | 107,998 | 10,799 | 0.50 | 6.1 | 87.91% |
| Ours* (SLVT) | 107,998 | 2,048 | 0.50 | 750.5 | 91.88% |
| Ours* + LRD (rank 8) | 107,998 | 8,192 | 0.50 | 172.4 | 90.67% |
| Ours* + Prune 90% | 107,998 | 204 | 0.50 | 709.0 | 88.70% |

