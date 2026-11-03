| classifier (test accuracy) | moons | circles | three Gaussian classes (4D) | total time |
|---|---:|---:|---:|---:|
| decision tree (depth 6) | 0.926 | 0.964 | 0.852 | 1 ms |
| random forest (200 trees) | 0.936 | 0.977 | 0.888 | 155 ms |
| gradient boosting (200 × depth 3) | 0.929 | 0.958 | 0.899 | 258 ms |
| MLP (32 × 32) | 0.943 | 0.990 | 0.879 | 672 ms |
| SVM (RBF) | 0.939 | 0.992 | 0.918 | 21 ms |
