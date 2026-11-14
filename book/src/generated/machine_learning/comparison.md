| classifier | three Gaussian classes, 4D | moons | circles | fit time (moons) |
|---|---:|---:|---:|---:|
| logistic regression | 0.910 | 0.870 | 0.407 | 134 µs |
| Gaussian naive Bayes | 0.907 | 0.883 | 0.997 | 3 µs |
| k-nearest neighbours, k = 15 | 0.907 | 0.950 | 0.997 | 0 µs |
| decision tree, depth 6 | 0.827 | 0.937 | 0.983 | 99 µs |
| random forest, 100 trees | 0.900 | 0.937 | 0.987 | 7.8 ms |
| SVM, RBF kernel γ = 1, C = 1 | 0.893 | 0.950 | 1.000 | 449 µs |

Test accuracy with half of each data set for training (standardised features); the fit time is for 300 points in 2D.
