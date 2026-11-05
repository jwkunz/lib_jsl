| integral | library Gauss-Kronrod | library tanh-sinh | SciPy `quad` | SciPy `tanhsinh` |
|---|---:|---:|---:|---:|
| exp(-x^2) on [0, 1] | 6e-16 | 6e-16 | 4e-16 | 3e-16 |
| 1/(1 + 25 x^2) on [-1, 1] | 0 | 4e-16 | 0 | 4e-16 |
| sqrt(x) ln(x) on [0, 1] | 2e-16 | 0 | 7e-16 | 2e-16 |
| 1/sqrt(1 - x^2) on [-1, 1] | 1e-8 | 7e-9 | 6e-14 | 7e-9 |
| cos(30 x) exp(x) on [0, 2] | 4e-15 | 2e-15 | 5e-15 | 3e-14 |
| 1/sqrt(1 - x^2), library tanh-sinh with the endpoint offset |  | 0 |  |  |
