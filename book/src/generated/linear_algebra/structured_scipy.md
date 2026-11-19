| functions | largest relative difference from SciPy |
|---|---:|
| constructors: `toeplitz`, `hankel`, `circulant`, `hadamard`, `block_diag`, `companion`, `pascal`, `invpascal`, `hilbert`, `invhilbert`, `leslie`, `convolution_matrix`, `fiedler` | 0 |
| `dft` (three scalings); Hermitian `toeplitz` | 8e-16 |
| `solve_toeplitz`, n = 40, nonsymmetric | 0 |
| `solve_toeplitz`, n = 40, complex Hermitian | 6e-16 |
| `solve_circulant`, n = 40, real and complex | 4e-15 |
| `solve_banded`, n = 40, l = 3, u = 2 (pivoting) | 9e-13 |
| `solveh_banded`, n = 40, bandwidth 2 | 3e-16 |
