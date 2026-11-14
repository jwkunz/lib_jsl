| computation (SciPy reference) | largest difference |
|---|---:|
| `new_standard`: response of a 3-state, 2-input, 2-output system to 60 input samples (`signal.dlsim` with the same (A, B, C, D)) | 2e-15 |
| `new`: the same matrices, output read after the update (`signal.dlsim` with (A, B, CA, CB + D)) | 2e-15 |
| its poles, the eigenvalues of A (`linalg.eigvals`) | 6e-16 |
