| function | f': central difference | f': five-point stencil | f': SciPy `derivative` | f'': five-point stencil | f'': SciPy, nested |
|---|---:|---:|---:|---:|---:|
| exp(x) at 1 | 8e-12 | 3e-14 | 1e-14 (2e-13 with a first step of 0.01) | 7e-11 | 5e-10 |
| sin(x) at 1 | 4e-12 | 4e-14 | 1e-14 (2e-13 with a first step of 0.01) | 2e-11 | 2e-10 |
| ln(x) at 0.1 | 1e-9 | 2e-9 | NaN (5e-15 with a first step of 0.01) | 5e-7 | 4e-13 |
| 1/(1 + 25 x^2) at 0.3 | 2e-10 | 3e-11 | 8e-11 (9e-15 with a first step of 0.01) | 7e-9 | 1e-12 |
| x^(5/2) at 2 | 5e-12 | 7e-14 | 4e-15 (1e-13 with a first step of 0.01) | 2e-12 | 5e-10 |
