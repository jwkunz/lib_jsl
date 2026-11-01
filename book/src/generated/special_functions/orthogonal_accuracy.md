| function | test points | median relative error | largest relative error | at arguments |
|---|---:|---:|---:|---|
| `legendre` $P_n$ | 61 | 0 | 8e-15 | (30, -0.7) |
| `chebyshev_t` $T_n$ | 61 | 0 | 2e-13 | (100, 0.99) |
| `chebyshev_u` $U_n$ | 61 | 0 | 1e-15 | (100, 3) |
| `hermite` $H_n$ | 61 | 0 | 6e-15 | (30, -0.2) |
| `hermite_he` $He_n$ | 59 | 0 | 7e-16 | (100, -0.2) |
| `laguerre` $L_n^{(1/2)}$ | 62 | 1e-16 | 5e-14 | (30, 0.5, 0.3) |
| `jacobi` $P_n^{(1/2,-0.3)}$ | 63 | 6e-16 | 2e-13 | (100, 0.5, -0.3, 1) |
| `associated_legendre` $P_l^m$ | 36 | 2e-16 | 7e-15 | (10, 3, -0.3) |
| `spherical_harmonic`, real part | 15 | 4e-16 | 2e-15 | (20, -7, 2.9, -1) |
| `spherical_harmonic`, imaginary part | 15 | 4e-16 | 2e-15 | (20, -7, 2.9, -1) |
| `real_spherical_harmonic` | 15 | 4e-16 | 2e-15 | (20, -7, 2.9, -1) |
