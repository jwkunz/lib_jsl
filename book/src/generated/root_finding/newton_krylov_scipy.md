| grid | inner solver | Newton iterations (library / SciPy) | evaluations of F (library / SciPy) | final max \|F\| (library / SciPy) | peak of u, relative difference from SciPy's LGMRES solution |
|---|---:|---:|---:|---:|---:|
| 16 × 16 | GMRES(20) | 4 / 4 | 81 / 81 | 5e-7 / 5e-7 | 7e-11 |
| 16 × 16 | BiCGSTAB | 4 / 4 | 108 / 108 | 4e-8 / 8e-8 | 1e-10 |
| 16 × 16 | GMRES(20), Laplacian LU preconditioner | 4 / 4 | 42 / 20 | 8e-9 / 9e-9 | 5e-12 |
| 16 × 16 | LGMRES (SciPy's default) | 4 / 4 | 79 / 79 | 9e-9 / 8e-9 | 2e-11 |
| 32 × 32 | GMRES(20) | 20 / 20 | 441 / 441 | 2e-6 / 2e-6 | 2e-7 |
| 32 × 32 | BiCGSTAB | 5 / 4 | 195 / 160 | 4e-7 / 6e-6 | 3e-8 |
| 32 × 32 | GMRES(20), Laplacian LU preconditioner | 4 / 4 | 42 / 19 | 3e-8 / 6e-8 | 6e-10 |
| 32 × 32 | LGMRES (SciPy's default) | 4 / 4 | 114 / 114 | 1e-7 / 1e-7 | 6e-10 |
| 64 × 64 | GMRES(20) | 69 / 69 | 1519 / 1519 | 6e-6 / 6e-6 | 7e-7 |
| 64 × 64 | BiCGSTAB | 14 / 18 | 512 / 604 | 2e-6 / 3e-6 | 2e-7 |
| 64 × 64 | GMRES(20), Laplacian LU preconditioner | 4 / 4 | 41 / 19 | 3e-7 / 3e-7 | 3e-9 |
| 64 × 64 | LGMRES (SciPy's default) | 6 / 6 | 187 / 187 | 2e-6 / 1e-6 | 3e-9 |
