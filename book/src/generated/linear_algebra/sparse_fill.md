| ordering | bandwidth (library / SciPy) | nnz(L) + nnz(U), library | nnz(L) + nnz(U), SciPy's SuperLU |
|---|---:|---:|---:|
| the shuffled matrix, as given | 890 / 890 | 123840 | 123840 |
| reverse Cuthill-McKee | 30 / 30 | 38630 | 38630 |
| COLAMD (SciPy's default for `splu`) | — | — | 29360 |
| minimum degree on AᵀA + A (SciPy's `MMD_AT_PLUS_A`) | — | — | 21000 |
