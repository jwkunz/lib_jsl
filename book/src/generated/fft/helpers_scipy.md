| computation (SciPy reference) | largest difference (relative) |
|---|---:|
| `fftfreq` and `rfftfreq`, n = 7 | 0 |
| `fftfreq` and `rfftfreq`, n = 8 | 0 |
| `next_fast_len` (real) of 7, 13, 97, …, 999983: 8, 15, 100, 1000, 1024, 4320, 65610, 1000000 | 0 |
| `next_fast_len` (complex) of 7, 13, 97, …, 999983: 7, 14, 98, 1000, 1024, 4116, 65610, 1000000 | 0 |
| `rfftn`, 4 × 5 × 6 (round trip through `irfftn`: 1e-15) | 5e-16 |
| `rfftn`, 3 × 4 × 7 (round trip through `irfftn`: 2e-15) | 5e-16 |
| `dctn`, types I–IV, both normalisations, 5 × 6 | 5e-16 |
| `dstn`, types I–IV, both normalisations, 5 × 6 | 7e-16 |
