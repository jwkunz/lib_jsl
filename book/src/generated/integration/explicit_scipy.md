| method and tolerance (atol = rtol/1000) | steps, 4 problems (library / SciPy) | function evaluations | problems with identical steps and evaluations | final state (relative) | dense output (relative) |
|---|---:|---:|---:|---:|---:|
| RK45, rtol = 10⁻³ | 169 / 169 | 1256 / 1256 | 4 of 4 | 1e-9 | 1e-11 |
| RK45, rtol = 10⁻⁶ | 559 / 559 | 3956 / 3956 | 4 of 4 | 2e-12 | 1e-13 |
| RK45, rtol = 10⁻¹⁰ | 3382 / 3382 | 20828 / 20828 | 4 of 4 | 2e-13 | 7e-14 |
| DOP853, rtol = 10⁻³ | 114 / 114 | 2066 / 2066 | 4 of 4 | 2e-9 | 2e-9 |
| DOP853, rtol = 10⁻⁶ | 212 / 212 | 3872 / 3872 | 4 of 4 | 2e-12 | 1e-11 |
| DOP853, rtol = 10⁻¹⁰ | 598 / 597 | 10490 / 10451 | 3 of 4 | 1e-10 | 2e-10 |
