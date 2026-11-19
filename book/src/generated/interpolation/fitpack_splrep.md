| Case (80 points) | Knots | Knots vs `make_splrep` | Coefficients | fp: library / SciPy | FITPACK's `splrep` |
|---|---:|---:|---:|---:|---:|
| `cubic s=0.5` | 61 | identical | 2e-15 | 0.5000 / 0.5000 | 60 knots |
| `cubic s=2` | 18 | identical | 1e-15 | 1.9996 / 1.9996 | 21 knots |
| `cubic s=8` | 10 | identical | 1e-15 | 8.0000 / 8.0000 | same knots; 3e-16 |
| `linear s=1` | 35 | identical | 1e-15 | 1.0001 / 1.0001 | 43 knots |
| `quadratic s=1` | 51 | identical | 3e-15 | 1.0005 / 1.0005 | 48 knots |
| `quartic s=1` | 49 | identical | 4e-15 | 1.0009 / 1.0009 | 49 knots |
| `quintic s=1` | 58 | identical | 1e-14 | 1.0009 / 1.0009 | 58 knots |
| `weighted s=80` | 34 | identical | 2e-15 | 80.0082 / 80.0082 | 36 knots |
| `interval [-1, 11]` | 18 | identical | 6e-15 | 2.0000 / 2.0000 | 21 knots |
| `nest=14` | 14 | identical | 2e-15 | 7.8403 / 7.8403 | (no `nest` in `splrep`) |
| `s huge` | 8 | identical | 2e-15 | 35.7924 / 35.7924 | same knots; 0 |
| `make_splprep`, 3-D curve, s=0.05 | 18 | identical | 8e-16 (u: 0) | 0.0500 | same knots |
| `generate_knots`, s=0.5 | 8, 9, 10, 12, 16, 19, 25, 37, 61 | identical |  |  |  |
