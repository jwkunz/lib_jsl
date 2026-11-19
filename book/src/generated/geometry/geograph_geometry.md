| Function | Difference from SciPy | Notes |
|---|---:|---:|
| `align_vectors`, 12 noisy pairs | 0 | rssd 0 |
| `align_vectors`, weighted | 0 | rssd 7e-14 |
| `align_vectors`, one infinite weight | 0 | rssd 2e-16 |
| `align_vectors`, a single pair | 0 | rssd 1e-17 |
| `rotation_mean`, 10 rotations, plain and weighted | 0 |  |
| `from_rotvec`, `as_rotvec`, `as_mrp`, `from_mrp` (angles 3·10⁻⁶ to π − 10⁻⁹) | 0 | components 2e-16 |
| `procrustes`, 15 points in 3-D | 3e-16 | disparity 0.001887 (7e-19) |
| `geometric_slerp`, 4-D, 7 points | 2e-16 |  |
| `directed_hausdorff`, 200 and 150 points | 0 | same pair of points |
| `query_pairs` (268 pairs), `count_neighbors` ([80, 615, 4157]) | identical |  |
| Periodic box: 4 nearest of 20 queries; pairs | 0 | same neighbours; 295 pairs (same) |
