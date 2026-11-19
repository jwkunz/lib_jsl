| Function | Difference from SciPy | Notes |
|---|---:|---:|
| `johnson`, 40 vertices, 155 edges (42 negative) | 9e-16 | all 1600 distances |
| `yen`, 8 shortest paths 0 → 17, directed | 9e-16 | lengths 4.690 … 7.323 |
| `yen`, 8 shortest paths 0 → 17, undirected | 4e-16 | lengths 2.111 … 3.102 |
| `laplacian` (in-degrees) | 0 |  |
| `laplacian`, normalised | 0 |  |
| `laplacian`, out-degrees | 4e-15 |  |
| `maximum_bipartite_matching`, `structural_rank`, 30 × 40 | same size | 29 pairs; structural rank 29 |
| `min_weight_full_bipartite_matching`, 25 × 35 | 7e-15 | total 59.0774; same assignment |
| `min_weight_full_bipartite_matching`, 25 × 35, maximize | 0 | total 225.7493; same assignment |
