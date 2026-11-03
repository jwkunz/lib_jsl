| method | clusters | noise points | ARI (noise as its own group) |
|---|---:|---:|---:|
| DBSCAN, ε = 0.2 | 2 | 97 | 0.793 |
| DBSCAN, ε = 0.3 | 3 | 79 | 0.921 |
| DBSCAN, ε = 0.5 | 6 | 44 | 0.829 |
| DBSCAN, ε = 0.7 | 3 | 27 | 0.883 |
| DBSCAN, ε = 1 | 2 | 17 | 0.485 |
| DBSCAN, ε = 1.5 | 2 | 3 | 0.551 |
| HDBSCAN, min_cluster_size = 5 | 3 | 3 | 0.982 |
| HDBSCAN, min_cluster_size = 15 | 3 | 9 | 0.949 |
| HDBSCAN, min_cluster_size = 40 | 3 | 22 | 0.886 |
