| one million 64-bit keys | random | nearly sorted | sorted | reversed | three distinct keys |
|---|---:|---:|---:|---:|---:|
| quicksort | 127.2 ms | 25.6 ms | 22.8 ms | 24.7 ms | 60.2 ms |
| heapsort | 259.2 ms | 117.4 ms | 114.2 ms | 121.7 ms | 82.3 ms |
| Shell's method | 232.1 ms | 171.2 ms | 34.5 ms | 55.4 ms | 49.9 ms |
| std `sort_unstable` | 39.0 ms | 38.2 ms | 1.1 ms | 1.9 ms | 5.0 ms |
| std `sort` (stable) | 57.1 ms | 46.5 ms | 1.3 ms | 2.0 ms | 7.6 ms |
