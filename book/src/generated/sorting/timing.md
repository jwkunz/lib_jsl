| one million 64-bit keys | random | nearly sorted | sorted | reversed | three distinct keys |
|---|---:|---:|---:|---:|---:|
| quicksort | 63.2 ms | 8.2 ms | 6.5 ms | 6.7 ms | 15.4 ms |
| heapsort | 154.5 ms | 60.4 ms | 56.0 ms | 60.9 ms | 38.7 ms |
| Shell's method | 144.4 ms | 101.9 ms | 15.9 ms | 20.2 ms | 25.4 ms |
| std `sort_unstable` | 15.6 ms | 16.1 ms | 0.5 ms | 0.9 ms | 2.5 ms |
| std `sort` (stable) | 21.1 ms | 16.1 ms | 0.5 ms | 0.8 ms | 4.7 ms |
