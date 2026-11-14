| operation | time per sample |
|---|---:|
| xoshiro256++ `next_u64` | 1.0 ns |
| xoshiro256++ `next_f64` | 1.2 ns |
| SplitMix64 `next_u64` | 1.9 ns |
| exponential (inversion) | 5.1 ns |
| normal (Box-Muller, one pair per two samples) | 12.8 ns |
| gamma, shape 2.5 (Marsaglia-Tsang) | 34.5 ns |
| Poisson, $\lambda = 4$ | 23.5 ns |
