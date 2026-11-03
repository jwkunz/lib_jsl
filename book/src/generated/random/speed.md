| operation | time per sample |
|---|---:|
| xoshiro256++ `next_u64` | 2.2 ns |
| xoshiro256++ `next_f64` | 3.2 ns |
| SplitMix64 `next_u64` | 9.6 ns |
| exponential (inversion) | 22.6 ns |
| normal (Box-Muller, one pair per two samples) | 43.3 ns |
| gamma, shape 2.5 (Marsaglia-Tsang) | 133.6 ns |
| Poisson, $\lambda = 4$ | 69.0 ns |
