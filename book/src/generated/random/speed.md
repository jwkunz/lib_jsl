| operation | time per sample |
|---|---:|
| xoshiro256++ `next_u64` | 1.2 ns |
| xoshiro256++ `next_f64` | 1.4 ns |
| SplitMix64 `next_u64` | 2.4 ns |
| exponential (inversion) | 6.2 ns |
| normal (Box-Muller, one pair per two samples) | 15.3 ns |
| gamma, shape 2.5 (Marsaglia-Tsang) | 41.2 ns |
| Poisson, $\lambda = 4$ | 28.3 ns |
