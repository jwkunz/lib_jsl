| computation (SciPy reference) | largest difference (relative to the largest value) |
|---|---:|
| `group_delay`, Chebyshev I, order 6 | 2e-8 |
| `group_delay`, linear-phase FIR, 41 taps | 2e-10 |
| `minimum_phase`, homomorphic, half (31 → 16 taps) | 1e-14 |
| `minimum_phase`, homomorphic, full length | 2e-14 |
| `firls`, lowpass, 41 taps | 2e-16 |
| `firls`, weighted bandpass, 63 taps | 7e-14 |
| `firls`, differentiator (sloped band), 31 taps | 4e-16 |
| `decimate` by 2, zero-phase IIR | 3e-15 |
| `decimate` by 5, zero-phase IIR | 1e-14 |
| `decimate` by 3, causal IIR | 2e-15 |
| `resample`, 400 → 150 samples | 9e-16 |
| `resample`, 400 → 401 samples | 2e-15 |
| `resample`, 400 → 1000 samples | 1e-15 |
| `medfilt`, kernel 3, signal with spikes | 0 |
| `medfilt`, kernel 7, signal with spikes | 0 |
| `envelope` (`abs(hilbert(x))`) | 1e-15 |
| `csd`, Hann window of 64, overlap 32 | 9e-17 |
| `coherence` | 7e-16 |
