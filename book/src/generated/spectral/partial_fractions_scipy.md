| computation (SciPy reference) | largest difference (relative) |
|---|---:|
| `residue` and `invres`, (s + 2)/(s² + 4s + 3) | 0 |
| `residue` and `invres`, an improper fraction (direct term k) | 0 |
| `residue` and `invres`, a double pole | 4e-15 |
| `residue` and `invres`, a triple pole and a complex pair | 2e-15 |
| `residuez` and `invresz`, second order | 9e-16 |
| `residuez` and `invresz`, a double pole at z = 0.5 | 4e-16 |
| `residuez` and `invresz`, an FIR part (direct term k) | 0 |
| `deconvolve`, a noisy 44-sample convolution by a 5-tap filter: quotient and remainder | 6e-16 |
| `tf2zpk` (zeros and poles matched as sets) | 3e-16 |
| `zpk2tf` back | 4e-16 |
| `ss2zpk` (sets) | 8e-15 |
| Default time grid of `step` (7 time constants of the slowest mode) | 3e-16 |
| `zoom_fft`, 95–110 Hz in 128 points, endpoint false | 1e-14 |
| `zoom_fft`, 95–110 Hz in 128 points, endpoint true | 1e-14 |
| `sawtooth` (widths 1, 0.5, 0.25) and `square` (duty 0.3) | 6e-15 |
| `gausspulse`: in-phase, quadrature, envelope and cutoff time | 4e-15 |
| `chirp`, all four sweeps (and both quadratic vertices) | 1e-13 |
| `sweep_poly`, a cubic frequency law | 7e-15 |
