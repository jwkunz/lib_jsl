| Transform | Largest difference from SciPy (relative to the largest value) |
|---|---:|
| `hfft`, 9 inputs; n = 16, 13, 6 | 3e-15 |
| `ihfft`, 11 inputs | 8e-17 |
| `fht`, `ifht`, `fhtoffset`: μ = 0, r e^{−r²/2}, n = 128 | 1e-15 / 1e-15 / 0 |
| `fht`, `ifht`, `fhtoffset`: μ = 1, r² e^{−r}, n = 128 | 8e-16 / 9e-16 / 7e-16 |
| `fht`, `ifht`, `fhtoffset`: μ = ½, bias 0.4, log-normal, n = 128 | 2e-14 / 2e-14 / 4e-16 |
| `fht`, `ifht`, `fhtoffset`: μ = −½, bias −0.3, 1/(1 + r²), n = 128 | 1e-14 / 1e-14 / 3e-16 |
