| System and operation | Difference from SciPy (relative where scaled) |
|---|---:|
| Butterworth, order 6, low-pass: `freqresp`; `bode` (dB, degrees); `step`, `impulse` | 4e-16; 3e-14 / 1e-13; 1e-14 / 2e-14 |
| … `to_discrete(0.1, zoh)`: poles; zeros; gain; `dstep`, `dimpulse`; `dfreqresp` | 3e-11; 1e-6; 2e-8; 6e-11 / 6e-11; 4e-11 |
| … `to_discrete(0.1, bilinear)`: poles; zeros; gain; `dstep`, `dimpulse`; `dfreqresp` | 3e-12; 3e-2; 0; 9e-11 / 7e-11; 8e-11 (zeros exactly at -1: library off by 4e-2, SciPy by 6e-2) |
| … `to_discrete(0.1, foh)`: poles; zeros; gain; `dstep`, `dimpulse`; `dfreqresp` | 7e-12; 2e-6; 2e-8; 9e-11 / 6e-11; 5e-11 |
| Chebyshev I, order 5, high-pass: `freqresp`; `bode` (dB, degrees); `step`, `impulse` | 6e-16; 3e-14 / 6e-14; 1e-14 / 3e-15 |
| … `to_discrete(0.1, zoh)`: poles; zeros; gain; `dstep`, `dimpulse`; `dfreqresp` | 3e-14; 2e-13; 0; 4e-13 / 4e-14; 7e-13 |
| … `to_discrete(0.1, bilinear)`: poles; zeros; gain; `dstep`, `dimpulse`; `dfreqresp` | 3e-14; 7e-4; 0; 1e-12 / 1e-13; 6e-13 (zeros exactly at 1: library off by 1e-3, SciPy by 1e-3) |
| … `to_discrete(0.1, foh)`: poles; zeros; gain; `dstep`, `dimpulse`; `dfreqresp` | 3e-14; 2e-6; 2e-15; 1e-12 / 2e-13; 1e-12 |
