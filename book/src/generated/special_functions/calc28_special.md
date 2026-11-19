| Function | Values | Largest relative difference from SciPy | Notes |
|---|---:|---:|---:|
| `wofz`, upper half plane | 40 | 9e-15 | Taylor series near the real axis, Weideman (\|z\| < 6), continued fraction beyond |
| `wofz`, real axis | 8 | 2e-16 | from Dawson's integral |
| `wofz`, lower half plane | 16 | 2e-15 | reflection; relative to \|w\|, which grows like e^{y²} |
| `voigt_profile` | 40 | 7e-15 | including the pure Gaussian and Lorentzian |
| `struve_h` (H_ν) | 54 | 1e-13 | best of four methods by error estimate |
| `struve_l` (L_ν) | 54 | 4e-14 | at ν = −½ against √(2/(πx)) sinh x, from which SciPy differs by 8e-11 |
| `kelvin` (all eight), x ∉ [8, 20) | 64 | 1e-13 | trapezoid rules for I₀, I₁, K₀, K₁ at x e^{iπ/4} |
| `kelvin`, x = 8 and 10 | 16 | 5e-10 | SciPy's expansion; the library agrees with the power series to 4e-16 |
| `hankel1`, `hankel2` | 24 | 6e-14 | J ± iY |
| `jn_zeros`, `jnp_zeros`, `yn_zeros`, `ynp_zeros` | 160 | 1e-15 | n = 0, 1, 5, 20; ten zeros each |
