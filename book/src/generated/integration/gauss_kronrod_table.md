| integrand on $[0, 1]$ | evaluations | true error | estimated error | adaptive trapezoid evaluations |
|---|---:|---:|---:|---:|
| $e^x$ | 15 | 2.2e-16 | 1.9e-14 | 95609 |
| $1/(10^{-4} + (x - 0.3)^2)$ | 585 | 0.0e0 | 2.4e-11 | 36217969 |
| $\sqrt{\lvert x - 1/\pi \rvert}$ | 855 | 3.2e-13 | 5.7e-11 | 241255 |
| $\ln x$ | 1005 | 2.0e-13 | 8.6e-11 | fails (f(0) is infinite) |
| $x^{-0.9}$ | 10755 | 8.2e-11 | 9.4e-11 | fails (f(0) is infinite) |
