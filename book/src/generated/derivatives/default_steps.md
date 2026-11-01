| formula | truncation error | rounding error | default relative step | error for $e^x$ at $x = 1$ |
|---|---:|---:|---:|---:|
| central difference, $f'$ | $h^2$ | $\varepsilon/h$ | $\varepsilon^{1/3} \approx 6.1e-6$ | 2.3e-11 |
| five-point, $f'$ | $h^4$ | $\varepsilon/h$ | $\varepsilon^{1/5} \approx 7.4e-4$ | 8.9e-14 |
| five-point, $f''$ | $h^4$ | $\varepsilon/h^2$ | $\varepsilon^{1/6} \approx 2.5e-3$ | 1.8e-10 |
| five-point, $f'''$ | $h^2$ | $\varepsilon/h^3$ | $\varepsilon^{1/5} \approx 7.4e-4$ | 4.3e-7 |
| five-point, $f''''$ | $h^2$ | $\varepsilon/h^4$ | $\varepsilon^{1/6} \approx 2.5e-3$ | 2.5e-5 |
