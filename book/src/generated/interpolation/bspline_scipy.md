| computation (SciPy reference) | values | first derivative | integrals, antiderivative |
|---|---:|---:|---:|
| cubic not-a-knot (`make_interp_spline`) | 7e-15 | 2e-14 | 2e-15 |
| quadratic not-a-knot (`make_interp_spline`) | 6e-16 | 1e-15 | 4e-16 |
| quartic not-a-knot (`make_interp_spline`) | 4e-14 | 1e-13 | 8e-15 |
| quintic not-a-knot (`make_interp_spline`) | 4e-14 | 1e-13 | 4e-15 |
| cubic natural (`make_interp_spline`) | 3e-16 | 2e-15 | 1e-15 |
| cubic clamped (`make_interp_spline`) | 4e-16 | 7e-16 | 7e-16 |
| cubic given slopes (`make_interp_spline`) | 2e-16 | 3e-16 | 2e-16 |
| quintic given derivatives (`make_interp_spline`) | 3e-15 | 1e-14 | 1e-15 |
| linear (`make_interp_spline`) | 6e-17 | 0 | 4e-17 |
| periodic, degree 3 | 9e-16 |  | 1e-15 |
| periodic, degree 4 | 2e-15 |  | 2e-16 |
| weighted least squares, 5 interior knots (`make_lsq_spline`) | 1e-14 |  |  |
| weighted smoothing spline, λ = 0.05 (`make_smoothing_spline`) | 5e-12 |  |  |
| weighted smoothing spline, λ = 2 (`make_smoothing_spline`) | 1e-10 |  |  |
| smoothing spline, λ by GCV (here 0.0856) | 2e-8 |  |  |
| parametric cubic curve (`splprep`, s = 0; parameters differ by 0) | 1e-15 |  |  |
