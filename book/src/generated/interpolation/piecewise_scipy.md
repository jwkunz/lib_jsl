| computation (SciPy reference) | largest difference (relative to the largest value) |
|---|---:|
| `PPoly.from_spline`: values, first and second derivatives (with extrapolation) | 4e-15 |
| `integrate` on [0.3, 3.7] and [−0.5, 4.5]; second antiderivative | 8e-16 |
| `roots` (4, two by extrapolation) and `solve(0.5)` | 4e-15 |
| `CubicHermiteSpline`: values and slopes | 1e-15 |
| `CubicHermiteSpline`: integral, roots and `solve(0.25)` | 4e-16 |
| `Akima1DInterpolator`, Akima's data and data with flat stretches | 1e-15 |
| `Akima1DInterpolator(method="makima")`, the same data | 1e-15 |
| `KroghInterpolator` with derivative data: values and derivatives 0–3 | 6e-15 |
| `RectBivariateSpline`, degrees (3, 3): values, ∂x and ∂x∂y, integral | 2e-15 |
| `RectBivariateSpline`, degrees (1, 1): values, integral | 5e-16 |
| `RectBivariateSpline`, degrees (5, 3): values, ∂x and ∂x∂y, integral | 4e-15 |
| `LinearNDInterpolator`, 64 scattered points, 169 queries | 3e-16 |
| `estimate_gradients_2d_global` (Nielson's method, tolerance 10⁻⁶) | 8e-16 |
| `CloughTocher2DInterpolator` | 7e-16 |
| `NearestNDInterpolator` | 0 |
| `griddata` outside the hull: NaN and NaN (SciPy: NaN) | 0 |
| `RBFInterpolator` in 3-D: 8 kernels, smoothing, degree 2 (largest: quintic) | 1e-14 |
