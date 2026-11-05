| interpolant (SciPy reference) | largest difference |
|---|---:|
| natural cubic spline, a smooth curve (`CubicSpline`, natural) | 2e-15 |
| PCHIP, a smooth curve (`PchipInterpolator`) | 2e-15 |
| Akima, a smooth curve (`Akima1DInterpolator`) | 2e-15 |
| linear, a smooth curve (`numpy.interp`) | 2e-15 |
| natural cubic spline, data with flat stretches (`CubicSpline`, natural) | 5e-15 |
| PCHIP, data with flat stretches (`PchipInterpolator`) | 5e-15 |
| Akima, data with flat stretches (`Akima1DInterpolator`) | 6e-15 |
| linear, data with flat stretches (`numpy.interp`) | 5e-15 |
| polynomial through 9 Chebyshev points (`BarycentricInterpolator`) | 3e-15 |
| Floater-Hormann, 21 equispaced points, d = 3 (`FloaterHormannInterpolator`) | 8e-16 |
| AAA rational approximation of tan(pi x / 2.2) (`AAA`) | 1e-15 (relative); each within 4e-16 of the function |
| [4/4] Pade approximant of exp, coefficients (`pade`) | numerator 2e-15, denominator 2e-15 |
| Chebyshev coefficients of exp, 12 terms (`numpy.polynomial.chebyshev.chebinterpolate`) | 1e-15 |
| Savitzky-Golay coefficients, 11 points, cubic, derivative 0 (`savgol_coeffs`) | 8e-16 |
| Savitzky-Golay filter, derivative 0, with end fitting (`savgol_filter`, interp) | 4e-15 |
| Savitzky-Golay coefficients, 11 points, cubic, derivative 1 (`savgol_coeffs`) | 1e-15 |
| Savitzky-Golay filter, derivative 1, with end fitting (`savgol_filter`, interp) | 4e-15 |
| trigonometric interpolation, 15 samples (`signal.resample`) | 2e-15 |
| trigonometric interpolation, 16 samples (`signal.resample`) | 2e-15 |
| bilinear on an uneven grid (`RegularGridInterpolator`, linear) | 2e-16 |
| nearest neighbour on an uneven grid (`RegularGridInterpolator`, nearest) | 0 |
| trilinear (`RegularGridInterpolator`, linear) | 3e-16 |
| n-dimensional multilinear, in 3D (`RegularGridInterpolator`, linear) | 3e-16 |
| tricubic natural B-spline (`CubicSpline`, natural, along each axis) | 9e-16 |
| radial basis functions, thin plate spline kernel, 30 scattered points (`RBFInterpolator`) | 5e-15 |
| radial basis functions, cubic kernel, 30 scattered points (`RBFInterpolator`) | 8e-14 |
| radial basis functions, gaussian kernel, 30 scattered points (`RBFInterpolator`) | 7e-9 |
| radial basis functions, multiquadric kernel, 30 scattered points (`RBFInterpolator`) | 1e-9 |
| radial basis functions, inverse multiquadric kernel, 30 scattered points (`RBFInterpolator`) | 8e-11 |
