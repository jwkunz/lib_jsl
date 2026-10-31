//! Savitzky-Golay smoothing and differentiation by local polynomial least squares.

use lib_jsl_core::*;
use ndarray::Array2;
use ndarray_linalg::Inverse;

/// Convolution coefficients of a Savitzky-Golay filter.
///
/// `window` must be odd and greater than `order`. `deriv` selects the derivative (0 = smoothing)
/// and `delta` is the sample spacing (needed for `deriv > 0`). The coefficients are ordered for
/// correlation with the data centred on the middle tap (i.e. `y_smooth[i] = Σ c[k] y[i - m + k]`
/// with `m = window / 2`).
pub fn savitzky_golay_coefficients(window: usize, order: usize, deriv: usize, delta: f64) -> Result<Vec<f64>, ErrorsJSL> {
    if window.is_multiple_of(2) || window <= order {
        return Err(ErrorsJSL::InvalidInputRange("window must be odd and larger than the polynomial order"));
    }
    if deriv > order {
        return Ok(vec![0.0; window]);
    }
    let m = (window / 2) as f64;
    // Vandermonde in the offsets -m..m; the filter is row `deriv` of (AᵀA)⁻¹Aᵀ, scaled by deriv!.
    let a = Array2::from_shape_fn((window, order + 1), |(i, j)| (i as f64 - m).powi(j as i32));
    let ata_inv = a.t().dot(&a).inv().map_err(|_| ErrorsJSL::RuntimeError("singular normal equations"))?;
    let pinv = ata_inv.dot(&a.t());
    let factorial: f64 = (1..=deriv).map(|k| k as f64).product();
    let scale = factorial / delta.powi(deriv as i32);
    Ok(pinv.row(deriv).iter().map(|&c| c * scale).collect())
}

/// Applies a Savitzky-Golay filter to `y`. Near the ends the polynomial fitted to the first/last
/// `window` samples is evaluated (SciPy's `mode="interp"`), so the output has the input's length.
/// Requires `y.len() >= window`.
pub fn savitzky_golay(y: VR1D, window: usize, order: usize, deriv: usize, delta: f64) -> Result<R1D, ErrorsJSL> {
    let n = y.len();
    if n < window {
        return Err(ErrorsJSL::IncompatibleArraySizes((n, window)));
    }
    let c = savitzky_golay_coefficients(window, order, deriv, delta)?;
    let m = window / 2;
    let mut out = R1D::zeros(n);
    for i in m..n - m {
        out[i] = (0..window).map(|k| c[k] * y[i + k - m]).sum();
    }
    // Edges: fit one polynomial to the first (last) window and evaluate it/its derivative there.
    let fit_edge = |start: usize, out: &mut R1D, range: std::ops::Range<usize>| -> Result<(), ErrorsJSL> {
        let a = Array2::from_shape_fn((window, order + 1), |(i, j)| (i as f64).powi(j as i32));
        let ata_inv = a.t().dot(&a).inv().map_err(|_| ErrorsJSL::RuntimeError("singular normal equations"))?;
        let rhs = a.t().dot(&y.slice(ndarray::s![start..start + window]));
        let coef = ata_inv.dot(&rhs);
        for i in range {
            let t = (i - start) as f64;
            // d-th derivative of Σ coef[j] t^j.
            let mut v = 0.0;
            for j in deriv..=order {
                let falling: f64 = (0..deriv).map(|q| (j - q) as f64).product();
                v += coef[j] * falling * t.powi((j - deriv) as i32);
            }
            out[i] = v / delta.powi(deriv as i32);
        }
        Ok(())
    };
    fit_edge(0, &mut out, 0..m)?;
    fit_edge(n - window, &mut out, n - m..n)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classic_quadratic_five_point_coefficients() {
        let c = savitzky_golay_coefficients(5, 2, 0, 1.0).unwrap();
        let want = [-3.0 / 35.0, 12.0 / 35.0, 17.0 / 35.0, 12.0 / 35.0, -3.0 / 35.0];
        for (a, b) in c.iter().zip(want) {
            assert!((a - b).abs() < 1e-12);
        }
        // First-derivative coefficients of the same filter: (-2,-1,0,1,2)/10.
        let d = savitzky_golay_coefficients(5, 2, 1, 1.0).unwrap();
        for (a, b) in d.iter().zip([-0.2, -0.1, 0.0, 0.1, 0.2]) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn preserves_polynomials_and_their_derivatives() {
        let n = 30;
        let dt = 0.1;
        let y = R1D::from_iter((0..n).map(|k| {
            let t = k as f64 * dt;
            1.0 - 2.0 * t + 0.5 * t * t + 0.25 * t * t * t
        }));
        let s = savitzky_golay(y.view(), 7, 3, 0, dt).unwrap();
        let d = savitzky_golay(y.view(), 7, 3, 1, dt).unwrap();
        let d2 = savitzky_golay(y.view(), 7, 3, 2, dt).unwrap();
        for k in 0..n {
            let t = k as f64 * dt;
            assert!((s[k] - y[k]).abs() < 1e-9, "k={k}");
            assert!((d[k] - (-2.0 + t + 0.75 * t * t)).abs() < 1e-8, "d k={k}");
            assert!((d2[k] - (1.0 + 1.5 * t)).abs() < 1e-7, "d2 k={k}");
        }
    }

    #[test]
    fn rejects_even_window() {
        assert!(savitzky_golay_coefficients(4, 2, 0, 1.0).is_err());
    }
}
