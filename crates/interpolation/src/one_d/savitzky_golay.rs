//! Savitzky-Golay smoothing and differentiation by local polynomial least squares.
//!
//! See the book entry [Savitzky-Golay smoothing](https://jwkunz.github.io/lib_jsl/book/interpolation/savitzky-golay.html).

use lib_jsl_core::*;
use ndarray::Array2;
use lib_jsl_linalg::SVD;

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
    let m = window / 2;
    // ANCHOR: savitzky_golay
    // Least-squares polynomial in the scaled offset t = (i − m)/m ∈ [−1, 1]; the filter is row
    // `deriv` of the pseudo-inverse, times deriv! / (m Δ)^deriv to undo the scaling.
    let pinv = local_fit(window, order, m as f64, m.max(1) as f64)?;
    let factorial: f64 = (1..=deriv).map(|k| k as f64).product();
    let scale = factorial / (m.max(1) as f64 * delta).powi(deriv as i32);
    Ok(pinv.row(deriv).iter().map(|&c| c * scale).collect())
    // ANCHOR_END: savitzky_golay
}

/// The `(order + 1) × window` pseudo-inverse of the Vandermonde matrix `Aᵢⱼ = tᵢʲ` in the scaled
/// variable `tᵢ = (i − centre) / width`, computed from an SVD: scaling keeps the Vandermonde
/// matrix well conditioned, and the SVD avoids squaring its condition number as the normal
/// equations would.
fn local_fit(window: usize, order: usize, centre: f64, width: f64) -> Result<Array2<f64>, ErrorsJSL> {
    let a = Array2::from_shape_fn((window, order + 1), |(i, j)| ((i as f64 - centre) / width).powi(j as i32));
    let (u, sigma, vt) = a.svd(true, true).map_err(|_| ErrorsJSL::RuntimeError("SVD failed"))?;
    let (u, vt) = (u.ok_or(ErrorsJSL::RuntimeError("SVD returned no U"))?, vt.ok_or(ErrorsJSL::RuntimeError("SVD returned no V"))?);
    if sigma.iter().any(|&s| s <= sigma[0] * 1e-13) {
        return Err(ErrorsJSL::RuntimeError("the local polynomial fit is numerically singular"));
    }
    // A⁺ = V Σ⁻¹ Uᵀ restricted to the order + 1 nonzero singular values.
    Ok(Array2::from_shape_fn((order + 1, window), |(j, k)| (0..=order).map(|i| vt[[i, j]] * u[[k, i]] / sigma[i]).sum()))
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
    // Edges: fit one polynomial (in the centred variable t = (i − start − h)/h ∈ [−1, 1], h half the
    // window) to the first (last) window and evaluate it or its derivative there.
    let h = (m.max(1)) as f64;
    let pinv = local_fit(window, order, m as f64, h)?;
    let fit_edge = |start: usize, out: &mut R1D, range: std::ops::Range<usize>| {
        let coef = pinv.dot(&y.slice(ndarray::s![start..start + window]));
        for i in range {
            let t = (i as f64 - start as f64 - m as f64) / h;
            // d-th derivative of Σ coef[j] t^j, with dt/dx = 1/(h Δ).
            let mut v = 0.0;
            for j in deriv..=order {
                let falling: f64 = (0..deriv).map(|q| (j - q) as f64).product();
                v += coef[j] * falling * t.powi((j - deriv) as i32);
            }
            out[i] = v / (h * delta).powi(deriv as i32);
        }
    };
    fit_edge(0, &mut out, 0..m);
    fit_edge(n - window, &mut out, n - m..n);
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

    #[test]
    fn wide_windows_and_high_orders_stay_exact() {
        // The old normal-equations construction lost 9 digits at window 51, order 8, and failed
        // as singular at window 101.
        for (window, order) in [(11, 4), (51, 8), (101, 12)] {
            let n = 300;
            let p = |t: f64| (0..=order).map(|k| (t - 0.4).powi(k as i32)).sum::<f64>();
            let y = R1D::from_vec((0..n).map(|i| p(i as f64 / n as f64)).collect());
            let s = savitzky_golay(y.view(), window, order, 0, 1.0).unwrap();
            let err = s.iter().zip(y.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(err < 1e-9, "window {window}, order {order}: {err:e}");
            // The first derivative of the same polynomial, with spacing 1/n.
            let d = savitzky_golay(y.view(), window, order, 1, 1.0 / n as f64).unwrap();
            let dp = |t: f64| (1..=order).map(|k| k as f64 * (t - 0.4).powi(k as i32 - 1)).sum::<f64>();
            let derr = d.iter().enumerate().map(|(i, v)| (v - dp(i as f64 / n as f64)).abs()).fold(0.0, f64::max);
            assert!(derr < 1e-6, "derivative, window {window}, order {order}: {derr:e}");
        }
    }
}
