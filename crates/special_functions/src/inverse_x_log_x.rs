//! Inverse of x log(x).
//!
//! Solutions of `x ln x = y` are `x = e^W(y)`, where `W` is the Lambert W function.

const INV_E: f64 = 0.367_879_441_171_442_33;

/// Halley iteration for `w e^w = y` from the starting guess `w`.
fn halley(y: f64, mut w: f64) -> f64 {
    for _ in 0..50 {
        let ew = w.exp();
        let f = w * ew - y;
        let denom = ew * (w + 1.0) - (w + 2.0) * f / (2.0 * w + 2.0);
        if denom == 0.0 || !denom.is_finite() {
            break;
        }
        let dw = f / denom;
        w -= dw;
        if dw.abs() <= 1e-16 * (1.0 + w.abs()) {
            break;
        }
    }
    w
}

/// The principal branch W₀(y) of the Lambert W function, for `y >= -1/e`.
pub fn lambert_w0(y: f64) -> f64 {
    if y.is_nan() || y < -INV_E {
        return f64::NAN;
    }
    if y == f64::INFINITY {
        return f64::INFINITY;
    }
    if y == 0.0 {
        return 0.0;
    }
    let guess = if y < -0.25 {
        // Series about the branch point.
        let p = (2.0 * (std::f64::consts::E * y + 1.0)).max(0.0).sqrt();
        -1.0 + p - p * p / 3.0 + 11.0 / 72.0 * p * p * p
    } else if y < 3.0 {
        let l = (1.0 + y).ln();
        l * (1.0 - l.ln_1p() / (2.0 + l))
    } else {
        let l = y.ln();
        l - l.ln() + l.ln() / l
    };
    halley(y, guess)
}

/// The lower branch W₋₁(y) of the Lambert W function, for `-1/e <= y < 0`.
pub fn lambert_w_m1(y: f64) -> f64 {
    if y.is_nan() || y < -INV_E || y >= 0.0 {
        return if y == 0.0 { f64::NEG_INFINITY } else { f64::NAN };
    }
    let guess = if y < -0.25 {
        let p = (2.0 * (std::f64::consts::E * y + 1.0)).max(0.0).sqrt();
        -1.0 - p - p * p / 3.0 - 11.0 / 72.0 * p * p * p
    } else {
        let l1 = (-y).ln();
        let l2 = (-l1).ln();
        l1 - l2 + l2 / l1
    };
    halley(y, guess)
}

/// Solves `x ln x = y` for the solution `x >= 1/e` (unique for every `y >= -1/e`).
///
/// For `y >= 0` this is the only positive solution `x >= 1`. Returns `NaN` for `y < -1/e`.
/// See [`inverse_x_log_x_lower`] for the second solution when `y < 0`.
pub fn inverse_x_log_x(y: f64) -> f64 {
    if y.is_nan() || y < -INV_E {
        return f64::NAN;
    }
    if y == 0.0 {
        return 1.0;
    }
    if y == -INV_E {
        return INV_E;
    }
    lambert_w0(y).exp()
}

/// Solves `x ln x = y` for the solution `0 < x <= 1/e`, which exists for `-1/e <= y < 0`
/// (tends to 0 as `y → 0⁻`). Returns `NaN` outside that range.
pub fn inverse_x_log_x_lower(y: f64) -> f64 {
    if y.is_nan() || y < -INV_E || y >= 0.0 {
        return f64::NAN;
    }
    if y == -INV_E {
        return INV_E;
    }
    lambert_w_m1(y).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs()
    }

    #[test]
    fn upper_branch_values() {
        assert!(close(inverse_x_log_x(-0.3), 0.6129927150689683, 1e-14));
        assert_eq!(inverse_x_log_x(0.0), 1.0);
        assert!(close(inverse_x_log_x(1.0), 1.7632228343518967, 1e-14));
        assert!(close(inverse_x_log_x(std::f64::consts::E), std::f64::consts::E, 1e-14));
        assert!(close(inverse_x_log_x(10.0), 5.728925565386942, 1e-14));
        assert!(close(inverse_x_log_x(100.0), 29.536599054329338, 1e-14));
        assert!(close(inverse_x_log_x(1e6), 87847.53957775977, 1e-14));
    }

    #[test]
    fn lower_branch_values() {
        assert!(close(inverse_x_log_x_lower(-0.3), 0.1684128247802058, 1e-13));
        assert!(close(inverse_x_log_x_lower(-0.1), 0.027955199614682573, 1e-13));
        assert!(close(inverse_x_log_x_lower(-1e-3), 0.00010967309611437798, 1e-13));
    }

    #[test]
    fn round_trip_and_domain() {
        for &y in &[-0.36, -0.2, -1e-5, 0.5, 3.0, 50.0, 1e10] {
            let x = inverse_x_log_x(y);
            assert!((x * x.ln() - y).abs() <= 1e-12 * y.abs() + 2e-16, "y={y}");
        }
        for &y in &[-0.367, -0.2, -1e-8] {
            let x = inverse_x_log_x_lower(y);
            assert!((x * x.ln() - y).abs() <= 1e-12 * y.abs() + 2e-16, "y={y}");
        }
        assert!(inverse_x_log_x(-0.5).is_nan());
        assert!(inverse_x_log_x_lower(0.1).is_nan());
        assert!(close(inverse_x_log_x(-INV_E), INV_E, 1e-15));
    }
}
