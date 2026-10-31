use super::{extend, len_guard, linspace, truncate};

/// The Parzen window is a piecewise cubic B-spline taper: a cubic polynomial near the center and a cubed linear falloff toward the edges, with strictly non-negative coefficients.
pub fn parzen(m: usize, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let n = linspace(-(m2 as f64 - 1.0) / 2.0, (m2 as f64 - 1.0) / 2.0, m2);
    let mut w = vec![0.0; m2];
    for (i, ni) in n.iter().enumerate() {
        let an = ni.abs();
        let h = m2 as f64 / 2.0;
        w[i] = if an <= (m2 as f64 - 1.0) / 4.0 {
            1.0 - 6.0 * (an / h).powi(2) + 6.0 * (an / h).powi(3)
        } else {
            2.0 * (1.0 - an / h).powi(3)
        };
    }
    truncate(w, needs_trunc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn test_parzen() {
        assert_window(include_str!("../test_data/parzen.csv"), parzen(7, true));
    }
}
