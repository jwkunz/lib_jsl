use std::f64::consts::PI;

use super::{extend, len_guard, linspace, truncate};

/// The Bohman window is defined by a specific mathematical formula that combines a linear tapering with a cosine modulation. The coefficients are computed based on the position within the window and the total length, with special handling for the edges to ensure proper symmetry and smoothness.
pub fn bohman(m: usize, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let x = linspace(-1.0, 1.0, m2);
    let mut w = vec![0.0; m2];
    for i in 1..(m2 - 1) {
        let fac = x[i].abs();
        w[i] = (1.0 - fac) * (PI * fac).cos() + (PI * fac).sin() / PI;
    }
    truncate(w, needs_trunc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn test_bohman() {
        assert_window(include_str!("../test_data/bohman.csv"), bohman(7, true));
    }
}
