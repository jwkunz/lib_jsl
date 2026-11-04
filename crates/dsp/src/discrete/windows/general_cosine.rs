//! The general cosine-sum window.
//!
//! See the book entry [Window functions](https://jwkunz.github.io/lib_jsl/book/spectral/windows.html#families).

use std::f64::consts::PI;

use super::{extend, len_guard, linspace, truncate};

/// This function generates a general cosine window based on the specified coefficients `a` and symmetry. It is used as a building block for more specific window functions like the Blackman and Nuttall windows, which can be defined as special cases of the general cosine window with specific coefficient values.
pub fn general_cosine(m: usize, a: &[f64], sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    // ANCHOR: general_cosine
    let fac = linspace(-PI, PI, m2);
    let mut w = vec![0.0; m2];
    for (k, ak) in a.iter().enumerate() {
        for n in 0..m2 {
            w[n] += ak * ((k as f64) * fac[n]).cos();
        }
    }
    truncate(w, needs_trunc)
    // ANCHOR_END: general_cosine
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn test_general_cosine() {
        assert_window(include_str!("../test_data/general_cosine.csv"), general_cosine(7, &[1.0, 1.942604, 1.340318, 0.440811, 0.043097], true));
    }
}
