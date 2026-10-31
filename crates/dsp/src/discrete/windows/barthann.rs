use std::f64::consts::PI;

use super::{extend, len_guard, truncate};

/// The modified Bartlett-Hann window combines a linear taper with a cosine taper (`0.62 - 0.48|x - 0.5| - 0.38 cos(2*pi*x)` for `x` in `[0, 1]`), giving lower side lobes than the plain Bartlett window.
pub fn barthann(m: usize, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let mut w = vec![0.0; m2];
    for (n, wn) in w.iter_mut().enumerate() {
        let x = n as f64 / (m2 as f64 - 1.0);
        *wn = 0.62 - 0.48 * (x - 0.5).abs() - 0.38 * (2.0 * PI * x).cos();
    }
    truncate(w, needs_trunc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn test_barthann() {
        assert_window(include_str!("../test_data/barthann.csv"), barthann(7, true));
    }
}
