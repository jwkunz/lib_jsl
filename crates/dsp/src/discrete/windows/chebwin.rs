//! The Dolph-Chebyshev window.
//!
//! See the book entry [Window functions](https://jwkunz.github.io/lib_jsl/book/spectral/windows.html#the-dolph-chebyshev-and-dpss-windows).

use std::f64::consts::PI;

use super::{extend, len_guard, truncate};

/// The Dolph-Chebyshev window of length `m` with side lobes `at` dB below the main lobe.
///
/// Its transform is the Chebyshev polynomial `T_{m−1}(β cos(πk/m))`, with
/// `β = cosh(acosh(10^{at/20}) / (m − 1))`, which has all side lobes at the same level and the
/// narrowest main lobe for that level. The window is obtained from it by a DFT, centred and
/// normalised to a peak of 1, following SciPy's `chebwin`.
pub fn chebwin(m: usize, at: f64, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let order = m2 as f64 - 1.0;
    let beta = ((10_f64.powf(at.abs() / 20.0)).acosh() / order).cosh();
    let p: Vec<f64> = (0..m2)
        .map(|k| {
            let x = beta * (PI * k as f64 / m2 as f64).cos();
            if x > 1.0 {
                (order * x.acosh()).cosh()
            } else if x < -1.0 {
                let sign = if m2 % 2 == 1 { 1.0 } else { -1.0 };
                sign * (order * (-x).acosh()).cosh()
            } else {
                (order * x.acos()).cos()
            }
        })
        .collect();
    // The real part of the DFT of p (with a half-sample shift for even lengths).
    let shift = if m2 % 2 == 1 { 0.0 } else { PI / m2 as f64 };
    let dft = |n: usize| -> f64 {
        p.iter().enumerate().map(|(k, &pk)| pk * (shift * k as f64 - 2.0 * PI * (k * n) as f64 / m2 as f64).cos()).sum()
    };
    let mut w: Vec<f64> = if m2 % 2 == 1 {
        let half = m2.div_ceil(2);
        let tail: Vec<f64> = (0..half).map(dft).collect();
        tail[1..].iter().rev().chain(tail.iter()).copied().collect()
    } else {
        let half = m2 / 2 + 1;
        let tail: Vec<f64> = (0..half).map(dft).collect();
        tail[1..].iter().rev().chain(tail[1..].iter()).copied().collect()
    };
    let maxv = w.iter().fold(0.0_f64, |a, &b| a.max(b));
    w.iter_mut().for_each(|v| *v /= maxv);
    truncate(w, needs_trunc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn test_chebwin() {
        assert_window(include_str!("../test_data/chebwin.csv"), chebwin(7, 100.0, true));
    }
}
