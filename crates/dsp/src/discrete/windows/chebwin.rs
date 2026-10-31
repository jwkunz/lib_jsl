use std::f64::consts::PI;

use super::{extend, len_guard, truncate};

/// The Chebyshev window is defined by a specific mathematical formula that involves the hyperbolic cosine function and the inverse hyperbolic cosine function. The coefficients are computed based on the position within the window, the total length, and the attenuation parameter `at`, which controls the shape of the window and its spectral properties.
pub fn chebwin(m: usize, at: f64, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let beta = ((10_f64.powf(at / 20.0)).acosh() / m2 as f64).cosh();
    let den = (m2 as f64 * beta.acosh()).cosh();
    let mut wf = vec![0.0; m2];
    for (k, v) in wf.iter_mut().enumerate() {
        let x = beta * (PI * k as f64 / m2 as f64).cos();
        *v = if x.abs() <= 1.0 {
            (m2 as f64 * x.acos()).cos() / den
        } else if x > 1.0 {
            (m2 as f64 * x.acosh()).cosh() / den
        } else {
            let s = if m2 % 2 == 0 { 1.0 } else { -1.0 };
            s * (m2 as f64 * (-x).acosh()).cosh() / den
        };
    }
    let mut w = vec![0.0; m2];
    for (n, wn) in w.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (k, wk) in wf.iter().enumerate() {
            let phase = 2.0 * PI * k as f64 * (n as f64 - (m2 as f64 - 1.0) / 2.0) / m2 as f64;
            acc += *wk * phase.cos();
        }
        *wn = acc / m2 as f64;
    }
    let maxv = w.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    if maxv > 0.0 {
        for v in &mut w {
            *v /= maxv;
        }
    }
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
