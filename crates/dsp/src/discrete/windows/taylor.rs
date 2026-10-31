use std::f64::consts::PI;

use super::{extend, len_guard, truncate};

/// The Taylor window is defined by a specific mathematical formula that involves the hyperbolic cosine function and the inverse hyperbolic cosine function, as well as a set of coefficients that are computed based on the position within the window, the total length, and the parameters `nbar` and `sll`, which control the shape of the window and its spectral properties. The coefficients are computed using a specific algorithm that involves a product of terms based on the position within the window and the parameters, and then normalized to ensure that the maximum value of the window is 1.0 if `norm` is true.
pub fn taylor(m: usize, nbar: usize, sll: f64, norm: bool, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    if nbar < 2 {
        return vec![1.0; m];
    }
    let (m2, needs_trunc) = extend(m, sym);
    let a = ((10_f64.powf(sll / 20.0)).acosh()) / PI;
    let s2 = (nbar as f64).powi(2) / (a * a + (nbar as f64 - 0.5).powi(2));
    let ma: Vec<f64> = (1..nbar).map(|x| x as f64).collect();
    let m2v: Vec<f64> = ma.iter().map(|x| x * x).collect();

    let mut fm = vec![0.0; nbar - 1];
    for mi in 0..ma.len() {
        let mval2 = m2v[mi];
        let sign = if mi % 2 == 0 { 1.0 } else { -1.0 };
        let mut numer = sign;
        for ma_i in &ma {
            numer *= 1.0 - mval2 / s2 / (a * a + (ma_i - 0.5).powi(2));
        }
        let mut denom = 2.0;
        for &m2vj in m2v.iter().take(mi) {
            denom *= 1.0 - mval2 / m2vj;
        }
        for &m2vj in m2v.iter().skip(mi + 1) {
            denom *= 1.0 - mval2 / m2vj;
        }
        fm[mi] = numer / denom;
    }

    let mut w = vec![0.0; m2];
    for (n, wn) in w.iter_mut().enumerate() {
        let x = (n as f64) - m2 as f64 / 2.0 + 0.5;
        let mut val = 1.0;
        for (i, mval) in ma.iter().enumerate() {
            val += 2.0 * fm[i] * (2.0 * PI * mval * x / m2 as f64).cos();
        }
        *wn = val;
    }

    if norm {
        let x = (m2 as f64 - 1.0) / 2.0 - m2 as f64 / 2.0 + 0.5;
        let mut wc = 1.0;
        for (i, mval) in ma.iter().enumerate() {
            wc += 2.0 * fm[i] * (2.0 * PI * mval * x / m2 as f64).cos();
        }
        if wc != 0.0 {
            for v in &mut w {
                *v /= wc;
            }
        }
    }
    truncate(w, needs_trunc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn test_taylor() {
        assert_window(include_str!("../test_data/taylor.csv"), taylor(7, 4, 30.0, true, true));
    }
}
