use crate::discrete::sinc::sinc;

use super::{extend, len_guard, truncate};

/// The Lanczos window is the central lobe of the sinc function, `sinc(2n/(M-1) - 1)`, giving a smooth taper that is zero-crossing at the edges.
pub fn lanczos(m: usize, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let mut w = vec![0.0; m2];
    for (n, wn) in w.iter_mut().enumerate() {
        let x = 2.0 * n as f64 / (m2 as f64 - 1.0) - 1.0;
        *wn = sinc(x);
    }
    truncate(w, needs_trunc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn test_lanczos() {
        assert_window(include_str!("../test_data/lanczos.csv"), lanczos(7, true));
    }
}
