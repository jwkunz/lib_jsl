use super::{extend, len_guard, truncate};

/// The triangular window, also known as the Bartlett window, is defined by a linear tapering of the coefficients from the center to the edges. The coefficients are computed based on the position within the window and the total length, with special handling for even and odd lengths to ensure proper symmetry.
pub fn triang(m: usize, sym: bool) -> Vec<f64> {
    if let Some(w) = len_guard(m) {
        return w;
    }
    let (m2, needs_trunc) = extend(m, sym);
    let n: Vec<f64> = (1..=m2.div_ceil(2)).map(|x| x as f64).collect();
    let mut w = if m2.is_multiple_of(2) {
        n.iter().map(|x| (2.0 * x - 1.0) / m2 as f64).collect::<Vec<_>>()
    } else {
        n.iter().map(|x| 2.0 * x / (m2 as f64 + 1.0)).collect::<Vec<_>>()
    };
    let mut tail = w.clone();
    if m2 % 2 != 0 {
        tail.pop();
    }
    tail.reverse();
    w.extend(tail);
    truncate(w, needs_trunc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::windows::test_support::assert_window;

    #[test]
    fn test_triang() {
        assert_window(include_str!("../test_data/triang.csv"), triang(7, true));
    }
}
