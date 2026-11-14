//! Boundary modes for the `ndimage`-style routines (rank filters, morphology, derivative filters
//! and geometric transforms), with SciPy's names and meanings.
//!
//! For a row `a b c d`, the modes extend it as:
//!
//! - `Reflect`: `d c b a` before and `d c b a` after (the edge sample is repeated);
//! - `Mirror`: `d c b` before and `c b a` after (reflection about the centre of the edge sample);
//! - `Nearest`: `a a a` before and `d d d` after;
//! - `Constant(k)`: `k k k` on both sides;
//! - `Wrap`: `b c d` before and `a b c` after.

/// How values beyond the edge of an image are supplied (SciPy's `mode`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    /// Reflect about the edge, repeating the edge sample (SciPy's default for filters).
    Reflect,
    /// Reflect about the centre of the edge sample, without repeating it.
    Mirror,
    /// Repeat the edge sample.
    Nearest,
    /// A constant value.
    Constant(f64),
    /// Periodic extension.
    Wrap,
}

/// Maps an index into `0..n` for a mode; `None` means "outside, use the constant".
pub fn mode_index(i: isize, n: usize, mode: Mode) -> Option<usize> {
    let ni = n as isize;
    if (0..ni).contains(&i) {
        return Some(i as usize);
    }
    match mode {
        Mode::Constant(_) => None,
        Mode::Nearest => Some(i.clamp(0, ni - 1) as usize),
        Mode::Wrap => Some(i.rem_euclid(ni) as usize),
        Mode::Reflect => {
            let m = i.rem_euclid(2 * ni);
            Some(if m < ni { m as usize } else { (2 * ni - 1 - m) as usize })
        }
        Mode::Mirror => {
            if n == 1 {
                return Some(0);
            }
            let period = 2 * ni - 2;
            let m = i.rem_euclid(period);
            Some(if m < ni { m as usize } else { (period - m) as usize })
        }
    }
}

/// The value at `(r, c)` of an image extended by `mode`.
pub fn sample(image: &ndarray::Array2<f64>, r: isize, c: isize, mode: Mode) -> f64 {
    let (rows, cols) = image.dim();
    match (mode_index(r, rows, mode), mode_index(c, cols, mode)) {
        (Some(i), Some(j)) => image[(i, j)],
        _ => match mode {
            Mode::Constant(k) => k,
            _ => unreachable!("only the constant mode leaves the image"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_extensions_of_a_b_c_d() {
        let ext = |mode| (-3..7).map(|i| mode_index(i, 4, mode).map_or(-1, |j| j as i32)).collect::<Vec<i32>>();
        assert_eq!(ext(Mode::Reflect), vec![2, 1, 0, 0, 1, 2, 3, 3, 2, 1]);
        assert_eq!(ext(Mode::Mirror), vec![3, 2, 1, 0, 1, 2, 3, 2, 1, 0]);
        assert_eq!(ext(Mode::Nearest), vec![0, 0, 0, 0, 1, 2, 3, 3, 3, 3]);
        assert_eq!(ext(Mode::Wrap), vec![1, 2, 3, 0, 1, 2, 3, 0, 1, 2]);
        assert_eq!(ext(Mode::Constant(0.0)), vec![-1, -1, -1, 0, 1, 2, 3, -1, -1, -1]);
    }
}
