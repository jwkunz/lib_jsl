//! Sample points for the composite Newton-Cotes rules.

/// `n` equally spaced points from `a` to `b` inclusive, and their spacing `h = (b − a)/(n − 1)`:
/// the nodes of the composite trapezoid and Simpson rules.
pub fn grid_points(a: f64, b: f64, n: usize) -> (Vec<f64>, f64) {
    let h = (b - a) / ((n.max(2) - 1) as f64);
    ((0..n).map(|i| a + i as f64 * h).collect(), h)
}

/// The midpoints of `n` equal panels of `[a, b]`, `a + (i + ½) h` with `h = (b − a)/n`, and `h`:
/// the nodes of the composite midpoint rule.
pub fn panel_midpoints(a: f64, b: f64, n: usize) -> (Vec<f64>, f64) {
    let h = (b - a) / (n.max(1) as f64);
    ((0..n).map(|i| a + (i as f64 + 0.5) * h).collect(), h)
}

/// The grid points of [`grid_points`]. Despite its name this never returned midpoints.
#[deprecated(note = "use `grid_points` (or `panel_midpoints` for the midpoints of panels)")]
pub fn calculate_midpoints(a: f64, b: f64, n: usize) -> (Vec<f64>, f64) {
    grid_points(a, b, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grids_and_midpoints() {
        let (x, h) = grid_points(0.0, 1.0, 5);
        assert_eq!((x, h), (vec![0.0, 0.25, 0.5, 0.75, 1.0], 0.25));
        let (m, h) = panel_midpoints(0.0, 1.0, 4);
        assert_eq!((m, h), (vec![0.125, 0.375, 0.625, 0.875], 0.25));
    }
}
