// Helper for computing equally spaced points
pub fn calculate_midpoints(a: f64, b: f64, n: usize) -> (Vec<f64>, f64) {
    let dx = (b - a) / ((n - 1) as f64);
    let x_points = (0..n).into_iter().map(|x| x as f64 * dx + a).collect();
    (x_points, dx)
}