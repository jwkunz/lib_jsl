//! Newton-Raphson method.

pub fn newton_raphson<F, D>(_f: F, _df: D, _x0: f64, _tol: f64) -> Result<f64, String>
where
    F: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    todo!("newton_raphson is not yet implemented")
}
