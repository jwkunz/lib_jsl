//! Root finding algorithms.

pub fn bisection<F>(_f: F, _a: f64, _b: f64, _tol: f64) -> Result<f64, String>
where
    F: Fn(f64) -> f64,
{
    todo!("bisection is not yet implemented")
}

pub fn secant_method<F>(_f: F, _x0: f64, _x1: f64, _tol: f64) -> Result<f64, String>
where
    F: Fn(f64) -> f64,
{
    todo!("secant_method is not yet implemented")
}

pub fn false_position<F>(_f: F, _a: f64, _b: f64, _tol: f64) -> Result<f64, String>
where
    F: Fn(f64) -> f64,
{
    todo!("false_position is not yet implemented")
}

pub fn ridders_method<F>(_f: F, _a: f64, _b: f64, _tol: f64) -> Result<f64, String>
where
    F: Fn(f64) -> f64,
{
    todo!("ridders_method is not yet implemented")
}

pub fn brent_method<F>(_f: F, _a: f64, _b: f64, _tol: f64) -> Result<f64, String>
where
    F: Fn(f64) -> f64,
{
    todo!("brent_method is not yet implemented")
}

pub fn newton_raphson<F, D>(_f: F, _df: D, _x0: f64, _tol: f64) -> Result<f64, String>
where
    F: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    todo!("newton_raphson is not yet implemented")
}

pub fn polynomial_roots(_coeffs: &[f64]) -> Vec<f64> {
    todo!("polynomial_roots is not yet implemented")
}
