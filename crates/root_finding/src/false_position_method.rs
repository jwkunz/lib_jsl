//! False position method.

pub fn false_position<F>(_f: F, _a: f64, _b: f64, _tol: f64) -> Result<f64, String>
where
    F: Fn(f64) -> f64,
{
    todo!("false_position is not yet implemented")
}
