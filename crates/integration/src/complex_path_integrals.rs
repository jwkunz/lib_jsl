//! Complex path integrals.

pub fn complex_path_integral<F>(_f: F, _path: &[(f64, f64)]) -> (f64, f64)
where
    F: Fn((f64, f64)) -> (f64, f64),
{
    todo!("complex_path_integral is not yet implemented")
}
