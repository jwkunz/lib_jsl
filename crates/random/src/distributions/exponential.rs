use crate::uniform_generator::UniformGenerator;

/// An exponential sample with **rate** `lambda` (mean `1/λ`) by inversion, `−ln(1 − u)/λ`: the
/// waiting time between events of a Poisson process.
pub fn exponential_distribution<U>(uniform_generator: &mut U, lambda: f64) -> f64
where
    U: UniformGenerator,
{
    let uniform_sample: f64 = uniform_generator.next_f64();
    -(1.0 - uniform_sample).ln() / lambda
}
