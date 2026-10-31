use crate::uniform_generator::UniformGenerator;

/// The number of failures before first success
pub fn geometric_distribution<U>(uniform_generator: &mut U, p: f64) -> f64
where
    U: UniformGenerator,
{
    let uniform_sample: f64 = uniform_generator.next_f64();
    ((1.0 - uniform_sample).ln() / (1.0 - p).ln()).floor()
}
