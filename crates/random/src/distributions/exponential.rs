use crate::uniform_generator::UniformGenerator;

/// Inter arrival times
pub fn exponential_distribution<U>(uniform_generator: &mut U, lambda: f64) -> f64
where
    U: UniformGenerator,
{
    let uniform_sample: f64 = uniform_generator.next_f64();
    -(1.0 - uniform_sample).ln() / lambda
}
