use crate::uniform_generator::UniformGenerator;

/// This file contains transformation functions that transform uniform floating point distributions [0,1) into the desired distribution
/// e.g)  let sample = distribution(rng.generate_f64(), other args...)
///
/// Uniform distribution
pub fn uniform_distribution<U>(uniform_generator: &mut U, lower_bound: f64, upper_bound: f64) -> f64
where
    U: UniformGenerator,
{
    let uniform_sample: f64 = uniform_generator.next_f64();
    lower_bound + (upper_bound - lower_bound) * uniform_sample
}
