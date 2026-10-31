use crate::uniform_generator::UniformGenerator;

/// Generalized exponential
pub fn weibull_distribution<U>(uniform_generator: &mut U, k: f64, lambda: f64) -> f64
where
    U: UniformGenerator,
{
    let uniform_sample: f64 = uniform_generator.next_f64();
    lambda * (-uniform_sample.ln()).powf(1.0 / k)
}
