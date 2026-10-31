use crate::uniform_generator::UniformGenerator;

/// Bernoulli X = 1 for x < p, 0 else
pub fn bernoulli_distribution<U>(uniform_generator: &mut U, p: f64) -> f64
where
    U: UniformGenerator,
{
    let uniform_sample: f64 = uniform_generator.next_f64();
    if uniform_sample < p { 1.0 } else { 0.0 }
}
