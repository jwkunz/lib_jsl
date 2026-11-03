use crate::uniform_generator::UniformGenerator;

/// A Bernoulli sample: 1 with probability `p` (a uniform below `p`), 0 otherwise.
pub fn bernoulli_distribution<U>(uniform_generator: &mut U, p: f64) -> f64
where
    U: UniformGenerator,
{
    let uniform_sample: f64 = uniform_generator.next_f64();
    if uniform_sample < p { 1.0 } else { 0.0 }
}
