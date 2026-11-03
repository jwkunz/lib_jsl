use crate::uniform_generator::UniformGenerator;

/// The number of failures before the first success in independent trials of probability `p`
/// (`P(k) = (1 − p)ᵏ p`), by inversion of the exponential: `⌊ln(1 − u) / ln(1 − p)⌋`.
pub fn geometric_distribution<U>(uniform_generator: &mut U, p: f64) -> f64
where
    U: UniformGenerator,
{
    let uniform_sample: f64 = uniform_generator.next_f64();
    ((1.0 - uniform_sample).ln() / (1.0 - p).ln()).floor()
}
