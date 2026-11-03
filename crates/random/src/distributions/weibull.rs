use crate::uniform_generator::UniformGenerator;

/// A Weibull sample with shape `k` and scale `lambda`, `λ (−ln(1 − u))^{1/k}` by inversion (`k = 1` is
/// the exponential distribution with mean `λ`).
pub fn weibull_distribution<U>(uniform_generator: &mut U, k: f64, lambda: f64) -> f64
where
    U: UniformGenerator,
{
    let uniform_sample: f64 = uniform_generator.next_f64();
    // 1 − u lies in (0, 1], so the logarithm is finite (u itself can be exactly 0).
    lambda * (-(1.0 - uniform_sample).ln()).powf(1.0 / k)
}
