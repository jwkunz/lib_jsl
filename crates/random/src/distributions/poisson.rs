use std::ops::Neg;

use crate::uniform_generator::UniformGenerator;

/// Poisson distribution (via Knuth's algorithm for small lambda)
pub fn poisson_distribution_knuth<U>(uniform_generator: &mut U, lambda: f64) -> f64
where
    U: UniformGenerator,
{
    let uniform_sample: f64 = uniform_generator.next_f64();
    let l = lambda.neg().exp();
    let mut k = 0.0;
    let mut p = 1.0;
    while p < l {
        k += 1.0;
        p *= uniform_sample
    }
    k - 1.0
}
