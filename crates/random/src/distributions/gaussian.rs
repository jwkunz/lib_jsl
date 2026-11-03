use std::{f64::consts::PI, ops::Mul};

use crate::uniform_generator::UniformGenerator;

// ANCHOR: box_muller
/// Gaussian distribution (via Box - Muller transform)
/// Yields two independent samples
pub fn gaussian_distribution_box_muller<U>(
    uniform_generator: &mut U,
    mean: f64,
    sd: f64,
) -> (f64, f64)
where
    U: UniformGenerator,
{
    let uniform_sample_0: f64 = uniform_generator.next_f64();
    let uniform_sample_1: f64 = uniform_generator.next_f64();
    // 1 − u lies in (0, 1], so the logarithm is finite (u itself can be exactly 0).
    let a = (1.0 - uniform_sample_0).ln().mul(-2.0).sqrt();
    let z1 = a * uniform_sample_1.mul(2.0 * PI).cos();
    let z2 = a * uniform_sample_1.mul(2.0 * PI).sin();
    (z1 * sd + mean, z2 * sd + mean)
}
// ANCHOR_END: box_muller

// Utility for efficiently creating a vector of gaussian random variables
pub fn gaussian_distribution_box_muller_vec<U>(
    count : usize,
    uniform_generator: &mut U,
    mean: f64,
    sd: f64,
)-> Vec<f64>
where
    U: UniformGenerator{
    let mut result = Vec::<f64>::with_capacity(count+1);
    if count == 1{
        let z = gaussian_distribution_box_muller(uniform_generator, mean, sd);
        result.push(z.0);
    }else{
        while result.len() < count{
            let z = gaussian_distribution_box_muller(uniform_generator, mean, sd);
            result.push(z.0);
            result.push(z.1);
        }
        // Each draw makes two values: drop the spare one for an odd count.
        result.truncate(count);
    } 
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::chi_square::chi_square_distribution;
    use crate::uniform_generator::UniformRNG;

    #[test]
    fn the_vector_has_the_requested_length() {
        // Even counts once returned two extra values (and a count of 0 returned two).
        let mut rng = UniformRNG::from_seed(3);
        for count in 0..8 {
            assert_eq!(gaussian_distribution_box_muller_vec(count, &mut rng, 0.0, 1.0).len(), count);
        }
    }

    #[test]
    fn chi_square_with_even_degrees_has_mean_k() {
        // Through the bug above, chi-square with even k summed k + 2 squares: mean k + 2.
        let mut rng = UniformRNG::from_seed(5);
        for k in [2usize, 4] {
            let n = 40_000;
            let mean = (0..n).map(|_| chi_square_distribution(&mut rng, k, 0.0, 1.0)).sum::<f64>() / n as f64;
            assert!((mean - k as f64).abs() < 0.1, "k = {k}: mean {mean}");
        }
    }
}
