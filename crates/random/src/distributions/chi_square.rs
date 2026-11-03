use crate::{distributions::gaussian::gaussian_distribution_box_muller_vec, uniform_generator::UniformGenerator};

/// The sum of the squares of `k` independent `N(mean, sd²)` samples. With `mean = 0` and `sd = 1`
/// this is the chi-square distribution with `k` degrees of freedom; a non-zero mean gives the
/// noncentral chi-square distribution (scaled by `sd²`).
pub fn chi_square_distribution<U>(uniform_generator: &mut U, k: usize, mean : f64, sd : f64) -> f64
where
    U: UniformGenerator,
{
    gaussian_distribution_box_muller_vec(k,uniform_generator,mean,sd).iter().map(|x| x*x).sum()
}
