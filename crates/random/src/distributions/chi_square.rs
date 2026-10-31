use crate::{distributions::gaussian::gaussian_distribution_box_muller_vec, uniform_generator::UniformGenerator};

/// Chi-Square
/// The order k is the number of independent gaussian_samples given in the slice
pub fn chi_square_distribution<U>(uniform_generator: &mut U, k: usize, mean : f64, sd : f64) -> f64
where
    U: UniformGenerator,
{
    gaussian_distribution_box_muller_vec(k,uniform_generator,mean,sd).iter().map(|x| x*x).sum()
}
