use crate::{distributions::gaussian::gaussian_distribution_box_muller_vec, uniform_generator::UniformGenerator};

/// Student's t distribution with k degrees of freedom
pub fn students_t_distribution<U>(uniform_generator: &mut U, k: usize, mean : f64, sd : f64) -> f64 
where U : UniformGenerator{

    let gaussian_samples = gaussian_distribution_box_muller_vec(k+1,uniform_generator,mean,sd);
    let first = gaussian_samples.first().expect("checked above");
    let chi_square: f64 = gaussian_samples.iter().skip(1).map(|x| x*x).sum();
    first / (chi_square / k as f64).sqrt()
}
