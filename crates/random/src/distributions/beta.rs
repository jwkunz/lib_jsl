use crate::{distributions::gamma::gamma_distribution, uniform_generator::UniformGenerator};
use lib_jsl_core::ErrorsJSL;

/// Generate a Beta(alpha, beta) random variable
pub fn beta_distribution<U>(uniform_generator: &mut U, alpha: f64, beta: f64) -> Result<f64,ErrorsJSL> where U : UniformGenerator{
    let x = gamma_distribution(uniform_generator, alpha, 1.0)?;
    let y = gamma_distribution(uniform_generator, beta, 1.0)?;
    Ok(x / (x+y))
}
