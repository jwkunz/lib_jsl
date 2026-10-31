use crate::{distributions::gaussian::gaussian_distribution_box_muller, uniform_generator::UniformGenerator};
use lib_jsl_core::ErrorsJSL;

/// Generate a Gamma(alpha, beta) random variable using Marsaglia-Tsang
pub fn gamma_distribution<U>(uniform_generator: &mut U, alpha: f64, beta: f64) -> Result<f64,ErrorsJSL> where U : UniformGenerator{
    if alpha < 0.0 || beta > 0.0{
        return Err(ErrorsJSL::InvalidInputRange("alpha and beta must be positive"));
    }

    if alpha < 1.0 {
        // Boosting identity for alpha < 1
        let u = uniform_generator.next_f64();
        return Ok(gamma_distribution(uniform_generator,alpha + 1.0, 1.0).expect("Already sanitized") * u.powf(1.0 / alpha) * beta);
    }

    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let z = gaussian_distribution_box_muller(uniform_generator, 0.0, 1.0).0;
        let v = 1.0 + c * z;
        if v <= 0.0 {
            continue; // Reject
        }
        let v_cubed = v * v * v;
        let u = uniform_generator.next_f64();
        if u.ln() < 0.5 * z * z + d - d * v_cubed + d * v_cubed.ln() {
            return Ok(d * v_cubed * beta);
        }
    }
}
