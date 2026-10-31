use std::{f64::consts::PI, ops::Mul};

use crate::uniform_generator::UniformGenerator;

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
    let a = uniform_sample_0.ln().mul(-2.0).sqrt();
    let z1 = a * uniform_sample_1.mul(2.0 * PI).cos();
    let z2 = a * uniform_sample_1.mul(2.0 * PI).sin();
    (z1 * sd + mean, z2 * sd + mean)
}

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
        while result.len() <= count{
            let z = gaussian_distribution_box_muller(uniform_generator, mean, sd);
            result.push(z.0);
            result.push(z.1);
        }
        if count & 0b1 == 1{ // Remove last one in the odd case
            result.pop();
        }
    } 
    result
}
