use std::{
    f64::consts::PI,
    ops::{Mul, Neg},
};

use crate::{prelude::ErrorsJSL, random::uniform_generator::UniformGenerator};

/// This file contains transformation functions that transform uniform floating point distributions [0,1) into the desired distribution
/// e.g)  let sample = distribution(rng.generate_f64(), other args...)

/// Uniform distribution
pub fn uniform_distribution<U>(uniform_generator: &mut U, lower_bound: f64, upper_bound: f64) -> f64
where
    U: UniformGenerator,
{
    let uniform_sample: f64 = uniform_generator.next_f64();
    lower_bound + (upper_bound - lower_bound) * uniform_sample
}

/// Inter arrival times
pub fn exponential_distribution<U>(uniform_generator: &mut U, lambda: f64) -> f64
where
    U: UniformGenerator,
{
    let uniform_sample: f64 = uniform_generator.next_f64();
    -(1.0 - uniform_sample).ln() / lambda
}

/// Bernoulli X = 1 for x < p, 0 else
pub fn bernouli_distribution<U>(uniform_generator: &mut U, p: f64) -> f64
where
    U: UniformGenerator,
{
    let uniform_sample: f64 = uniform_generator.next_f64();
    if uniform_sample < p { 1.0 } else { 0.0 }
}

/// The number of failures before first success
pub fn geometric_distribution<U>(uniform_generator: &mut U, p: f64) -> f64
where
    U: UniformGenerator,
{
    let uniform_sample: f64 = uniform_generator.next_f64();
    ((1.0 - uniform_sample).ln() / (1.0 - p).ln()).floor()
}

/// Generalized exponential
pub fn weibull_distribution<U>(uniform_generator: &mut U, k: f64, lambda: f64) -> f64
where
    U: UniformGenerator,
{
    let uniform_sample: f64 = uniform_generator.next_f64();
    lambda * (-uniform_sample.ln()).powf(1.0 / k)
}

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

/// Guassian distribution (via Box - Muller transform)
/// Yields two independent samples
pub fn guassian_distribution_box_muller<U>(
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

// Utility for efficiently creating a vector of guassian random variables
pub fn guassian_distribution_box_muller_vec<U>(
    count : usize,
    uniform_generator: &mut U,
    mean: f64,
    sd: f64,
)-> Vec<f64>
where
    U: UniformGenerator{
    let mut result = Vec::<f64>::with_capacity(count+1);
    if count == 1{
        let z = guassian_distribution_box_muller(uniform_generator, mean, sd);
        result.push(z.0);
    }else{
        while result.len() <= count{
            let z = guassian_distribution_box_muller(uniform_generator, mean, sd);
            result.push(z.0);
            result.push(z.1);
        }
        if count & 0b1 == 1{ // Remove last one in the odd case
            result.pop();
        }
    } 
    result
}

/// Chi-Square
/// The order k is the number of indpendent guassian_samples given in the slice
pub fn chi_square_distribution<U>(uniform_generator: &mut U, k: usize, mean : f64, sd : f64) -> f64
where
    U: UniformGenerator,
{
    guassian_distribution_box_muller_vec(k,uniform_generator,mean,sd).iter().map(|x| x*x).sum()
}

/// Student's t distribution with k degrees of freedom
pub fn students_t_distribution<U>(uniform_generator: &mut U, k: usize, mean : f64, sd : f64) -> f64 
where U : UniformGenerator{

    let guassian_samples = guassian_distribution_box_muller_vec(k+1,uniform_generator,mean,sd);
    let first = guassian_samples.first().expect("checked above");
    let chi_square: f64 = guassian_samples.iter().skip(1).map(|x| x*x).sum();
    first / (chi_square / k as f64).sqrt()
}


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
        let z = guassian_distribution_box_muller(uniform_generator, 0.0, 1.0).0;
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

/// Generate a Beta(alpha, beta) random variable
pub fn beta_distribution<U>(uniform_generator: &mut U, alpha: f64, beta: f64) -> Result<f64,ErrorsJSL> where U : UniformGenerator{
    let x = gamma_distribution(uniform_generator, alpha, 1.0)?;
    let y = gamma_distribution(uniform_generator, beta, 1.0)?;
    Ok(x / (x+y))
}