use crate::uniform_generator::UniformGenerator;
use lib_jsl_core::ErrorsJSL;
use std::f64::consts::PI;

/// A Cauchy (Lorentz) sample `x₀ + γ tan(π(u − ½))` by inversion. The distribution has no mean
/// or variance; its median is `x₀` and its quartiles are `x₀ ± γ`.
pub fn cauchy_distribution<U: UniformGenerator>(uniform_generator: &mut U, location: f64, scale: f64) -> Result<f64, ErrorsJSL> {
    if scale.is_nan() || scale <= 0.0 || !location.is_finite() {
        return Err(ErrorsJSL::InvalidInputRange("scale must be positive and location finite"));
    }
    Ok(location + scale * (PI * (uniform_generator.next_f64() - 0.5)).tan())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uniform_generator::UniformRNG;

    #[test]
    fn median_and_quartiles_match_theory() {
        let mut rng = UniformRNG::from_seed(5);
        let (x0, gamma) = (-1.5, 2.0);
        let mut s: Vec<f64> = (0..200_000).map(|_| cauchy_distribution(&mut rng, x0, gamma).unwrap()).collect();
        s.sort_by(f64::total_cmp);
        let q = |p: f64| s[(p * s.len() as f64) as usize];
        assert!((q(0.5) - x0).abs() < 0.03 && (q(0.25) - (x0 - gamma)).abs() < 0.05 && (q(0.75) - (x0 + gamma)).abs() < 0.05);
        assert!(cauchy_distribution(&mut rng, 0.0, 0.0).is_err());
    }
}
