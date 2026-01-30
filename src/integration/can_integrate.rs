use crate::prelude::ErrorsJSL;

// A trait that define a 1D integration routine
pub trait CanIntegrate1D {
    // Sets the lower bound of the x domain
    fn set_lower_bound(&mut self, x : f64);
    // Sets the upper bound of the x domain
    fn set_upper_bound(&mut self, x : f64);
    // Integrates f(x)dx
    fn integrate<F>(&self, f: F) -> Result<f64,ErrorsJSL> 
    where 
        F: Fn(f64) -> f64;
}