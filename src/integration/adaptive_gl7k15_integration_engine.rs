use crate::{integration::can_integrate::CanIntegrate1D, prelude::ErrorsJSL};

//const GUASS_LEGENDRE_NODES_7:[f64;4] =[0.0,0.405845151377397,0.741531185599394,0.949107912342759];
const GUASS_LEGENDRE_WEIGHTS_7: [f64; 4] = [
    0.417959183673469,
    0.381830050505119,
    0.279705391489277,
    0.129484966168870,
];
const KRONROD_NODES_15: [f64; 8] = [
    0.0,
    0.207784955007898,
    0.405845151377397,
    0.586087235467691,
    0.741531185599394,
    0.864864423359769,
    0.949107912342759,
    0.991455371120813,
];
const KRONROD_WEIGHTS_15: [f64; 8] = [
    0.209482141084728,
    0.204432940075298,
    0.190350578064785,
    0.169004726639267,
    0.140653259715525,
    0.104790010322250,
    0.063092092629979,
    0.022935322010529,
];

// Helper function to scale the sample locations
fn scale_x(a: f64, b: f64, x: f64) -> f64 {
    (x + 1.0) * 0.5 * (b - a) + a
}
// Helper function to scale the weights
fn scale_w(a: f64, b: f64, w: f64) -> f64 {
    w * 0.5 * (b - a)
}
// Calculation of the quadrature rules
fn calculate_quadrature<F>(a: f64, b: f64, f: &F) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    // Scale the nodes and weights
    let kr_nodes: Vec<f64> = KRONROD_NODES_15
        .iter()
        .rev()
        .map(|&x| scale_x(a, b, x))
        .chain(KRONROD_NODES_15.iter().skip(1).map(|&x| scale_x(a, b, -x)))
        .collect();
    let kr_weights: Vec<f64> = KRONROD_WEIGHTS_15
        .iter()
        .rev()
        .map(|&w| scale_w(a, b, w))
        .chain(KRONROD_WEIGHTS_15.iter().skip(1).map(|&w| scale_w(a, b, w)))
        .collect();
    let gl_weights: Vec<f64> = GUASS_LEGENDRE_WEIGHTS_7
        .iter()
        .rev()
        .map(|&w| scale_w(a, b, w))
        .chain(
            GUASS_LEGENDRE_WEIGHTS_7
                .iter()
                .skip(1)
                .map(|&w| scale_w(a, b, w)),
        )
        .collect();
    // Sample the function
    let samples: Vec<f64> = kr_nodes.iter().map(|&x| f(x)).collect();
    // Integrate
    let gl: f64 = samples
        .iter()
        .skip(1)
        .step_by(2)
        .zip(gl_weights)
        .map(|(x, w)| x * w)
        .sum();
    let kr: f64 = samples.iter().zip(kr_weights).map(|(x, w)| x * w).sum();
    (gl, kr)
}

// An struct the integrates a 1D real function f(x) using a Guass-Legendre 7 and Kronrod 15 quadrature to inspect the convergence error
pub struct AdaptiveGL7K15IntegrationEngine {
    tolerance: f64,   // Where to bound error on each subinterval
    lower_bound: f64, // The lower bound of x
    upper_bound: f64, // The upper obund of x
}

impl AdaptiveGL7K15IntegrationEngine {
    // Construct from bounds and number of points
    // The boolean flag is if you want to cache the x domain points to save speed on repeated calls (true) or recompute the x points each time (false)
    pub fn new(tolerance: f64, lower_bound: f64, upper_bound: f64) -> Self {
        AdaptiveGL7K15IntegrationEngine {
            tolerance,
            lower_bound,
            upper_bound,
        }
    }
    // Dynamically sets the number of points.  Will update cache if configured.
    pub fn set_tolerance(&mut self, x: f64) {
        self.tolerance = x;
    }
}

impl CanIntegrate1D for AdaptiveGL7K15IntegrationEngine {
    fn integrate<F>(&self, f: F) -> Result<f64, ErrorsJSL>
    where
        F: Fn(f64) -> f64,
    {
        Ok(recurse(
            self.lower_bound,
            self.upper_bound,
            &f,
            self.tolerance,
        ))
    }

    // Dynamically sets the lower bound.  Will update cache if configured.
    fn set_lower_bound(&mut self, x: f64) {
        self.lower_bound = x;
    }
    // Dynamically sets the upper bound.  Will update cache if configured.
    fn set_upper_bound(&mut self, x: f64) {
        self.upper_bound = x;
    }
}

fn recurse<F>(a: f64, b: f64, f: &F, tolerance: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    let (gl, kr) = calculate_quadrature(a, b, &f);
    // Compute difference
    let difference = (gl - kr).abs();
    // Decide to keep or split
    if difference < tolerance {
        kr
    } else {
        let mid = 0.5 * (a + b);
        let left = recurse(a, mid, f, tolerance);
        let right = recurse(mid, b, f, tolerance);
        left + right
    }
}

#[cfg(test)]
mod test {
    use std::f64::consts::PI;

    use super::*;
    #[test]
    fn test_gl7k15_adaptive_integration() {
        let error_limit = 1E-6;
        let upper_bound = 0.0;
        let lower_bound = PI / 2.0;
        let dut =
            AdaptiveGL7K15IntegrationEngine::new(error_limit * 0.001, lower_bound, upper_bound);
        let result = dut.integrate(|x| x.sin()).unwrap();
        let error = -1.0 - result;
        dbg!(&error);
        assert!(error.abs() < error_limit);

        let upper_bound = PI / 2.0;
        let lower_bound = -PI / 2.0;
        let dut =
            AdaptiveGL7K15IntegrationEngine::new(error_limit * 0.001, lower_bound, upper_bound);
        let result = dut.integrate(|x| x.sin()).unwrap();
        let error = 0.0 - result;
        dbg!(&error);
        assert!(error.abs() < error_limit);
    }
}
