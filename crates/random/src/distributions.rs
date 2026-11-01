//! Transformations of uniform floating point samples [0,1) into the desired distribution,
//! e.g. `let sample = distribution(&mut rng, other args...)`. One module per distribution.

pub mod uniform;
pub mod exponential;
pub mod bernoulli;
pub mod geometric;
pub mod weibull;
pub mod poisson;
pub mod gaussian;
pub mod chi_square;
pub mod students_t;
pub mod gamma;
pub mod beta;
pub mod log_normal;
pub mod cauchy;
pub mod binomial;

pub use uniform::uniform_distribution;
pub use exponential::exponential_distribution;
pub use bernoulli::bernoulli_distribution;
pub use geometric::geometric_distribution;
pub use weibull::weibull_distribution;
pub use poisson::poisson_distribution_knuth;
pub use gaussian::{gaussian_distribution_box_muller, gaussian_distribution_box_muller_vec};
pub use chi_square::chi_square_distribution;
pub use students_t::students_t_distribution;
pub use gamma::gamma_distribution;
pub use beta::beta_distribution;
pub use log_normal::log_normal_distribution;
pub use cauchy::cauchy_distribution;
pub use binomial::binomial_distribution;
