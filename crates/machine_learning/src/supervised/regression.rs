//! Regression.

pub mod least_squares;
pub mod general_least_squares;
pub mod ridge;
pub mod elastic_net;
pub mod support_vector_regression;

pub use least_squares::least_squares;
pub use general_least_squares::general_least_squares;
pub use ridge::ridge_regression;
pub use elastic_net::elastic_net;
pub use support_vector_regression::support_vector_regression;
