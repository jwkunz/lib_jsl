//! Regression: predicting a real-valued target from features.

pub mod least_squares;
pub mod general_least_squares;
pub mod ridge;
pub mod elastic_net;
pub mod support_vector_regression;

pub use elastic_net::ElasticNet;
pub use general_least_squares::GeneralLeastSquares;
pub use least_squares::LinearRegression;
pub use ridge::Ridge;
pub use support_vector_regression::SupportVectorRegression;
