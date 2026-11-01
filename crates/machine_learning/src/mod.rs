//! Machine learning algorithms, with the supporting traits, metrics, datasets and preprocessing
//! needed to try them out.
//!
//! - [`common`]: the model traits ([`common::Regressor`], [`common::Classifier`],
//!   [`common::Clusterer`], [`common::Transformer`]) and a seeded random helper
//! - [`ensembles`]: gradient-boosted trees for regression and classification
//! - [`datasets`]: reproducible synthetic data (blobs, moons, circles, Swiss roll, ...)
//! - [`preprocessing`]: scalers, train/test splitting, polynomial features
//! - [`kernels`]: linear, polynomial, RBF and sigmoid kernels
//! - [`metrics`]: errors, accuracy, silhouette, adjusted Rand index
//! - [`model_selection`]: k-fold and stratified cross-validation, grid search
//! - [`supervised`]: regression and classification
//! - [`unsupervised`]: clustering and dimensionality reduction

pub mod common;
pub mod datasets;
pub mod ensembles;
pub mod kernels;
pub mod metrics;
pub mod model_selection;
pub mod preprocessing;
pub mod supervised;
pub mod unsupervised;

// ndarray is built with BLAS support; referencing ndarray-linalg makes sure the BLAS/LAPACK
// libraries it links in are always pulled into dependents.
use ndarray_linalg as _;
