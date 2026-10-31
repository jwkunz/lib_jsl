//! Classification.

pub mod k_nearest_neighbors;
pub mod phylogenetic_trees;
pub mod viterbi_algorithm;
pub mod random_forest;
pub mod support_vector_machine;
pub mod logistic_regression;
pub mod naive_bayes;

pub use k_nearest_neighbors::KNearestNeighbors;
pub use phylogenetic_trees::phylogenetic_trees;
pub use viterbi_algorithm::viterbi_algorithm;
pub use random_forest::{DecisionTree, RandomForest};
pub use support_vector_machine::SupportVectorMachine;
pub use logistic_regression::LogisticRegression;
pub use naive_bayes::{GaussianNaiveBayes, MultinomialNaiveBayes};
