//! Statistics: descriptive statistics, correlation, distributions (univariate and multivariate),
//! hypothesis tests (including Cramér-von Mises, Brunner-Munzel and Tukey's HSD), effect sizes for
//! contingency tables, kernel density estimation and the bootstrap.
//!
//! See the book chapter [Statistics](https://jwkunz.github.io/lib_jsl/book/statistics/index.html).

// Guards such as `!(x > 0.0)` are deliberate: unlike `x <= 0.0` they also reject NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod descriptive;
pub mod distributions;
pub mod normality;
pub mod comparisons;
pub mod correlation;
pub mod hypothesis_tests;
pub mod kernel_density;
pub mod bootstrap;
pub mod location_scale;
pub mod regression;
pub mod distance;
pub mod multivariate;
pub mod goodness_of_fit;
pub mod posthoc;
pub mod contingency;
pub mod mvn_cdf;
pub mod qmc_normal;
