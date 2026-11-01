//! Distribution functions and quantiles for statistics.
//!
//! Run with `cargo run -p lib_jsl_special_functions --example special_statistical`.

use lib_jsl_special_functions::{binomial_cdf, chi_square_quantile, chi_square_survival, normal_cdf, normal_quantile, poisson_cdf, student_t_cdf, student_t_quantile};

fn main() {
    // A 95% confidence interval for a mean from 8 measurements uses the t quantile with 7
    // degrees of freedom instead of the normal 1.96.
    println!("z(0.975) = {:.6}, t₇(0.975) = {:.6}", normal_quantile(0.975, 0.0, 1.0), student_t_quantile(0.975, 7.0));
    // A two-sided p-value for t = 2.5 with 7 degrees of freedom.
    println!("p = {:.6}", 2.0 * (1.0 - student_t_cdf(2.5, 7.0)));
    // A chi-square goodness-of-fit statistic of 15.2 on 6 degrees of freedom.
    println!("χ² = 15.2, k = 6: p = {:.6}; critical value at 5%: {:.6}", chi_square_survival(15.2, 6.0), chi_square_quantile(0.95, 6.0));
    // Discrete distributions: at most 2 defects among 100 items at rate 1%; at most 3 events at rate 1.5.
    println!("binomial P(X ≤ 2) = {:.6}, Poisson P(N ≤ 3) = {:.6}", binomial_cdf(2, 100, 0.01), poisson_cdf(3, 1.5));
    println!("Φ(1.645) = {:.6}", normal_cdf(1.645, 0.0, 1.0));
}
