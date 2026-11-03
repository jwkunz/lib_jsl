//! Bootstrap confidence intervals for a median and a ratio of means.

use lib_jsl_statistics::bootstrap::bootstrap;
use lib_jsl_statistics::descriptive::median;

fn main() {
    // Response times in milliseconds, right-skewed.
    let times = [112.0, 98.0, 143.0, 87.0, 251.0, 104.0, 133.0, 96.0, 178.0, 121.0, 92.0, 310.0, 118.0, 101.0, 145.0, 89.0, 127.0, 166.0, 109.0, 95.0];
    let statistic = |x: &[f64]| median(x).unwrap();
    let b = bootstrap(&times, statistic, 5000, 42).unwrap();
    println!("median = {:.1} ms, bootstrap standard error = {:.1}, bias = {:+.2}", b.estimate, b.standard_error, b.bias);
    let (lo, hi) = b.percentile_interval(0.95);
    println!("95% percentile interval: [{lo:.1}, {hi:.1}]");
    let (lo, hi) = b.bca_interval(&times, statistic, 0.95);
    println!("95% BCa interval:        [{lo:.1}, {hi:.1}]");
    // The coefficient of variation, a statistic with no simple standard-error formula.
    let cv = |x: &[f64]| {
        let m = x.iter().sum::<f64>() / x.len() as f64;
        (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() - 1) as f64).sqrt() / m
    };
    let b = bootstrap(&times, cv, 5000, 7).unwrap();
    let (lo, hi) = b.bca_interval(&times, cv, 0.95);
    println!("coefficient of variation = {:.3}, 95% BCa interval [{lo:.3}, {hi:.3}]", b.estimate);
}
