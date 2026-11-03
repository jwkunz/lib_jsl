//! Summarise a skewed sample, robustly and otherwise.

use lib_jsl_statistics::descriptive::*;

fn main() {
    // Household incomes in thousands, with one very large value.
    let income = [28.0, 35.5, 41.0, 39.2, 52.8, 47.1, 33.4, 61.0, 44.6, 38.9, 55.3, 1250.0];
    let s = summary(&income).unwrap();
    println!("n = {}, mean = {:.1}, sd = {:.1}", s.count, s.mean, s.standard_deviation);
    println!("five-number summary: {:.1} {:.1} {:.1} {:.1} {:.1}", s.minimum, s.first_quartile, s.median, s.third_quartile, s.maximum);
    println!("robust spread: IQR = {:.1}, 1.4826 MAD = {:.1}", interquartile_range(&income).unwrap(), median_absolute_deviation(&income, 1.4826).unwrap());
    println!("skewness = {:.2}, excess kurtosis = {:.2}", skewness(&income).unwrap(), excess_kurtosis(&income).unwrap());
    println!("geometric mean = {:.1}, 90th percentile = {:.1}", geometric_mean(&income).unwrap(), quantile(&income, 0.9).unwrap());
}
