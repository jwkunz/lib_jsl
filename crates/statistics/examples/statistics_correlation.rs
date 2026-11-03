//! Three correlation coefficients on a monotone but nonlinear relation with an outlier.

use lib_jsl_statistics::correlation::{correlation_matrix, kendall, pearson, spearman};

fn main() {
    // Dose and response: saturating growth, with the last response misrecorded.
    let dose: Vec<f64> = (1..=15).map(|d| d as f64).collect();
    let mut response: Vec<f64> = dose.iter().map(|d| 10.0 * (1.0 - (-d / 4.0f64).exp()) + 0.3 * (d * 1.7).sin()).collect();
    response[14] = 0.5;
    for (name, r) in [("Pearson", pearson(&dose, &response)), ("Spearman", spearman(&dose, &response)), ("Kendall", kendall(&dose, &response))] {
        let r = r.unwrap();
        println!("{name:<9} {:+.3}  (p = {:.2e})", r.coefficient, r.p_value);
    }
    let m = correlation_matrix(&[dose.clone(), response.clone(), dose.iter().map(|d| d * d).collect()]).unwrap();
    for row in m {
        println!("{}", row.iter().map(|v| format!("{v:+.3}")).collect::<Vec<_>>().join("  "));
    }
}
