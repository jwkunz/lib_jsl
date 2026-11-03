//! Compare two treatments with parametric and rank tests.

use lib_jsl_statistics::hypothesis_tests::*;

fn main() {
    // Recovery times in days under two treatments.
    let a = [12.1, 9.8, 11.4, 14.0, 10.2, 13.5, 9.9, 12.8, 11.0, 15.2, 10.7, 12.3];
    let b = [8.9, 10.1, 7.5, 9.4, 11.8, 8.2, 9.0, 7.9, 10.5, 30.0];
    let welch = t_test_two_sample(&a, &b, false, Alternative::TwoSided).unwrap();
    let student = t_test_two_sample(&a, &b, true, Alternative::TwoSided).unwrap();
    let u = mann_whitney_u(&a, &b, Alternative::TwoSided).unwrap();
    let ks = ks_test_two_sample(&a, &b).unwrap();
    println!("Welch t = {:.3} on {:.1} df, p = {:.4}", welch.statistic, welch.degrees_of_freedom, welch.p_value);
    println!("Student t = {:.3} on {} df, p = {:.4}", student.statistic, student.degrees_of_freedom, student.p_value);
    println!("Mann-Whitney U = {}, p = {:.4}", u.statistic, u.p_value);
    println!("Kolmogorov-Smirnov D = {:.3}, p = {:.4}", ks.statistic, ks.p_value);
    // A contingency table: side effect (rows) by treatment (columns).
    let table = vec![vec![14.0, 5.0], vec![36.0, 45.0]];
    let chi = chi_square_independence(&table).unwrap();
    println!("chi-square = {:.3} on {} df, p = {:.4}", chi.statistic, chi.degrees_of_freedom, chi.p_value);
}
