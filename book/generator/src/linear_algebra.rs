//! Figures and tables for the dense linear algebra chapters.

use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_linalg::{Cholesky, Inverse, Solve, UPLO};
use lib_jsl_linalg_reference as reference;
use ndarray::{Array1, Array2, array};
use std::time::Instant;

pub(crate) const CHAPTER: &str = "linear_algebra";

fn inf_norm_matrix(a: &Array2<f64>) -> f64 {
    a.rows().into_iter().map(|r| r.iter().map(|v| v.abs()).sum::<f64>()).fold(0.0, f64::max)
}

fn inf_norm(v: &Array1<f64>) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

/// The median of several timings of `f`, in seconds.
pub(crate) fn time(mut f: impl FnMut()) -> f64 {
    let mut t: Vec<f64> = (0..5)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed().as_secs_f64()
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[2]
}

fn pivoting_tables() {
    // The 2 × 2 example [[ε, 1], [1, 1]] x = [1, 2], solution ≈ (1, 1).
    let mut table = String::from("| $\\varepsilon$ | without pivoting: $x_1$ | with partial pivoting: $x_1$ | exact $x_1 = 1/(1 - \\varepsilon)$ |\n|---:|---:|---:|---:|\n");
    for e in [1e-2, 1e-6, 1e-10, 1e-14, 1e-17] {
        let a = array![[e, 1.0], [1.0, 1.0]];
        let b = array![1.0, 2.0];
        let naive = reference::gaussian_elimination::solve_without_pivoting(&a, &b).unwrap();
        let pivoted = reference::gaussian_elimination::solve_with_partial_pivoting(&a, &b).unwrap();
        table.push_str(&format!("| $10^{{{}}}$ | {:.16} | {:.16} | {:.16} |\n", e.log10().round(), naive[0], pivoted[0], 1.0 / (1.0 - e)));
    }
    write(CHAPTER, "small_pivot.md", &table);

    // Wilkinson's matrix: 1 on the diagonal and in the last column, −1 below the diagonal.
    // Partial pivoting does no row swaps and the last column doubles at every step.
    let mut table = String::from("| $n$ | growth factor $2^{n-1}$ | relative error, partial pivoting | relative error, random matrix of the same order |\n|---:|---:|---:|---:|\n");
    for n in [10usize, 20, 30, 40, 50, 60] {
        let w = Array2::from_shape_fn((n, n), |(i, j)| if i == j || j == n - 1 { 1.0 } else if i > j { -1.0 } else { 0.0 });
        let x_true = Array1::from_shape_fn(n, |i| ((i + 1) as f64).sin());
        let err = |a: &Array2<f64>| {
            let b = a.dot(&x_true);
            let x = a.solve(&b).unwrap();
            inf_norm(&(&x - &x_true)) / inf_norm(&x_true)
        };
        let random = reference::random_matrix(n, n, 5);
        table.push_str(&format!("| {n} | {:.1e} | {:.1e} | {:.1e} |\n", 2f64.powi(n as i32 - 1), err(&w), err(&random)));
    }
    write(CHAPTER, "wilkinson_growth.md", &table);
}

fn conditioning_figure() {
    // Hilbert matrices H_ij = 1/(i + j + 1): the condition number and the error of solving H x = H·1.
    let mut plot = Plot::new("Solving Hilbert systems H x = b with LU and partial pivoting", "order n", "");
    plot.log_y = true;
    plot.legend = Legend::TopLeft;
    plot.y_range = Some((1e-17, 1e20));
    let (mut cond, mut err, mut bound, mut resid) = (vec![], vec![], vec![], vec![]);
    for n in 2..=16usize {
        let h = Array2::from_shape_fn((n, n), |(i, j)| 1.0 / (i + j + 1) as f64);
        let ones = Array1::from_elem(n, 1.0);
        let b = h.dot(&ones);
        let x = h.solve(&b).unwrap();
        let k = inf_norm_matrix(&h) * inf_norm_matrix(&h.inv().unwrap());
        cond.push((n as f64, k));
        bound.push((n as f64, k * f64::EPSILON));
        err.push((n as f64, (inf_norm(&(&x - &ones)) / 1.0).max(1e-17)));
        resid.push((n as f64, (inf_norm(&(h.dot(&x) - &b)) / (inf_norm_matrix(&h) * inf_norm(&x))).max(1e-17)));
    }
    plot.series.push(Series::new("condition number κ∞(H)", cond, Style::LineMarkers, 0));
    plot.series.push(Series::new("κ∞ ε, the expected error", bound, Style::Dashed, 0));
    plot.series.push(Series::new("forward error ‖x − 1‖∞", err, Style::LineMarkers, 1));
    plot.series.push(Series::new("backward error ‖b − Hx‖ / (‖H‖ ‖x‖)", resid, Style::LineMarkers, 2));
    write(CHAPTER, "hilbert.svg", &plot.render());
}

fn speed_figure() {
    let mut plot = Plot::new("LU and Cholesky factorisation speed (one core)", "order n", "GFLOP/s");
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::BottomRight;
    plot.x_range = Some((24.0, 1400.0));
    plot.y_range = Some((0.3, 50.0));
    let sizes = [32usize, 64, 128, 256, 512, 1024];
    let (mut ref_lu, mut prod_lu, mut ref_ch, mut prod_ch) = (vec![], vec![], vec![], vec![]);
    for &n in &sizes {
        let a = reference::random_matrix(n, n, 7);
        let spd = reference::random_positive_definite(n, 8);
        let lu_flops = 2.0 / 3.0 * (n as f64).powi(3);
        let ch_flops = 1.0 / 3.0 * (n as f64).powi(3);
        if n <= 512 {
            ref_lu.push((n as f64, lu_flops / time(|| { std::hint::black_box(reference::lu::lu(&a).unwrap()); }) / 1e9));
            ref_ch.push((n as f64, ch_flops / time(|| { std::hint::black_box(reference::cholesky::cholesky(&spd).unwrap()); }) / 1e9));
        }
        prod_lu.push((n as f64, lu_flops / time(|| { std::hint::black_box(lib_jsl_linalg::Factorize::factorize(&a).unwrap()); }) / 1e9));
        prod_ch.push((n as f64, ch_flops / time(|| { std::hint::black_box(spd.cholesky(UPLO::Lower).unwrap()); }) / 1e9));
    }
    plot.series.push(Series::new("LU, production (recursive, blocked)", prod_lu, Style::LineMarkers, 0));
    plot.series.push(Series::new("LU, reference (Doolittle)", ref_lu, Style::Dashed, 0));
    plot.series.push(Series::new("Cholesky, production (blocked)", prod_ch, Style::LineMarkers, 1));
    plot.series.push(Series::new("Cholesky, reference (Banachiewicz)", ref_ch, Style::Dashed, 1));
    write(CHAPTER, "factorisation_speed.svg", &plot.render());
}

fn cholesky_table() {
    let mut table = String::from("| matrix | Cholesky | LU |\n|---|---|---|\n");
    let rows: [(&str, Array2<f64>); 4] = [
        ("$\\begin{pmatrix}4 & 2\\\\ 2 & 3\\end{pmatrix}$, positive definite", array![[4.0, 2.0], [2.0, 3.0]]),
        ("$\\begin{pmatrix}1 & 2\\\\ 2 & 1\\end{pmatrix}$, indefinite (eigenvalues 3, −1)", array![[1.0, 2.0], [2.0, 1.0]]),
        ("$\\begin{pmatrix}1 & 1\\\\ 1 & 1\\end{pmatrix}$, semidefinite (singular)", array![[1.0, 1.0], [1.0, 1.0]]),
        ("Hilbert matrix of order 12, positive definite with $\\kappa \\approx 10^{16}$", Array2::from_shape_fn((12, 12), |(i, j)| 1.0 / (i + j + 1) as f64)),
    ];
    for (name, a) in rows {
        let ch = match a.cholesky(UPLO::Lower) {
            Ok(_) => "succeeds".to_string(),
            Err(e) => format!("fails: {e}"),
        };
        let lu = match lib_jsl_linalg::Factorize::factorize(&a) {
            Ok(_) => "succeeds".to_string(),
            Err(e) => format!("fails: {e}"),
        };
        table.push_str(&format!("| {name} | {ch} | {lu} |\n"));
    }
    write(CHAPTER, "cholesky_cases.md", &table);
}

pub fn generate() {
    pivoting_tables();
    conditioning_figure();
    cholesky_table();
    speed_figure();
}
