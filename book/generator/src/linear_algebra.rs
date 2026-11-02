//! Figures and tables for the dense linear algebra chapters.

use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_linalg::{Cholesky, Eig, EigVals, EigValsh, Eigh, Inverse, LeastSquaresSvd, SVD, Solve, SvdJacobi, UPLO};
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

/// An orthonormal basis of the columns of `a` (Gram-Schmidt twice, which is enough).
pub(crate) fn orthonormal(a: Array2<f64>) -> Array2<f64> {
    let (m, n) = a.dim();
    let mut q = Array2::<f64>::zeros((m, n));
    for j in 0..n {
        let mut v = a.column(j).to_owned();
        for _ in 0..2 {
            for i in 0..j {
                let r = q.column(i).dot(&v);
                v.scaled_add(-r, &q.column(i));
            }
        }
        let norm = v.dot(&v).sqrt();
        q.column_mut(j).assign(&(v / norm));
    }
    q
}

/// An m × n matrix U diag(σ) Vᵀ with singular values spread geometrically from 1 to 1/κ.
pub(crate) fn with_condition(m: usize, n: usize, kappa: f64, seed: u64) -> Array2<f64> {
    let u = orthonormal(reference::random_matrix(m, n, seed));
    let v = orthonormal(reference::random_matrix(n, n, seed + 1));
    let sigma = Array1::from_shape_fn(n, |i| kappa.powf(-(i as f64) / (n - 1) as f64));
    u.dot(&Array2::from_diag(&sigma)).dot(&v.t())
}

fn orthogonality_figure() {
    let mut plot = Plot::new("Loss of orthogonality in QR factorisation (60 × 20 matrices)", "condition number κ(A)", "‖QᵀQ − I‖");
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::TopLeft;
    plot.y_range = Some((1e-17, 1e2));
    let (mut cgs, mut mgs, mut hh, mut k2, mut k1) = (vec![], vec![], vec![], vec![], vec![]);
    for e in 0..=32 {
        let kappa = 10f64.powf(e as f64 / 2.0);
        let a = with_condition(60, 20, kappa, 11);
        let floor = |v: f64| v.max(1e-17);
        cgs.push((kappa, floor(reference::qr::orthogonality_error(&reference::qr::classical_gram_schmidt(&a).0))));
        mgs.push((kappa, floor(reference::qr::orthogonality_error(&reference::qr::modified_gram_schmidt(&a).0))));
        hh.push((kappa, floor(reference::qr::orthogonality_error(&reference::qr::householder(&a).0))));
        k2.push((kappa, (kappa * kappa * f64::EPSILON).min(1e2)));
        k1.push((kappa, kappa * f64::EPSILON));
    }
    plot.series.push(Series::new("classical Gram-Schmidt", cgs, Style::LineMarkers, 0));
    plot.series.push(Series::new("modified Gram-Schmidt", mgs, Style::LineMarkers, 1));
    plot.series.push(Series::new("Householder reflections", hh, Style::LineMarkers, 2));
    plot.series.push(Series::new("κ²ε", k2, Style::Dashed, 0));
    plot.series.push(Series::new("κε", k1, Style::Dashed, 1));
    write(CHAPTER, "orthogonality.svg", &plot.render());
}

fn least_squares_figure() {
    // Fit a polynomial of degree d with all coefficients 1, on 50 points in [0, 1], in the monomial basis.
    let mut plot = Plot::new("Polynomial fitting: normal equations against QR", "degree d", "relative error in the coefficients");
    plot.log_y = true;
    plot.legend = Legend::TopLeft;
    plot.y_range = Some((1e-17, 1e3));
    let (mut ne, mut qr, mut prod, mut kappa, mut kappa2) = (vec![], vec![], vec![], vec![], vec![]);
    for d in 1..=15usize {
        let a = Array2::from_shape_fn((50, d + 1), |(i, j)| (i as f64 / 49.0).powi(j as i32));
        let exact = Array1::from_elem(d + 1, 1.0);
        let b = a.dot(&exact);
        let err = |x: &Array1<f64>| (inf_norm(&(x - &exact)) / inf_norm(&exact)).max(1e-17);
        if let Ok(x) = reference::least_squares::normal_equations(&a, &b) {
            ne.push((d as f64, err(&x).min(1e3)));
        }
        qr.push((d as f64, err(&reference::least_squares::qr_least_squares(&a, &b).unwrap())));
        let r = a.least_squares(&b).unwrap();
        prod.push((d as f64, err(&r.solution)));
        let k = r.singular_values[0] / r.singular_values[d];
        kappa.push((d as f64, k * f64::EPSILON));
        kappa2.push((d as f64, (k * k * f64::EPSILON).min(1e3)));
    }
    plot.series.push(Series::new("normal equations (Cholesky of AᵀA)", ne, Style::LineMarkers, 0));
    plot.series.push(Series::new("Householder QR (reference)", qr, Style::LineMarkers, 1));
    plot.series.push(Series::new("pivoted QR (production least_squares)", prod, Style::LineMarkers, 2));
    plot.series.push(Series::new("κ²ε", kappa2, Style::Dashed, 0));
    plot.series.push(Series::new("κε", kappa, Style::Dashed, 1));
    write(CHAPTER, "least_squares.svg", &plot.render());
}

fn rank_deficient_table() {
    let fmt = |v: &Array1<f64>| format!("({})", v.iter().map(|x| format!("{:.4}", x + 0.0)).collect::<Vec<_>>().join(", "));
    let mut table = String::from("| system | rank found | solution | singular values |\n|---|---:|---|---|\n");
    let cases: [(&str, Array2<f64>, Array1<f64>); 3] = [
        ("full rank, 4 × 2: fit $y = c_0 + c_1t$ to $(0, 1), (1, 3), (2, 4), (3, 7)$", array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], array![1.0, 3.0, 4.0, 7.0]),
        ("rank 1, 3 × 2: the second column is twice the first", array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], array![1.0, 2.0, 3.0]),
        ("underdetermined, 1 × 3: $x_1 + x_2 + x_3 = 3$", array![[1.0, 1.0, 1.0]], array![3.0]),
    ];
    for (name, a, b) in cases {
        let r = a.least_squares(&b).unwrap();
        table.push_str(&format!("| {name} | {} | {} | {} |\n", r.rank, fmt(&r.solution), fmt(&r.singular_values)));
    }
    write(CHAPTER, "rank_deficient.md", &table);
}

/// A symmetric matrix Q diag(λ) Qᵀ with a random orthogonal Q.
fn symmetric_with_spectrum(lambda: &[f64], seed: u64) -> Array2<f64> {
    let n = lambda.len();
    let q = orthonormal(reference::random_matrix(n, n, seed));
    q.dot(&Array2::from_diag(&Array1::from(lambda.to_vec()))).dot(&q.t())
}

fn eigen_iteration_tables() {
    use reference::power_iteration::{inverse_iteration, power_iteration, rayleigh_quotient_iteration};
    let mut table = String::from("| spectrum of the $30 \\times 30$ matrix | power iteration | inverse iteration, shift 10.2 | Rayleigh quotient iteration |\n|---|---:|---:|---:|\n");
    for (name, top) in [("$10, 5, \\ldots$ (ratio 0.5)", 5.0), ("$10, 9, \\ldots$ (ratio 0.9)", 9.0), ("$10, 9.9, \\ldots$ (ratio 0.99)", 9.9)] {
        let mut lambda: Vec<f64> = (0..28).map(|i| 4.0 * i as f64 / 27.0 - 2.0).collect();
        lambda.push(top);
        lambda.push(10.0);
        let a = symmetric_with_spectrum(&lambda, 21);
        let start = Array1::from_shape_fn(30, |i| 1.0 + 0.1 * (i as f64).sin());
        let cell = |r: Result<reference::power_iteration::EigenEstimate, &str>| match r {
            Ok(e) => format!("{} (λ = {:.10})", e.iterations, e.value),
            Err(_) => "no convergence in 5000".to_string(),
        };
        table.push_str(&format!(
            "| {name} | {} | {} | {} |\n",
            cell(power_iteration(&a, &start, 1e-10, 5000)),
            cell(inverse_iteration(&a, 10.2, &start, 1e-10, 5000)),
            cell(rayleigh_quotient_iteration(&a, &start, 1e-10, 5000))
        ));
    }
    write(CHAPTER, "power_iteration.md", &table);

    let mut table = String::from("| order $n$ | Jacobi sweeps | unshifted QR steps | Wilkinson-shifted QR steps | steps per eigenvalue (shifted) |\n|---:|---:|---:|---:|---:|\n");
    for n in [10usize, 20, 40, 80] {
        let a = reference::random_symmetric(n, 31);
        let jac = reference::jacobi_eigenvalue::jacobi_eigen(&a, 1e-14).unwrap();
        let unshifted = match reference::qr_algorithm::unshifted_qr(&a, 1e-12, 20_000) {
            Ok(r) => format!("{}", r.iterations),
            Err(_) => "> 20 000".to_string(),
        };
        let shifted = reference::qr_algorithm::shifted_qr(&a, 1e-14, 20_000).unwrap();
        table.push_str(&format!("| {n} | {} | {unshifted} | {} | {:.2} |\n", jac.sweeps, shifted.iterations, shifted.iterations as f64 / n as f64));
    }
    write(CHAPTER, "jacobi_qr.md", &table);
}

fn eigen_accuracy_and_speed() {
    let frob = |a: &Array2<f64>| a.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut table = String::from("| order $n$ | symmetric: $\\|AV - V\\Lambda\\|_F / \\|A\\|_F$ | symmetric: $\\|V^{\\mathsf T}V - I\\|_F$ | general: $\\|AV - V\\Lambda\\|_F / \\|A\\|_F$ |\n|---:|---:|---:|---:|\n");
    for n in [50usize, 200, 500] {
        let s = reference::random_symmetric(n, 41);
        let (w, v) = s.eigh(UPLO::Lower).unwrap();
        let res = frob(&(s.dot(&v) - &v * &w)) / frob(&s);
        let orth = frob(&(v.t().dot(&v) - Array2::<f64>::eye(n)));
        let g = reference::random_matrix(n, n, 42);
        let (wc, vc) = g.eig().unwrap();
        let gc = g.mapv(|x| num_complex::Complex::new(x, 0.0));
        let r = gc.dot(&vc) - &vc * &wc;
        let resg = r.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt() / frob(&g);
        table.push_str(&format!("| {n} | {res:.1e} | {orth:.1e} | {resg:.1e} |\n"));
    }
    write(CHAPTER, "eigen_accuracy.md", &table);

    let mut plot = Plot::new("Time for all eigenvalues of a symmetric matrix (one core)", "order n", "seconds");
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::TopLeft;
    let (mut jac, mut qr, mut prod, mut prodv, mut general) = (vec![], vec![], vec![], vec![], vec![]);
    for n in [16usize, 32, 64, 128, 256, 512] {
        let a = reference::random_symmetric(n, 51);
        if n <= 128 {
            jac.push((n as f64, time(|| { std::hint::black_box(reference::jacobi_eigenvalue::jacobi_eigen(&a, 1e-14).unwrap()); })));
            qr.push((n as f64, time(|| { std::hint::black_box(reference::qr_algorithm::shifted_qr(&a, 1e-14, 100_000).unwrap()); })));
        }
        prod.push((n as f64, time(|| { std::hint::black_box(lib_jsl_linalg::EigValsh::eigvalsh(&a, UPLO::Lower).unwrap()); })));
        prodv.push((n as f64, time(|| { std::hint::black_box(a.eigh(UPLO::Lower).unwrap()); })));
        general.push((n as f64, time(|| { std::hint::black_box(a.eigvals().unwrap()); })));
    }
    plot.series.push(Series::new("reference: cyclic Jacobi (with vectors)", jac, Style::Dashed, 0));
    plot.series.push(Series::new("reference: shifted QR on the full matrix", qr, Style::Dashed, 1));
    plot.series.push(Series::new("production eigvalsh: tridiagonal + QL", prod, Style::LineMarkers, 2));
    plot.series.push(Series::new("production eigh: with eigenvectors", prodv, Style::LineMarkers, 3));
    plot.series.push(Series::new("production eigvals: general (Hessenberg + Francis QR)", general, Style::LineMarkers, 4));
    write(CHAPTER, "eigen_speed.svg", &plot.render());
}

fn nonnormal_figure() {
    // The Grcar matrix: −1 on the subdiagonal, 1 on the diagonal and the first three superdiagonals.
    let n = 100;
    let g = Array2::from_shape_fn((n, n), |(i, j)| if i == j + 1 { -1.0 } else if j >= i && j <= i + 3 { 1.0 } else { 0.0 });
    let sym = &g + &g.t();
    let mut plot = Plot::new("Eigenvalues of the 100 × 100 Grcar matrix and of perturbations of size 10⁻¹⁰", "Re λ", "Im λ");
    plot.legend = Legend::TopLeft;
    plot.x_range = Some((-1.5, 3.5));
    plot.y_range = Some((-4.0, 5.0));
    let mut perturbed = vec![];
    for seed in 0..20u64 {
        let e = reference::random_matrix(n, n, 100 + seed) * 1e-10;
        perturbed.extend((&g + &e).eigvals().unwrap().iter().map(|z| (z.re, z.im)));
    }
    plot.series.push(Series::new("20 perturbations of size 10⁻¹⁰", perturbed, Style::Markers, 7));
    plot.series.push(Series::new("computed eigenvalues of A", g.eigvals().unwrap().iter().map(|z| (z.re, z.im)).collect(), Style::Markers, 1));
    let mut sym_pts = vec![];
    for seed in 0..20u64 {
        let e = reference::random_symmetric(n, 200 + seed) * 0.5e-10;
        sym_pts.extend((&sym + &e).eigvalsh(UPLO::Lower).unwrap().iter().map(|&x| (x / 2.0, -3.0)));
    }
    plot.series.push(Series::new("symmetric (A + Aᵀ)/2 perturbed: eigenvalues stay put (shown at Im = −3)", sym_pts, Style::Markers, 0));
    write(CHAPTER, "grcar.svg", &plot.render());
}

const SVD_REFERENCE: &str = include_str!("../data/svd_reference.csv");

fn svd_accuracy_figure() {
    // A = B D with B random (40 × 20, well conditioned) and D = diag(2^−k), k = round(50 i / 19) in
    // the scrambled order i = 7 j mod 20: every entry is an exact double, and data/make_svd_ref.py
    // computes the singular values with mpmath at 50 digits.
    let (m, n) = (40, 20);
    let b = reference::random_matrix(m, n, 61);
    let d = Array1::from_shape_fn(n, |j| 2f64.powi(-((50.0 * ((7 * j) % n) as f64 / 19.0).round() as i32)));
    let a = &b * &d;
    let exact: Vec<f64> = SVD_REFERENCE.lines().filter(|l| !l.starts_with('#')).map(|l| l.parse().unwrap()).collect();
    let rel = |s: &[f64]| -> Vec<(f64, f64)> { exact.iter().zip(s).map(|(&t, &v)| (t, ((v - t) / t).abs().clamp(1e-17, 1e2))).collect() };
    let gk = a.svd(false, false).unwrap().1.to_vec();
    let jac = a.svd_jacobi(false, false).unwrap().1.to_vec();
    let naive = reference::naive_svd::svd_via_normal_matrix(&a).unwrap().singular_values.to_vec();
    let mut plot = Plot::new("Relative error of each singular value of a column-scaled matrix A = B D", "singular value σᵢ", "relative error");
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::TopRight;
    plot.y_range = Some((1e-17, 1e2));
    plot.series.push(Series::new("eigenvalues of AᵀA (reference)", rel(&naive), Style::LineMarkers, 0));
    plot.series.push(Series::new("Golub-Kahan (production svd)", rel(&gk), Style::LineMarkers, 1));
    plot.series.push(Series::new("one-sided Jacobi (production svd_jacobi)", rel(&jac), Style::LineMarkers, 2));
    plot.series.push(Series::new("ε σ₁ / σᵢ", exact.iter().map(|&t| (t, (f64::EPSILON * exact[0] / t).min(1e2))).collect(), Style::Dashed, 1));
    write(CHAPTER, "svd_accuracy.svg", &plot.render());
}

fn low_rank_figure() {
    // f(x, y) = 1 / (1 + x + y) on a 100 × 100 grid of [0, 1]²: a smooth function, numerically of low rank.
    let n = 100;
    let a = Array2::from_shape_fn((n, n), |(i, j)| 1.0 / (1.0 + i as f64 / 99.0 + j as f64 / 99.0) + 0.3 * ((6.0 * (i as f64 / 99.0)).sin() * (j as f64 / 99.0)).cos());
    let (u, s, vt) = a.svd(true, true).unwrap();
    let (u, vt) = (u.unwrap(), vt.unwrap());
    let norm2 = s[0];
    let mut errors = vec![];
    for k in 1..=30usize {
        let ak = u.slice(ndarray::s![.., ..k]).dot(&Array2::from_diag(&s.slice(ndarray::s![..k]).to_owned())).dot(&vt.slice(ndarray::s![..k, ..]));
        // The spectral norm of the error, as the largest singular value of A − A_k.
        let err = (&a - &ak).svd(false, false).unwrap().1[0];
        errors.push((k as f64, (err / norm2).max(1e-17)));
    }
    let mut plot = Plot::new("Best rank-k approximation of a sampled smooth function (Eckart-Young)", "rank k", "relative to σ₁");
    plot.log_y = true;
    plot.legend = Legend::TopRight;
    plot.y_range = Some((1e-17, 2.0));
    plot.series.push(Series::new("singular values σₖ₊₁ / σ₁", (1..=30).map(|k| (k as f64, (s[k] / norm2).max(1e-17))).collect(), Style::Markers, 0));
    plot.series.push(Series::new("‖A − Aₖ‖₂ / σ₁ for the truncated SVD Aₖ", errors, Style::Line, 1));
    write(CHAPTER, "low_rank.svg", &plot.render());
}

fn svd_speed_figure() {
    let mut plot = Plot::new("Time for the SVD with vectors of an n × n matrix (one core)", "order n", "seconds");
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::TopLeft;
    let (mut naive, mut gk, mut gkv, mut jac) = (vec![], vec![], vec![], vec![]);
    for n in [16usize, 32, 64, 128, 256, 512] {
        let a = reference::random_matrix(n, n, 71);
        if n <= 128 {
            naive.push((n as f64, time(|| { std::hint::black_box(reference::naive_svd::svd_via_normal_matrix(&a).unwrap()); })));
        }
        gk.push((n as f64, time(|| { std::hint::black_box(a.svd(false, false).unwrap()); })));
        gkv.push((n as f64, time(|| { std::hint::black_box(a.svd(true, true).unwrap()); })));
        jac.push((n as f64, time(|| { std::hint::black_box(a.svd_jacobi(true, true).unwrap()); })));
    }
    plot.series.push(Series::new("reference: Jacobi eigenvectors of AᵀA", naive, Style::Dashed, 0));
    plot.series.push(Series::new("Golub-Kahan, values only", gk, Style::LineMarkers, 1));
    plot.series.push(Series::new("Golub-Kahan, with vectors", gkv, Style::LineMarkers, 2));
    plot.series.push(Series::new("one-sided Jacobi, with vectors", jac, Style::LineMarkers, 3));
    write(CHAPTER, "svd_speed.svg", &plot.render());
}

pub fn generate() {
    pivoting_tables();
    conditioning_figure();
    cholesky_table();
    orthogonality_figure();
    least_squares_figure();
    rank_deficient_table();
    eigen_iteration_tables();
    nonnormal_figure();
    eigen_accuracy_and_speed();
    svd_accuracy_figure();
    low_rank_figure();
    svd_speed_figure();
    speed_figure();
}
