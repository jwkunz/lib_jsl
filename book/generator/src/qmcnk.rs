//! Tables and figures for release 2.3.0: quasi-Monte Carlo sampling (random chapter) and
//! Jacobian-free Newton-Krylov (root finding chapter), compared with SciPy.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_linalg::sparse::{CooMatrix, LinearOperator, Ordering, SparseLu};
use lib_jsl_random::qmc::{Discrepancy, discrepancy, latin_hypercube};
use lib_jsl_random::quasi_random::{HaltonSequence, SobolSequence};
use lib_jsl_random::uniform_generator::{UniformGenerator, UniformRNG};
use lib_jsl_root_finding::{InnerSolver, NewtonKrylovOptions, newton_krylov};

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/qmcnk_reference.txt"));
    sequences_table(&refs);
    randomised(&refs);
    newton_krylov_results(&refs);
}

fn sobol_plain(d: usize, n: usize) -> Vec<Vec<f64>> {
    let mut s = SobolSequence::new(d).unwrap();
    (0..n).map(|_| s.next_point()).collect()
}

fn halton_plain(d: usize, n: usize) -> Vec<Vec<f64>> {
    let mut s = HaltonSequence::new(d).unwrap();
    (0..n).map(|_| s.next_point()).collect()
}

fn max_diff(got: &[Vec<f64>], want: &[f64]) -> f64 {
    let flat: Vec<f64> = got.iter().flatten().copied().collect();
    assert_eq!(flat.len(), want.len());
    flat.iter().zip(want).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
}

fn sequences_table(refs: &Refs) {
    let mut rows = vec![
        vec!["Sobol, 1111 dimensions, unscrambled".into(), "first 64 points".into(), fmt_diff(max_diff(&sobol_plain(1111, 64), refs.values("sobol plain")))],
        vec!["Halton, 20 dimensions, unscrambled".into(), "points 1 to 64".into(), fmt_diff(max_diff(&halton_plain(20, 64), refs.values("halton plain")))],
    ];
    let formula: Vec<Vec<f64>> = (0..50).map(|i| (0..6).map(|k| ((i as f64 * 1.7 + k as f64 * 0.3).sin() + 1.0) / 2.0).collect()).collect();
    let grid: Vec<Vec<f64>> = (0..64).map(|k| vec![((k / 8) as f64 + 0.5) / 8.0, ((k % 8) as f64 + 0.5) / 8.0]).collect();
    let sets = [
        ("sobol 128x5", "128 Sobol points in 5 dimensions", sobol_plain(5, 128)),
        ("halton 100x3", "100 Halton points in 3 dimensions", halton_plain(3, 100)),
        ("grid 64x2", "8 × 8 grid of cell centres", grid),
        ("formula 50x6", "50 points of a smooth formula in 6 dimensions", formula),
    ];
    for (key, label, pts) in sets {
        let want = refs.values(&format!("disc {key}"));
        let methods = [Discrepancy::Centered, Discrepancy::WrapAround, Discrepancy::Mixture, Discrepancy::L2Star];
        let worst = methods.iter().zip(want).map(|(&m, &w)| ((discrepancy(&pts, m).unwrap() - w) / w).abs()).fold(0.0, f64::max);
        let values = methods.iter().map(|&m| format!("{:.3e}", discrepancy(&pts, m).unwrap())).collect::<Vec<_>>().join(", ");
        rows.push(vec![format!("discrepancies CD, WD, MD, L²-star: {label}"), values, fmt_diff(worst)]);
    }
    write("random", "qmc_scipy.md", &md_table(&["comparison", "library", "largest (relative) difference from SciPy"], &rows));
}

/// Sobol's g-function in 5 dimensions, `Π (|4x_k − 2| + k)/(1 + k)`, whose integral is 1.
fn gfun(x: &[f64]) -> f64 {
    x.iter().enumerate().map(|(k, &v)| ((4.0 * v - 2.0).abs() + (k + 1) as f64) / (2.0 + k as f64)).product()
}

fn draw(name: &str, n: usize, rng: &mut UniformRNG) -> Vec<Vec<f64>> {
    let d = 5;
    match name {
        "mc" => (0..n).map(|_| (0..d).map(|_| rng.next_f64()).collect()).collect(),
        "lhs" => latin_hypercube(n, d, false, rng),
        "sobol" => {
            let mut s = SobolSequence::scrambled(d, rng).unwrap();
            (0..n).map(|_| s.next_point()).collect()
        }
        "halton" => {
            let mut s = HaltonSequence::scrambled(d, rng).unwrap();
            (0..n).map(|_| s.next_point()).collect()
        }
        _ => unreachable!(),
    }
}

fn randomised(refs: &Refs) {
    let reps = 50;
    let names = [("mc", "pseudo-random (Monte Carlo)"), ("lhs", "Latin hypercube"), ("halton", "scrambled Halton"), ("sobol", "scrambled Sobol")];
    let mut rng = UniformRNG::from_seed(2024);
    let mut plot = Plot::new("Integrating Sobol's g-function in 5 dimensions: RMS error over 50 randomisations", "number of points", "RMS error");
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::BottomLeft;
    plot.y_range = Some((1e-6, 1.0));
    let mut rows = Vec::new();
    for (k, (key, label)) in names.into_iter().enumerate() {
        let mut pts = Vec::new();
        for m in 4..13 {
            let n = 1usize << m;
            let sq: f64 = (0..reps).map(|_| (draw(key, n, &mut rng).iter().map(|p| gfun(p)).sum::<f64>() / n as f64 - 1.0).powi(2)).sum();
            pts.push((n as f64, (sq / reps as f64).sqrt()));
        }
        plot.series.push(Series::new(label, pts, Style::LineMarkers, k));
        let scipy: Vec<(f64, f64)> = refs.values(&format!("rmse {key}")).iter().enumerate().map(|(i, &e)| ((1usize << (i + 4)) as f64, e)).collect();
        plot.series.push(Series::new("", scipy, Style::Markers, k));
        let cd: Vec<f64> = (0..reps).map(|_| discrepancy(&draw(key, 256, &mut rng), Discrepancy::Centered).unwrap()).collect();
        let mean = cd.iter().sum::<f64>() / reps as f64;
        let sd = (cd.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / reps as f64).sqrt();
        let w = refs.values(&format!("cd {key}"));
        rows.push(vec![label.to_string(), format!("{mean:.2e} ± {sd:.1e}"), format!("{:.2e} ± {:.1e}", w[0], w[1])]);
    }
    write("random", "qmc_rmse.svg", &plot.render());
    write("random", "qmc_discrepancy.md", &md_table(&["256 points in 5 dimensions", "centered discrepancy, library (mean ± sd)", "SciPy"], &rows));
}

/// The Bratu residual `Δu + λ eᵘ` on an `m × m` interior grid of the unit square.
fn bratu(m: usize, lambda: f64) -> impl FnMut(&[f64]) -> Vec<f64> {
    let h2 = 1.0 / ((m + 1) * (m + 1)) as f64;
    move |u: &[f64]| {
        let at = |i: isize, j: isize| if i < 0 || j < 0 || i >= m as isize || j >= m as isize { 0.0 } else { u[i as usize * m + j as usize] };
        (0..m * m)
            .map(|k| {
                let (i, j) = ((k / m) as isize, (k % m) as isize);
                (at(i - 1, j) + at(i + 1, j) + at(i, j - 1) + at(i, j + 1) - 4.0 * at(i, j)) / h2 + lambda * at(i, j).exp()
            })
            .collect()
    }
}

/// Solves with the LU factors of the discrete Laplacian: a preconditioner for Bratu's Jacobian.
struct LaplacianSolve(SparseLu, usize);

impl LinearOperator for LaplacianSolve {
    fn shape(&self) -> (usize, usize) {
        (self.1, self.1)
    }
    fn matvec(&self, x: &[f64]) -> Vec<f64> {
        self.0.solve(x).unwrap()
    }
    fn rmatvec(&self, _: &[f64]) -> Vec<f64> {
        unimplemented!()
    }
}

fn laplacian_lu(m: usize) -> LaplacianSolve {
    let h2 = 1.0 / ((m + 1) * (m + 1)) as f64;
    let n = m * m;
    let mut coo = CooMatrix::new(n, n);
    for i in 0..m {
        for j in 0..m {
            let r = i * m + j;
            coo.push(r, r, -4.0 / h2);
            if i > 0 {
                coo.push(r, r - m, 1.0 / h2);
            }
            if i + 1 < m {
                coo.push(r, r + m, 1.0 / h2);
            }
            if j > 0 {
                coo.push(r, r - 1, 1.0 / h2);
            }
            if j + 1 < m {
                coo.push(r, r + 1, 1.0 / h2);
            }
        }
    }
    LaplacianSolve(SparseLu::factor(&coo.to_csr().unwrap(), Ordering::MinimumDegree).unwrap(), n)
}

fn newton_krylov_results(refs: &Refs) {
    let mut rows = Vec::new();
    let mut plot = Plot::new("Newton-Krylov on the Bratu problem, 64 × 64 grid", "Newton iteration", "max |F(x)|");
    plot.log_y = true;
    plot.legend = Legend::TopRight;
    for m in [16, 32, 64] {
        let reference = refs.get(&format!("nk {m} lgmres"));
        let peak_ref = reference[0][4];
        for (k, (key, label, inner, precond)) in [("gmres", "GMRES(20)", InnerSolver::Gmres, false), ("bicgstab", "BiCGSTAB", InnerSolver::Bicgstab, false), ("gmres lu", "GMRES(20), Laplacian LU preconditioner", InnerSolver::Gmres, true)].into_iter().enumerate() {
            let o = NewtonKrylovOptions { inner, ..Default::default() };
            let pre = precond.then(|| laplacian_lu(m));
            let r = newton_krylov(bratu(m, 6.0), &vec![0.0; m * m], pre.as_ref().map(|p| p as &dyn LinearOperator), &o).unwrap();
            assert!(r.converged, "{m} {key}");
            if m == 64 {
                // The history of max |F| over the Newton iterates, by rerunning with the iteration
                // count capped (the solver has no callback).
                let mut history = Vec::new();
                for it in 0..r.iterations {
                    let oi = NewtonKrylovOptions { max_iter: Some(it), f_tol: 0.0, ..o };
                    let ri = newton_krylov(bratu(m, 6.0), &vec![0.0; m * m], pre.as_ref().map(|p| p as &dyn LinearOperator), &oi).unwrap();
                    history.push((it as f64, ri.fun.iter().fold(0.0f64, |a, v| a.max(v.abs()))));
                }
                plot.series.push(Series::new(&format!("{label} (library)"), history, Style::LineMarkers, k));
                let w = refs.get(&format!("nk {m} {key}"));
                let spts: Vec<(f64, f64)> = w[1].iter().enumerate().map(|(i, &v)| (i as f64, v)).collect();
                plot.series.push(Series::new(&format!("{label} (SciPy)"), spts, Style::Markers, k));
            }
            let w = refs.get(&format!("nk {m} {key}"));
            let peak = r.x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let fmax = r.fun.iter().fold(0.0f64, |a, v| a.max(v.abs()));
            rows.push(vec![
                format!("{m} × {m}"),
                label.to_string(),
                format!("{} / {}", r.iterations - 1, w[0][0]),
                format!("{} / {}", r.nfev, w[0][1]),
                format!("{fmax:.0e} / {:.0e}", w[0][3]),
                fmt_diff((peak - peak_ref).abs() / peak_ref),
            ]);
        }
    }
    write("root_finding", "newton_krylov_scipy.md", &md_table(&["grid", "inner solver", "Newton iterations (library / SciPy)", "evaluations of F (library / SciPy)", "final max \\|F\\| (library / SciPy)", "peak of u, relative difference from SciPy's LGMRES solution"], &rows));
    write("root_finding", "newton_krylov_history.svg", &plot.render());
}
