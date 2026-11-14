//! Tables and figures for the 1.18.0 additions to "Sparse matrices and iterative solvers": fill
//! under minimum degree, incomplete-LU preconditioning, `expm_multiply` and shift-invert `eigsh`.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_linalg::expm;
use lib_jsl_linalg::sparse::{CooMatrix, CsrMatrix, Ilu, IterativeOptions, Jacobi, LinearOperator, Ordering, SparseLu, eigsh_shift_invert, expm_multiply, gmres};
use ndarray::Array1;

const CHAPTER: &str = "linear_algebra";

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/sparse2_reference.txt"));
    fill_table(&refs);
    ilu_table(&refs);
    expm_and_eigen_table(&refs);
    fill_figure();
}

fn grid(m: usize, east: f64, west: f64, north: f64, south: f64) -> CsrMatrix {
    let mut coo = CooMatrix::new(m * m, m * m);
    for i in 0..m {
        for j in 0..m {
            let k = i * m + j;
            coo.push(k, k, 4.0);
            if j + 1 < m {
                coo.push(k, k + 1, east);
            }
            if j > 0 {
                coo.push(k, k - 1, west);
            }
            if i + 1 < m {
                coo.push(k, k + m, north);
            }
            if i > 0 {
                coo.push(k, k - m, south);
            }
        }
    }
    coo.to_csr().unwrap()
}

fn permute(a: &CsrMatrix, perm: &[usize]) -> CsrMatrix {
    let mut pos = vec![0; perm.len()];
    for (k, &v) in perm.iter().enumerate() {
        pos[v] = k;
    }
    let mut coo = CooMatrix::new(a.nrows, a.ncols);
    for i in 0..a.nrows {
        for q in a.indptr[i]..a.indptr[i + 1] {
            coo.push(pos[i], pos[a.indices[q]], a.data[q]);
        }
    }
    coo.to_csr().unwrap()
}

fn fill_table(refs: &Refs) {
    let mut rows = Vec::new();
    let mut add = |label: String, a: &CsrMatrix, key: &str| {
        let f = |o: Ordering| SparseLu::factor(a, o).unwrap().nnz().to_string();
        let s = refs.values(key);
        rows.push(vec![label, f(Ordering::Natural), f(Ordering::ReverseCuthillMcKee), f(Ordering::MinimumDegree), format!("{}", s[0]), format!("{}", s[1]), format!("{}", s[2])]);
    };
    for m in [30usize, 60] {
        let perm: Vec<usize> = refs.values(&format!("perm {m}")).iter().map(|&v| v as usize).collect();
        add(format!("shuffled 2-D Laplacian, n = {}", m * m), &permute(&grid(m, -1.0, -1.0, -1.0, -1.0), &perm), &format!("fill laplacian {m}"));
    }
    add("convection-diffusion, n = 1600".into(), &grid(40, -0.7, -1.3, -0.9, -1.1), "fill convdiff 40");
    write(CHAPTER, "sparse_md_fill.md", &md_table(&["matrix", "natural", "RCM", "minimum degree", "SuperLU natural", "SuperLU MMD on A + Aᵀ", "SuperLU COLAMD"], &rows));
}

fn ilu_table(refs: &Refs) {
    let n = 1600;
    let b: Vec<f64> = (0..n).map(|k| 1.0 + 0.5 * (0.37 * k as f64).sin()).collect();
    let opts = IterativeOptions { rtol: 1e-8, atol: 0.0, max_iterations: Some(2000 * 30) };
    let mut rows = Vec::new();
    for (name, label, a) in [("convdiff", "convection-diffusion (Péclet 0.3, 0.1)", grid(40, -0.7, -1.3, -0.9, -1.1)), ("hard", "convection-dominated (Péclet 1.4, 0.9)", grid(40, 0.4, -2.4, -0.1, -1.9))] {
        let rel = |x: &[f64]| {
            let r: Vec<f64> = a.matvec(x).iter().zip(&b).map(|(p, q)| q - p).collect();
            (r.iter().map(|v| v * v).sum::<f64>() / b.iter().map(|v| v * v).sum::<f64>()).sqrt()
        };
        let mut push = |what: String, m: Option<&dyn LinearOperator>, nnz: String| {
            let r = gmres(&a, &b, None, m, 30, opts).unwrap();
            rows.push(vec![label.to_string(), what, r.iterations.to_string(), nnz, fmt_diff(rel(&r.x))]);
        };
        push("none".into(), None, "—".into());
        let jac = Jacobi::new(&a).unwrap();
        push("Jacobi".into(), Some(&jac), n.to_string());
        let i0 = Ilu::ilu0(&a).unwrap();
        push("ILU(0)".into(), Some(&i0), i0.nnz().to_string());
        for (tau, p) in [(1e-2, 10), (1e-4, 30)] {
            let it = Ilu::ilut(&a, tau, p).unwrap();
            push(format!("ILUT(τ = {tau:e}, p = {p})"), Some(&it), it.nnz().to_string());
        }
        for (key, what) in [("none", "SciPy, none"), ("spilu 1e-2", "SciPy `spilu(drop_tol = 1e-2)`"), ("spilu 1e-4", "SciPy `spilu(drop_tol = 1e-4)`")] {
            let s = refs.values(&format!("gmres {name} {key}"));
            rows.push(vec![label.to_string(), what.to_string(), format!("{}", s[0]), s.get(3).map_or("—".into(), |v| format!("{v}")), fmt_diff(s[2])]);
        }
    }
    write(CHAPTER, "sparse_ilu.md", &md_table(&["matrix (n = 1600)", "preconditioner", "GMRES(30) iterations", "nnz of the factors", "final relative residual"], &rows));
}

fn rel_scaled(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    got.iter().zip(want).fold(0.0f64, |m, (g, w)| m.max((g - w).abs())) / scale
}

fn expm_and_eigen_table(refs: &Refs) {
    let mut rows = Vec::new();
    let lap = grid(30, -1.0, -1.0, -1.0, -1.0);
    let b: Vec<f64> = (0..900).map(|k| (0.11 * k as f64).cos()).collect();
    let neg = |a: &CsrMatrix, t: f64| CsrMatrix { data: a.data.iter().map(|v| -t * v).collect(), ..a.clone() };
    for t in [0.01, 1.0, 10.0, 40.0] {
        let got = expm_multiply(&neg(&lap, 1.0), &b, t).unwrap();
        let mut row = vec![format!("`expm_multiply`, heat equation e^(−tA) b, t = {t}"), fmt_diff(rel_scaled(&got, refs.values(&format!("expm heat {t}"))))];
        if t == 1.0 {
            let dense = expm(&neg(&lap, t).to_dense()).unwrap().dot(&Array1::from_vec(b.clone()));
            row[0] += &format!(" (dense `expm` times b: {})", fmt_diff(rel_scaled(&got, dense.as_slice().unwrap())));
        }
        rows.push(row);
    }
    let cd = grid(30, -0.7, -1.3, -0.9, -1.1);
    for t in [0.5, 3.0] {
        let got = expm_multiply(&neg(&cd, 1.0), &b, t).unwrap();
        rows.push(vec![format!("`expm_multiply`, convection-diffusion, t = {t}"), fmt_diff(rel_scaled(&got, refs.values(&format!("expm convdiff {t}"))))]);
    }
    for sigma in [2.0, 5.5] {
        let (vals, _) = eigsh_shift_invert(&lap, 6, sigma, 0.0).unwrap();
        rows.push(vec![format!("`eigsh` shift-invert, the 6 eigenvalues nearest σ = {sigma} (three double pairs)"), fmt_diff(rel_scaled(&vals, refs.values(&format!("eigsh sigma {sigma}"))))]);
    }
    write(CHAPTER, "sparse_expm_eigsh.md", &md_table(&["computation (SciPy reference), n = 900", "largest difference (relative)"], &rows));
}

/// Fill of the LU factors of the m × m Laplacian in its natural (banded) order, after RCM and
/// after minimum degree, against n = m².
fn fill_figure() {
    let (mut nat, mut rcm, mut md) = (Vec::new(), Vec::new(), Vec::new());
    for m in [10usize, 20, 30, 40, 60, 80] {
        let a = grid(m, -1.0, -1.0, -1.0, -1.0);
        let n = (m * m) as f64;
        nat.push((n, SparseLu::factor(&a, Ordering::Natural).unwrap().nnz() as f64));
        rcm.push((n, SparseLu::factor(&a, Ordering::ReverseCuthillMcKee).unwrap().nnz() as f64));
        md.push((n, SparseLu::factor(&a, Ordering::MinimumDegree).unwrap().nnz() as f64));
    }
    let mut plot = Plot::new("Fill of the LU factors of the 2-D Laplacian", "unknowns n = m²", "nnz(L) + nnz(U)");
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::TopLeft;
    let guide = |c: f64, p: f64| vec![(100.0, c * 100f64.powf(p)), (6400.0, c * 6400f64.powf(p))];
    plot.series.push(Series::new("natural (banded) order", nat, Style::LineMarkers, 1));
    plot.series.push(Series::new("reverse Cuthill-McKee", rcm, Style::LineMarkers, 4));
    plot.series.push(Series::new("minimum degree", md, Style::LineMarkers, 0));
    plot.series.push(Series::new("guides: 2n¹·⁵ and 3n ln n", guide(2.0, 1.5), Style::Dashed, 7));
    plot.series.push(Series::new("", (1..=20).map(|i| { let n = 100.0 * 64f64.powf((i - 1) as f64 / 19.0); (n, 3.0 * n * n.ln()) }).collect(), Style::Dashed, 7));
    write(CHAPTER, "sparse_fill_growth.svg", &plot.render());
}
