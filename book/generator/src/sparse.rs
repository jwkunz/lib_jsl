//! Tables and figures for "Sparse matrices and iterative solvers" (the linear algebra chapter):
//! the SciPy comparison, Krylov convergence curves, and the effect of ordering on bandwidth and
//! fill-in.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_linalg::sparse::{CooMatrix, CsrMatrix, IterativeOptions, IterativeResult, Jacobi, Ordering, SparseLu, Which, bandwidth, bicgstab, cg, eigsh, gmres, lsqr, minres, reverse_cuthill_mckee, spsolve, svds};

const CHAPTER: &str = "linear_algebra";
const M: usize = 30;

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/sparse_reference.txt"));
    scipy_reference(&refs);
    convergence_figures();
    ordering(&refs);
    spy_plots();
}

/// The 5-point stencil on an `m × m` grid (the same construction as `make_sparse_refs.py`).
fn grid(m: usize, east: f64, west: f64, north: f64, south: f64, extra: impl Fn(usize) -> f64) -> CsrMatrix {
    let mut coo = CooMatrix::new(m * m, m * m);
    for i in 0..m {
        for j in 0..m {
            let k = i * m + j;
            coo.push(k, k, 4.0 + extra(k));
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

fn poisson(m: usize) -> CsrMatrix {
    grid(m, -1.0, -1.0, -1.0, -1.0, |_| 0.0)
}

fn convdiff() -> CsrMatrix {
    grid(M, -1.0 + 0.3, -1.0 - 0.3, -1.0 + 0.1, -1.0 - 0.1, |_| 0.0)
}

fn varied() -> CsrMatrix {
    grid(M, -1.0, -1.0, -1.0, -1.0, |k| 50.0 * ((k * 13) % 17) as f64 / 17.0)
}

fn rhs() -> Vec<f64> {
    (0..M * M).map(|k| 1.0 + 0.5 * (0.37 * k as f64).sin()).collect()
}

fn rel(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    got.iter().zip(want).fold(0.0f64, |m, (a, b)| m.max((a - b).abs())) / scale
}

const OPTS: IterativeOptions = IterativeOptions { rtol: 1e-8, atol: 0.0, max_iterations: None };

fn scipy_reference(refs: &Refs) {
    let b = rhs();
    let (p, c, v) = (poisson(M), convdiff(), varied());
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row = |what: &str, d: f64| rows.push(vec![what.to_string(), fmt_diff(d)]);
    for (name, a, label) in [("poisson", &p, "2-D Laplacian"), ("convdiff", &c, "convection-diffusion"), ("varied", &v, "varying diagonal")] {
        row(&format!("`spsolve`, {label}"), rel(&spsolve(a, &b).unwrap(), refs.values(&format!("spsolve {name}"))));
    }
    let iters = |r: &IterativeResult, key: &str| format!("{} / {}", r.iterations, refs.values(key)[0]);
    // Iterative solutions are compared with the direct solution, for the library and for SciPy:
    // the solvers stop at slightly different points, so the two iterates need not agree closely.
    let mut solver_rows: Vec<Vec<String>> = Vec::new();
    let mut solver = |what: String, r: &IterativeResult, name: &str, key: &str| {
        let exact = refs.values(&format!("spsolve {name}"));
        let theirs = if key == "pcg" { "—".to_string() } else { fmt_diff(rel(refs.values(&format!("x {key} {name}")), exact)) };
        let it_key = if key == "pcg" { "iters pcg varied".to_string() } else { format!("iters {key} {name}") };
        solver_rows.push(vec![what, iters(r, &it_key), format!("{} / {}", fmt_diff(rel(&r.x, exact)), theirs)]);
    };
    for (name, a, label) in [("poisson", &p, "2-D Laplacian"), ("varied", &v, "varying diagonal")] {
        solver(format!("`cg`, {label}"), &cg(a, &b, None, None, OPTS).unwrap(), name, "cg");
        solver(format!("`minres`, {label}"), &minres(a, &b, None, OPTS).unwrap(), name, "minres");
    }
    let pre = Jacobi::new(&v).unwrap();
    solver("`cg` with Jacobi preconditioning, varying diagonal".into(), &cg(&v, &b, None, Some(&pre), OPTS).unwrap(), "varied", "pcg");
    for (name, a, label) in [("poisson", &p, "2-D Laplacian"), ("convdiff", &c, "convection-diffusion")] {
        solver(format!("`bicgstab`, {label}"), &bicgstab(a, &b, None, None, OPTS).unwrap(), name, "bicgstab");
        solver(format!("`gmres` (restart 30), {label}"), &gmres(a, &b, None, None, 30, OPTS).unwrap(), name, "gmres");
    }
    write(CHAPTER, "sparse_solvers.md", &md_table(&["solver, n = 900, rtol = 10⁻⁸", "iterations (library / SciPy)", "error against the direct solution (library / SciPy)"], &solver_rows));
    let t = refs.get("triplets rect");
    let rows_idx: Vec<usize> = t[2].iter().map(|&x| x as usize).collect();
    let cols_idx: Vec<usize> = t[3].iter().map(|&x| x as usize).collect();
    let rect = CsrMatrix::from_triplets(t[0][0] as usize, t[1][0] as usize, &rows_idx, &cols_idx, &t[4]).unwrap();
    let brect: Vec<f64> = (0..rect.nrows).map(|k| (0.21 * k as f64).cos()).collect();
    let r = lsqr(&rect, &brect, IterativeOptions { rtol: 1e-10, ..OPTS }).unwrap();
    row(&format!("`lsqr`, random 200 × 80 (5 % filled), {} / {} iterations, against SciPy", r.iterations, refs.values("iters lsqr rect")[0]), rel(&r.x, refs.values("x lsqr rect")));
    row("`lsqr`, the same, against the dense least-squares solution", rel(&r.x, refs.values("lstsq rect")));
    for (which, w, label) in [(Which::LargestAlgebraic, "LA", "largest"), (Which::SmallestAlgebraic, "SA", "smallest"), (Which::LargestMagnitude, "LM", "largest in magnitude")] {
        let (vals, _) = eigsh(&p, 4, which, 0.0).unwrap();
        row(&format!("`eigsh`, the 4 {label} eigenvalues of the 2-D Laplacian"), rel(&vals, refs.values(&format!("eigsh {w}"))));
    }
    let (s, _, _) = svds(&rect, 4, 0.0).unwrap();
    row("`svds`, the 4 largest singular values of the 200 × 80 matrix", rel(&s, refs.values("svds rect")));
    write(CHAPTER, "sparse_scipy.md", &md_table(&["computation (n = 900 unless stated)", "largest difference from SciPy (relative)"], &rows));
}

fn history(label: &str, r: &IterativeResult, color: usize) -> Series {
    let r0 = r.residual_history[0];
    // For GMRES the history holds one value per inner iteration, the others one per iteration.
    Series::new(label, r.residual_history.iter().enumerate().map(|(i, &v)| (i as f64, v / r0)).collect(), Style::Line, color)
}

fn convergence_figures() {
    let b = rhs();
    let p = poisson(M);
    let mut plot = Plot::new("Krylov solvers on the 2-D Laplacian (n = 900)", "iteration", "relative residual ‖r‖ / ‖r₀‖");
    plot.log_y = true;
    plot.legend = Legend::TopRight;
    plot.series.push(history("CG", &cg(&p, &b, None, None, OPTS).unwrap(), 0));
    plot.series.push(history("MINRES", &minres(&p, &b, None, OPTS).unwrap(), 1));
    plot.series.push(history("BiCGSTAB", &bicgstab(&p, &b, None, None, OPTS).unwrap(), 2));
    plot.series.push(history("GMRES(30)", &gmres(&p, &b, None, None, 30, OPTS).unwrap(), 3));
    plot.series.push(history("GMRES (no restart)", &gmres(&p, &b, None, None, 900, OPTS).unwrap(), 4));
    write(CHAPTER, "sparse_krylov.svg", &plot.render());

    let c = convdiff();
    let mut plot = Plot::new("A nonsymmetric system: convection-diffusion (n = 900)", "iteration", "relative residual ‖r‖ / ‖r₀‖");
    plot.log_y = true;
    plot.series.push(history("BiCGSTAB", &bicgstab(&c, &b, None, None, OPTS).unwrap(), 2));
    plot.series.push(history("GMRES(10)", &gmres(&c, &b, None, None, 10, OPTS).unwrap(), 5));
    plot.series.push(history("GMRES(30)", &gmres(&c, &b, None, None, 30, OPTS).unwrap(), 3));
    plot.series.push(history("GMRES (no restart)", &gmres(&c, &b, None, None, 900, OPTS).unwrap(), 4));
    write(CHAPTER, "sparse_nonsymmetric.svg", &plot.render());

    let v = varied();
    let pre = Jacobi::new(&v).unwrap();
    let mut plot = Plot::new("Jacobi preconditioning on a varying diagonal (n = 900)", "iteration", "relative residual ‖r‖ / ‖r₀‖");
    plot.log_y = true;
    plot.series.push(history("CG", &cg(&v, &b, None, None, OPTS).unwrap(), 0));
    plot.series.push(history("CG + Jacobi", &cg(&v, &b, None, Some(&pre), OPTS).unwrap(), 1));
    plot.series.push(history("CG on the 2-D Laplacian (for scale)", &cg(&p, &b, None, None, OPTS).unwrap(), 7));
    write(CHAPTER, "sparse_preconditioning.svg", &plot.render());
}

/// Symmetric permutation `P A Pᵀ` with `perm[k]` the original index at position `k`.
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

fn ordering(refs: &Refs) {
    let perm: Vec<usize> = refs.values("perm").iter().map(|&v| v as usize).collect();
    let shuffled = permute(&poisson(M), &perm);
    let rcm = reverse_cuthill_mckee(&shuffled);
    let bw = refs.values("bandwidth");
    let natural = SparseLu::factor(&shuffled, Ordering::Natural).unwrap().nnz();
    let ordered = SparseLu::factor(&shuffled, Ordering::ReverseCuthillMcKee).unwrap().nnz();
    let fill = |k: &str| format!("{}", refs.values(k)[0]);
    let rows = vec![
        vec!["the shuffled matrix, as given".into(), format!("{} / {}", bandwidth(&shuffled, None), bw[0]), format!("{natural}"), fill("fill NATURAL")],
        vec!["reverse Cuthill-McKee".into(), format!("{} / {}", bandwidth(&shuffled, Some(&rcm)), bw[1]), format!("{ordered}"), fill("fill RCM then NATURAL")],
        vec!["COLAMD (SciPy's default for `splu`)".into(), "—".into(), "—".into(), fill("fill COLAMD")],
        vec!["minimum degree on AᵀA + A (SciPy's `MMD_AT_PLUS_A`)".into(), "—".into(), "—".into(), fill("fill MMD_AT_PLUS_A")],
    ];
    write(CHAPTER, "sparse_fill.md", &md_table(&["ordering", "bandwidth (library / SciPy)", "nnz(L) + nnz(U), library", "nnz(L) + nnz(U), SciPy's SuperLU"], &rows));
}

/// Spy plots of a shuffled 12 × 12-grid Laplacian before and after reverse Cuthill-McKee.
fn spy_plots() {
    let m = 12;
    let n = m * m;
    // A fixed pseudo-random shuffle (Fisher-Yates with a xorshift generator).
    let mut perm: Vec<usize> = (0..n).collect();
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for i in (1..n).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        perm.swap(i, (state % (i as u64 + 1)) as usize);
    }
    let shuffled = permute(&poisson(m), &perm);
    let rcm = reverse_cuthill_mckee(&shuffled);
    for (name, a, title) in [("sparse_spy_shuffled.svg", shuffled.clone(), "Shuffled 2-D Laplacian (n = 144)"), ("sparse_spy_rcm.svg", permute(&shuffled, &rcm), "After reverse Cuthill-McKee")] {
        let mut pts = Vec::new();
        for i in 0..a.nrows {
            for q in a.indptr[i]..a.indptr[i + 1] {
                pts.push((a.indices[q] as f64, -(i as f64)));
            }
        }
        let mut plot = Plot::new(&format!("{title}: bandwidth {}", bandwidth(&a, None)), "column", "−row");
        plot.x_range = Some((0.0, n as f64));
        plot.y_range = Some((-(n as f64), 0.0));
        plot.series.push(Series::new("non-zero", pts, Style::Markers, 0));
        write(CHAPTER, name, &plot.render());
    }
}
