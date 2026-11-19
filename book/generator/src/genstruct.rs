//! Tables and a figure for release 2.6.0 (linear algebra chapter): generalised eigenproblems,
//! structured matrices and their fast solvers, compared with SciPy.

use crate::linear_algebra::time;
use crate::refs::{Refs, fmt_diff, max_matched_distance};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_linalg::*;
use ndarray::{Array1, Array2};
use num_complex::Complex;

type C = Complex<f64>;

const CHAPTER: &str = "linear_algebra";

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/genstruct_reference.txt"));
    generalized_table(&refs);
    structured_table(&refs);
    timing_figure();
}

fn pencil(n: usize, seed: f64) -> (Array2<f64>, Array2<f64>) {
    let a = Array2::from_shape_fn((n, n), |(i, j)| (seed + 1.3 * i as f64 + 0.7 * (j * j) as f64 + 0.37 * (i * j) as f64).sin());
    let b = Array2::from_shape_fn((n, n), |(i, j)| (seed + 0.5 * i as f64 - 0.9 * j as f64 + 0.21 * (i * j) as f64).cos() + if i == j { 3.0 } else { 0.0 });
    (a, b)
}

fn complex_list(refs: &[Vec<f64>], start: usize) -> Vec<C> {
    refs[start].iter().zip(&refs[start + 1]).map(|(&a, &b)| C::new(a, b)).collect()
}

/// The largest difference between columns of `got` and `want` (n × n, row-major lists), after
/// rotating each column of `got` by the phase that best aligns it with `want`'s.
fn columns_up_to_phase(got: &Array2<C>, want: &[C]) -> f64 {
    let n = got.ncols();
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.norm()));
    let mut worst = 0.0f64;
    for k in 0..n {
        let inner: C = (0..got.nrows()).map(|i| got[(i, k)].conj() * want[i * n + k]).sum();
        let phase = if inner.norm() > 0.0 { inner / inner.norm() } else { C::new(1.0, 0.0) };
        for i in 0..got.nrows() {
            worst = worst.max((got[(i, k)] * phase - want[i * n + k]).norm());
        }
    }
    worst / scale
}

/// Reorders the columns of `vectors` (eigenvectors for `values`) to match the order of `want`.
fn match_order(values: &[C], vectors: &Array2<C>, want: &[C]) -> Array2<C> {
    let mut used = vec![false; values.len()];
    let order: Vec<usize> = want
        .iter()
        .map(|w| {
            let k = (0..values.len()).filter(|&k| !used[k]).min_by(|&x, &y| (values[x] - w).norm().total_cmp(&(values[y] - w).norm())).unwrap();
            used[k] = true;
            k
        })
        .collect();
    Array2::from_shape_fn(vectors.dim(), |(i, j)| vectors[(i, order[j])])
}

fn rel(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(f64::MIN_POSITIVE);
    got.iter().zip(want).fold(0.0f64, |m, (a, b)| m.max((a - b).abs())) / scale
}

fn generalized_table(refs: &Refs) {
    let mut rows = Vec::new();
    // Symmetric-definite, real.
    let (a, b) = pencil(7, 1.0);
    let sa = &a + &a.t();
    let sb = b.t().dot(&b);
    let (w, v) = eigh_generalized(&sa, &sb, UPLO::Lower).unwrap();
    let r = refs.get("eigh real");
    let vc = v.mapv(|x| C::new(x, 0.0));
    let want: Vec<C> = r[1].iter().map(|&x| C::new(x, 0.0)).collect();
    rows.push(vec!["`eigh_generalized`, real symmetric-definite, n = 7".into(), fmt_diff(rel(w.as_slice().unwrap(), &r[0])), fmt_diff(columns_up_to_phase(&vc, &want))]);
    // Hermitian-definite.
    let (ca, cb) = pencil(6, 2.0);
    let z = Array2::from_shape_fn((6, 6), |(i, j)| C::new(ca[(i, j)], cb[(i, j)]));
    let ha = &z + &z.t().mapv(|x| x.conj());
    let y = Array2::from_shape_fn((6, 6), |(i, j)| C::new(cb[(i, j)], ca[(i, j)]));
    let hb = y.t().mapv(|x| x.conj()).dot(&y) + Array2::<C>::eye(6) * C::new(6.0, 0.0);
    let (w, v) = eigh_generalized(&ha, &hb, UPLO::Lower).unwrap();
    let r = refs.get("eigh complex");
    rows.push(vec!["`eigh_generalized`, Hermitian-definite, n = 6".into(), fmt_diff(rel(w.as_slice().unwrap(), &r[0])), fmt_diff(columns_up_to_phase(&v, &complex_list(r, 2)))]);
    // General pencils.
    let mut general = |label: &str, key: &str, e: GeneralizedEigen| {
        let r = refs.get(key);
        let want = complex_list(r, 0);
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.norm()));
        let dv = max_matched_distance(e.values.as_slice().unwrap(), &r[0], &r[1]) / scale;
        let ordered = match_order(e.values.as_slice().unwrap(), &e.vectors, &want);
        rows.push(vec![label.into(), fmt_diff(dv), fmt_diff(columns_up_to_phase(&ordered, &complex_list(r, 2)))]);
    };
    for (n, seed) in [(6, 3.0), (12, 4.0)] {
        let (a, b) = pencil(n, seed);
        general(&format!("`eig_generalized`, real, n = {n}"), &format!("eig {n}"), eig_generalized(&a, &b).unwrap());
    }
    let (a, b) = pencil(5, 5.0);
    let za = Array2::from_shape_fn((5, 5), |(i, j)| C::new(a[(i, j)], b[(j, i)]));
    let zb = Array2::from_shape_fn((5, 5), |(i, j)| C::new(b[(i, j)], 0.3 * a[(j, i)]));
    general("`eig_generalized`, complex, n = 5", "eig complex", eig_generalized(&za, &zb).unwrap());
    // Singular B.
    let a = Array2::from_shape_vec((3, 3), vec![2.0, 1.0, 0.5, 0.0, 3.0, 1.0, 1.0, 0.0, 4.0]).unwrap();
    let b = Array2::from_diag(&Array1::from_vec(vec![1.0, 1.0, 0.0]));
    let e = eig_generalized(&a, &b).unwrap();
    let r = refs.get("eig singular");
    let got: Vec<C> = e.values.iter().map(|v| if v.re.is_infinite() { C::new(1e300, 0.0) } else { *v }).collect();
    let finite: Vec<(f64, f64)> = r[0].iter().zip(&r[1]).filter(|(x, _)| **x < 1e299).map(|(x, y)| (*x, *y)).collect();
    let got_finite: Vec<C> = got.iter().copied().filter(|v| v.re < 1e299).collect();
    let fr: Vec<f64> = finite.iter().map(|p| p.0).collect();
    let fi: Vec<f64> = finite.iter().map(|p| p.1).collect();
    assert_eq!(got.iter().filter(|v| v.re >= 1e299).count(), 1, "one infinite eigenvalue");
    rows.push(vec!["`eig_generalized`, singular B (one infinite eigenvalue)".into(), fmt_diff(max_matched_distance(&got_finite, &fr, &fi) / 3.0), "—".into()]);
    // QZ.
    let (a, b) = pencil(8, 6.0);
    let f = qz(&a, &b).unwrap();
    let lam: Vec<C> = (0..8).map(|i| f.s[(i, i)] / f.t[(i, i)]).collect();
    let r = refs.get("qz 8");
    let scale = r[0].iter().zip(&r[1]).fold(0.0f64, |m, (x, y)| m.max(x.hypot(*y)));
    let zh = f.z.t().mapv(|v| v.conj());
    let rec = (f.q.dot(&f.s).dot(&zh) - a.mapv(|v| C::new(v, 0.0))).iter().fold(0.0f64, |m, v| m.max(v.norm()));
    rows.push(vec![format!("`qz` (complex), n = 8: eigenvalues; ‖Q S Zᴴ − A‖ = {rec:.0e}"), fmt_diff(max_matched_distance(&lam, &r[0], &r[1]) / scale), "—".into()]);
    write(CHAPTER, "generalized_scipy.md", &md_table(&["problem", "eigenvalues (relative difference)", "eigenvectors, up to phase (relative)"], &rows));
}

fn flat(a: &Array2<f64>) -> Vec<f64> {
    a.iter().copied().collect()
}

fn flat_c(a: &Array2<C>) -> (Vec<f64>, Vec<f64>) {
    (a.iter().map(|v| v.re).collect(), a.iter().map(|v| v.im).collect())
}

fn structured_table(refs: &Refs) {
    let mut worst_constructors = 0.0f64;
    let mut check = |got: Vec<f64>, key: &str| {
        let w = refs.values(key);
        assert_eq!(got.len(), w.len(), "{key}: shape");
        worst_constructors = worst_constructors.max(rel(&got, w));
    };
    let c = [1.0, 2.5, -1.0, 4.0];
    let r = [9.0, -3.0, 0.5];
    check(flat(&toeplitz(&c, Some(&r))), "toeplitz");
    check(flat(&hankel(&c, Some(&r))), "hankel");
    check(flat(&hankel(&c, None)), "hankel zeros");
    check(flat(&circulant(&c)), "circulant");
    check(flat(&hadamard(16).unwrap()), "hadamard 16");
    let bd = block_diag(&[Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap(), Array2::from_elem((1, 1), 5.0), Array2::from_shape_vec((1, 3), vec![6.0, 7.0, 8.0]).unwrap()]);
    check(flat(&bd), "block_diag");
    check(flat(&companion(&[2.0, -3.0, 0.5, 4.0]).unwrap()), "companion");
    for (kind, name) in [(PascalKind::Symmetric, "symmetric"), (PascalKind::Lower, "lower"), (PascalKind::Upper, "upper")] {
        check(flat(&pascal(6, kind)), &format!("pascal {name}"));
        check(flat(&invpascal(6, kind)), &format!("invpascal {name}"));
    }
    check(flat(&hilbert(5)), "hilbert");
    check(flat(&invhilbert(8)), "invhilbert");
    check(flat(&leslie(&[0.1, 2.0, 1.0, 0.1], &[0.2, 0.8, 0.7]).unwrap()), "leslie");
    for (mode, name) in [(ConvolutionMode::Full, "full"), (ConvolutionMode::Same, "same"), (ConvolutionMode::Valid, "valid")] {
        check(flat(&convolution_matrix(&[1.0, -2.0, 3.0, 0.5], 6, mode).unwrap()), &format!("convolution {name}"));
        check(flat(&convolution_matrix(&[1.0, -2.0, 3.0, 0.5, 2.0, 1.0], 4, mode).unwrap()), &format!("convolution short {name}"));
    }
    check(flat(&fiedler(&[1.0, 4.0, 12.0, 45.0, 77.0])), "fiedler");
    let mut worst_complex = 0.0f64;
    for (scale, name) in [(DftScale::None, "None"), (DftScale::SqrtN, "sqrtn"), (DftScale::N, "n")] {
        let (re, im) = flat_c(&dft(5, scale));
        let w = refs.get(&format!("dft {name}"));
        worst_complex = worst_complex.max(rel(&re, &w[0])).max(rel(&im, &w[1]));
    }
    let cc = [C::new(1.0, 1.0), C::new(2.0, -1.0), C::new(0.0, 0.5)];
    let (re, im) = flat_c(&toeplitz(&cc, None));
    let w = refs.get("toeplitz hermitian");
    worst_complex = worst_complex.max(rel(&re, &w[0])).max(rel(&im, &w[1]));

    // Solvers.
    let n = 40;
    let mut c: Vec<f64> = (0..n).map(|k| 4.0 / (1.0 + k as f64)).collect();
    c[0] += 2.0;
    let mut r: Vec<f64> = (0..n).map(|k| -0.7 / (1.0 + (k * k) as f64)).collect();
    r[0] = c[0];
    let b: Vec<f64> = (0..n).map(|k| (0.7 * k as f64).sin() + 1.0).collect();
    let tz = rel(solve_toeplitz(&c, Some(&r), &b).unwrap().as_slice().unwrap(), refs.values("solve_toeplitz"));
    let ch: Vec<C> = c.iter().enumerate().map(|(k, &v)| C::new(v, if k == 0 { 0.0 } else { 0.3 / k as f64 })).collect();
    let bh: Vec<C> = b.iter().map(|&v| C::new(v, -v)).collect();
    let x = solve_toeplitz(&ch, None, &bh).unwrap();
    let w = refs.get("solve_toeplitz hermitian");
    let tzh = rel(&x.iter().map(|v| v.re).collect::<Vec<_>>(), &w[0]).max(rel(&x.iter().map(|v| v.im).collect::<Vec<_>>(), &w[1]));
    let circ = rel(solve_circulant(&c, &b, None).unwrap().as_slice().unwrap(), refs.values("solve_circulant"));
    let bc: Vec<C> = b.iter().map(|&v| C::new(v, 0.5)).collect();
    let x = solve_circulant(&ch, &bc, None).unwrap();
    let w = refs.get("solve_circulant complex");
    let circ_c = rel(&x.iter().map(|v| v.re).collect::<Vec<_>>(), &w[0]).max(rel(&x.iter().map(|v| v.im).collect::<Vec<_>>(), &w[1]));
    let (l, u) = (3, 2);
    let ab = Array2::from_shape_fn((l + u + 1, n), |(i, j)| (0.37 * (7 * (i + 1) + 3 * j) as f64).sin() + if i == u { 0.01 } else { 0.0 });
    let banded = rel(solve_banded((l, u), &ab, &Array1::from_vec(b.clone())).unwrap().as_slice().unwrap(), refs.values("solve_banded"));
    let wd = 2;
    let upper = Array2::from_shape_fn((wd + 1, n), |(k, j)| if j + k >= wd { if k == wd { 6.0 } else { 1.0 / (1 + j + (j + k - wd)) as f64 } } else { 0.0 });
    let banded_h = rel(solveh_banded(&upper, &Array1::from_vec(b.clone()), false).unwrap().as_slice().unwrap(), refs.values("solveh_banded"));

    let rows = vec![
        vec!["constructors: `toeplitz`, `hankel`, `circulant`, `hadamard`, `block_diag`, `companion`, `pascal`, `invpascal`, `hilbert`, `invhilbert`, `leslie`, `convolution_matrix`, `fiedler`".into(), fmt_diff(worst_constructors)],
        vec!["`dft` (three scalings); Hermitian `toeplitz`".into(), fmt_diff(worst_complex)],
        vec!["`solve_toeplitz`, n = 40, nonsymmetric".into(), fmt_diff(tz)],
        vec!["`solve_toeplitz`, n = 40, complex Hermitian".into(), fmt_diff(tzh)],
        vec!["`solve_circulant`, n = 40, real and complex".into(), fmt_diff(circ.max(circ_c))],
        vec!["`solve_banded`, n = 40, l = 3, u = 2 (pivoting)".into(), fmt_diff(banded)],
        vec!["`solveh_banded`, n = 40, bandwidth 2".into(), fmt_diff(banded_h)],
    ];
    write(CHAPTER, "structured_scipy.md", &md_table(&["functions", "largest relative difference from SciPy"], &rows));
}

fn timing_figure() {
    let mut plot = Plot::new("Solving structured systems: fast solvers against dense LU", "n", "time (s)");
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::TopLeft;
    plot.x_range = Some((80.0, 2000.0));
    plot.y_range = Some((1e-6, 100.0));
    let sizes = [100usize, 200, 400, 800, 1600];
    let mut series: Vec<(&str, Vec<(f64, f64)>)> = vec![("dense LU (solve)", vec![]), ("Levinson (solve_toeplitz)", vec![]), ("FFT (solve_circulant)", vec![]), ("band LU, l = u = 2 (solve_banded)", vec![])];
    for &n in &sizes {
        let c: Vec<f64> = (0..n).map(|k| 1.0 / (1.0 + k as f64) + if k == 0 { 3.0 } else { 0.0 }).collect();
        let b: Vec<f64> = (0..n).map(|k| (0.3 * k as f64).cos()).collect();
        let t = toeplitz(&c, None);
        let bv = Array1::from_vec(b.clone());
        if n <= 1600 {
            series[0].1.push((n as f64, time(|| {
                let _ = t.solve(&bv).unwrap();
            })));
        }
        series[1].1.push((n as f64, time(|| {
            let _ = solve_toeplitz(&c, None, &b).unwrap();
        })));
        series[2].1.push((n as f64, time(|| {
            let _ = solve_circulant(&c, &b, None).unwrap();
        })));
        let ab = Array2::from_shape_fn((5, n), |(i, j)| if i == 2 { 4.0 } else { 1.0 / (1 + i + j) as f64 });
        series[3].1.push((n as f64, time(|| {
            let _ = solve_banded((2, 2), &ab, &bv).unwrap();
        })));
    }
    for (k, (label, pts)) in series.into_iter().enumerate() {
        plot.series.push(Series::new(label, pts, Style::LineMarkers, k));
    }
    write(CHAPTER, "structured_timing.svg", &plot.render());
}
