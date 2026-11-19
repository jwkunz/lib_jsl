//! Tables for release 2.5.0 (linear algebra chapter): QR with and without pivoting, RQ,
//! Hessenberg, real Schur, LDLᵀ, polar decomposition, subspace angles and funm against SciPy.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::write;
use lib_jsl_linalg::{Eig, QrMode, funm, hessenberg, ldl, polar, qr, rq, schur_real, subspace_angles};
use ndarray::Array2;
use num_complex::Complex;

type C = Complex<f64>;

const CHAPTER: &str = "linear_algebra";

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/decomp_reference.txt"));
    table(&refs);
}

fn real(m: usize, n: usize, seed: f64) -> Array2<f64> {
    Array2::from_shape_fn((m, n), |(i, j)| (seed + 1.3 * i as f64 + 0.7 * (j * j) as f64 + 0.37 * (i * j) as f64).sin() + if i == j { 2.0 } else { 0.0 })
}

fn cplx(m: usize, n: usize, seed: f64) -> Array2<C> {
    let r = real(m, n, seed);
    Array2::from_shape_fn((m, n), |(i, j)| C::new(r[(i, j)], (seed + 0.4 * (i * i) as f64 - 0.9 * j as f64).cos()))
}

fn to_c(a: &Array2<f64>) -> Array2<C> {
    a.mapv(|v| C::new(v, 0.0))
}

/// The largest difference between `got` and the reference lists `(re, im)` starting at list
/// `start`, relative to the reference's largest entry.
fn diff(got: &Array2<C>, refs: &[Vec<f64>], start: usize) -> f64 {
    let (re, im) = (&refs[start], &refs[start + 1]);
    assert_eq!(re.len(), got.len(), "shape mismatch");
    let scale = re.iter().zip(im).fold(0.0f64, |m, (a, b)| m.max(a.hypot(*b))).max(f64::MIN_POSITIVE);
    got.iter().zip(re.iter().zip(im)).fold(0.0f64, |m, (g, (a, b))| m.max((g - C::new(*a, *b)).norm())) / scale
}

/// As [`diff`], over the first `cols` columns only.
fn diff_cols(got: &Array2<C>, refs: &[Vec<f64>], start: usize, cols: usize) -> f64 {
    let n = got.ncols();
    let (re, im) = (&refs[start], &refs[start + 1]);
    let pairs: Vec<(C, C)> = (0..got.nrows()).flat_map(|i| (0..cols).map(move |j| (i, j))).map(|(i, j)| (got[(i, j)], C::new(re[i * n + j], im[i * n + j]))).collect();
    let scale = pairs.iter().fold(0.0f64, |m, p| m.max(p.1.norm()));
    pairs.iter().fold(0.0f64, |m, (g, w)| m.max((g - w).norm())) / scale
}

fn diff_index(got: &[usize], refs: &[Vec<f64>], start: usize) -> f64 {
    let want: Vec<usize> = refs[start].iter().map(|&v| v as usize).collect();
    if got == want.as_slice() { 0.0 } else { 1.0 }
}

enum M {
    R(Array2<f64>),
    Z(Array2<C>),
}

fn sym(n: usize, seed: f64, small_diag: f64) -> Array2<f64> {
    let s = Array2::from_shape_fn((n, n), |(i, j)| (seed + 0.9 * (i + j) as f64 + 0.3 * (i * j) as f64).sin());
    Array2::from_shape_fn((n, n), |(i, j)| if i == j { small_diag * (i as f64 + seed).cos() } else { s[(i, j)] + s[(j, i)] })
}

fn herm(n: usize, seed: f64, small_diag: f64) -> Array2<C> {
    let s = sym(n, seed, small_diag);
    let im = Array2::from_shape_fn((n, n), |(i, j)| (seed + 0.5 * i as f64 - 0.8 * j as f64).cos());
    Array2::from_shape_fn((n, n), |(i, j)| {
        if i == j {
            C::new(small_diag * (i as f64 + seed).cos(), 0.0)
        } else {
            (C::new(s[(i, j)], im[(i, j)]) + C::new(s[(j, i)], im[(j, i)]).conj()) / 2.0
        }
    })
}

fn table(refs: &Refs) {
    let mut mats: Vec<(&str, M)> = vec![
        ("square", M::R(real(6, 6, 0.0))),
        ("tall", M::R(real(7, 4, 1.0))),
        ("wide", M::R(real(4, 7, 2.0))),
        ("complex", M::Z(cplx(5, 5, 3.0))),
        ("complex wide", M::Z(cplx(3, 5, 4.0))),
    ];
    let rd = real(6, 3, 5.0);
    let comb = Array2::from_shape_vec((3, 3), vec![1.0, 0.5, -2.0, 0.0, 1.0, 1.0, 3.0, -1.0, 0.5]).unwrap();
    let prod = rd.dot(&comb);
    mats.push(("rank deficient", M::R(Array2::from_shape_fn((6, 6), |(i, j)| if j < 3 { rd[(i, j)] } else { prod[(i, j - 3)] }))));

    let mut worst = [0.0f64; 5]; // qr, qr pivoted, rq, polar, hessenberg
    let mut rank_def = 0.0f64;
    for (name, a) in &mats {
        for (mode, mname) in [(QrMode::Full, "full"), (QrMode::Economic, "economic")] {
            let (q1, r1, q2, r2, p2, rr, rqq) = match a {
                M::R(a) => {
                    let f = qr(a, mode, false).unwrap();
                    let g = qr(a, mode, true).unwrap();
                    let (r, q) = rq(a, mode).unwrap();
                    (to_c(&f.q), to_c(&f.r), to_c(&g.q), to_c(&g.r), g.permutation, to_c(&r), to_c(&q))
                }
                M::Z(a) => {
                    let f = qr(a, mode, false).unwrap();
                    let g = qr(a, mode, true).unwrap();
                    let (r, q) = rq(a, mode).unwrap();
                    (f.q, f.r, g.q, g.r, g.permutation, r, q)
                }
            };
            if *name == "rank deficient" {
                // Beyond the rank (3), Q's columns, the pivot order and the matching columns of R
                // are decided by rounding errors; compare what is determined.
                let w = refs.get(&format!("qr {name} {mname}"));
                rank_def = rank_def.max(diff(&r1, w, 2)).max(diff_cols(&q1, w, 0, 3));
                let w = refs.get(&format!("qrp {name} {mname}"));
                rank_def = rank_def.max(diff_cols(&q2, w, 0, 3)).max(if p2[..3] == w[4][..3].iter().map(|&v| v as usize).collect::<Vec<_>>()[..] { 0.0 } else { 1.0 });
                let r_lead = r2.slice(ndarray::s![..3, ..3]).to_owned();
                let w_r: Vec<C> = (0..3).flat_map(|i| (0..3).map(move |j| (i, j))).map(|(i, j)| C::new(w[2][i * 6 + j], w[3][i * 6 + j])).collect();
                let scale = w_r.iter().fold(0.0f64, |m, v| m.max(v.norm()));
                rank_def = rank_def.max(r_lead.iter().zip(&w_r).fold(0.0f64, |m, (g, v)| m.max((g - v).norm())) / scale);
                let w = refs.get(&format!("rq {name} {mname}"));
                rank_def = rank_def.max(diff(&rr, w, 0));
                continue;
            }
            let w = refs.get(&format!("qr {name} {mname}"));
            worst[0] = worst[0].max(diff(&q1, w, 0)).max(diff(&r1, w, 2));
            let w = refs.get(&format!("qrp {name} {mname}"));
            worst[1] = worst[1].max(diff(&q2, w, 0)).max(diff(&r2, w, 2)).max(diff_index(&p2, w, 4));
            let w = refs.get(&format!("rq {name} {mname}"));
            worst[2] = worst[2].max(diff(&rr, w, 0)).max(diff(&rqq, w, 2));
        }
        for (right, side) in [(true, "right"), (false, "left")] {
            let (u, p) = match a {
                M::R(a) => {
                    let (u, p) = polar(a, right).unwrap();
                    (to_c(&u), to_c(&p))
                }
                M::Z(a) => polar(a, right).unwrap(),
            };
            let w = refs.get(&format!("polar {name} {side}"));
            if *name == "rank deficient" {
                rank_def = rank_def.max(diff(&p, w, 2));
                continue;
            }
            worst[3] = worst[3].max(diff(&u, w, 0)).max(diff(&p, w, 2));
        }
    }
    let big = real(30, 30, 6.0);
    for (name, a) in [("square", M::R(real(6, 6, 0.0))), ("complex", M::Z(cplx(5, 5, 3.0))), ("big", M::R(big.clone()))] {
        let (h, q) = match &a {
            M::R(a) => {
                let (h, q) = hessenberg(a).unwrap();
                (to_c(&h), to_c(&q))
            }
            M::Z(a) => hessenberg(a).unwrap(),
        };
        let w = refs.get(&format!("hess {name}"));
        worst[4] = worst[4].max(diff(&h, w, 0)).max(diff(&q, w, 2));
    }
    // Real Schur: block eigenvalues against SciPy's, the block structure, and the reconstruction.
    let mut schur_eig = 0.0f64;
    let mut schur_rec = 0.0f64;
    for (name, a) in [("square", real(6, 6, 0.0)), ("big", big)] {
        let (t, z) = schur_real(&a).unwrap();
        let n = t.nrows();
        for i in 0..n {
            for j in 0..i {
                let in_block = j + 1 == i && t[(i, j)] != 0.0;
                assert!(t[(i, j)] == 0.0 || in_block, "T not quasi-triangular at ({i}, {j})");
                if in_block {
                    assert!(t[(j, j)] == t[(i, i)] && t[(j, i)] * t[(i, j)] < 0.0, "2 × 2 block not standardised");
                }
            }
        }
        let rec = z.dot(&t).dot(&z.t());
        let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        schur_rec = schur_rec.max(rec.iter().zip(a.iter()).fold(0.0f64, |m, (x, y)| m.max((x - y).abs())) / scale);
        let (ev, _) = t.eig().unwrap();
        let mut ev: Vec<C> = ev.to_vec();
        ev.sort_by(|x, y| x.re.total_cmp(&y.re).then(x.im.total_cmp(&y.im)));
        let w = refs.get(&format!("schur {name}"));
        let wv: Vec<C> = w[0].iter().zip(&w[1]).map(|(a, b)| C::new(*a, *b)).collect();
        let s = wv.iter().fold(0.0f64, |m, v| m.max(v.norm()));
        schur_eig = schur_eig.max(crate::refs::max_matched_distance(&ev, &w[0], &w[1]) / s);
    }
    // LDL.
    let ldl_mats: Vec<(&str, M, bool)> = vec![
        ("spd", M::R(sym(6, 0.0, 0.0) + &(Array2::<f64>::eye(6) * 8.0)), true),
        ("indefinite", M::R(sym(7, 1.0, 0.05)), true),
        ("zero diagonal", M::R(sym(6, 2.0, 0.0)), true),
        ("hermitian", M::Z(herm(6, 3.0, 0.1)), true),
        ("complex symmetric", M::Z(Array2::from_shape_fn((5, 5), |(i, j)| C::new(sym(5, 4.0, 0.02)[(i, j)], sym(5, 5.0, 0.03)[(i, j)]))), false),
    ];
    let mut ldl_worst = 0.0f64;
    let mut two_by_two = 0;
    for (name, a, hermitian) in &ldl_mats {
        for (lower, lname) in [(true, "lower"), (false, "upper")] {
            let (l, d, p) = match a {
                M::R(a) => {
                    let f = ldl(a, lower, *hermitian).unwrap();
                    (to_c(&f.l), to_c(&f.d), f.perm)
                }
                M::Z(a) => {
                    let f = ldl(a, lower, *hermitian).unwrap();
                    (f.l, f.d, f.perm)
                }
            };
            two_by_two += (0..d.nrows() - 1).filter(|&i| d[(i + 1, i)] != C::new(0.0, 0.0)).count();
            let w = refs.get(&format!("ldl {name} {lname}"));
            ldl_worst = ldl_worst.max(diff(&l, w, 0)).max(diff(&d, w, 2)).max(diff_index(&p, w, 4));
        }
    }
    assert!(two_by_two > 0, "the tests should exercise 2 × 2 pivots");
    // Subspace angles.
    let a = real(8, 3, 7.0);
    let mut angles = 0.0f64;
    let check = |got: &[f64], want: &[f64]| got.iter().zip(want).fold(0.0f64, |m, (g, w)| m.max((g - w).abs() / w.abs().max(1e-300)));
    angles = angles.max(check(subspace_angles(&a, &real(8, 4, 8.0)).unwrap().as_slice().unwrap(), refs.values("angles general")));
    let near = &a + &(real(8, 3, 9.0) * 1e-7);
    let near_rel = check(subspace_angles(&a, &near).unwrap().as_slice().unwrap(), refs.values("angles near"));
    angles = angles.max(check(subspace_angles(&cplx(6, 2, 10.0), &cplx(6, 3, 11.0)).unwrap().as_slice().unwrap(), refs.values("angles complex")));
    // funm.
    let mut funm_worst = 0.0f64;
    let sq = real(6, 6, 0.0);
    type ScalarFn = fn(C) -> C;
    let fs: [(&str, ScalarFn); 3] = [("exp", |z| z.exp()), ("sin", |z| z.sin()), ("cube", |z| z * z * z)];
    for (name, f) in fs {
        let (fa, _) = funm(&sq, f).unwrap();
        funm_worst = funm_worst.max(diff(&fa, refs.get(&format!("funm square {name}")), 0));
    }
    let (fa, _) = funm(&cplx(5, 5, 3.0), |z| z.exp()).unwrap();
    funm_worst = funm_worst.max(diff(&fa, refs.get("funm complex exp"), 0));

    let rows = vec![
        vec!["`qr` (full and economic)".into(), "square, tall, wide and complex matrices".into(), fmt_diff(worst[0])],
        vec!["`qr` with column pivoting".into(), "the same; permutations identical".into(), fmt_diff(worst[1])],
        vec!["`rq` (full and economic)".into(), "the same".into(), fmt_diff(worst[2])],
        vec!["`polar` (right and left)".into(), "the same".into(), fmt_diff(worst[3])],
        vec!["all four, a 6 × 6 matrix of rank 3".into(), "the parts determined by the matrix: R, P, the first 3 pivots and columns of Q".into(), fmt_diff(rank_def)],
        vec!["`hessenberg`".into(), "6 × 6 real, 5 × 5 complex, 30 × 30 real".into(), fmt_diff(worst[4])],
        vec!["`schur_real`: eigenvalues of the blocks".into(), "6 × 6 and 30 × 30 real".into(), fmt_diff(schur_eig)],
        vec!["`schur_real`: ‖Z T Zᵀ − A‖ / max \\|A\\|".into(), "the same (2 × 2 blocks standardised)".into(), fmt_diff(schur_rec)],
        vec!["`ldl` (lower and upper)".into(), format!("definite, indefinite, zero-diagonal, Hermitian and complex symmetric; {two_by_two} 2 × 2 pivots"), fmt_diff(ldl_worst)],
        vec!["`subspace_angles` (relative)".into(), "real and complex subspaces".into(), fmt_diff(angles)],
        vec!["`subspace_angles`, angles near 10⁻⁷ (relative)".into(), "a slightly perturbed subspace".into(), fmt_diff(near_rel)],
        vec!["`funm`".into(), "exp, sin and x³ of a 6 × 6 real matrix; exp of a 5 × 5 complex one".into(), fmt_diff(funm_worst)],
    ];
    write(CHAPTER, "decomp_scipy.md", &md_table(&["function", "test matrices", "largest difference from SciPy (relative to the largest entry)"], &rows));
}
