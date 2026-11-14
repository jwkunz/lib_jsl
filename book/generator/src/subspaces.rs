//! Tables for "Matrix functions, subspaces and norms" (the linear algebra chapter): the SciPy and
//! NumPy comparison, and the effect of the truncation tolerance on a pseudoinverse.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::write;
use lib_jsl_linalg::{Norm, cond, fractional_matrix_power, logm, matrix_rank, norm, null_space, orth, pinv, sqrtm};
use ndarray::Array2;
use num_complex::Complex;

const CHAPTER: &str = "linear_algebra";
type C = Complex<f64>;

pub fn generate() {
    scipy_reference();
    truncation_table();
}

fn rel(got: impl Iterator<Item = f64>, want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    got.zip(want).fold(0.0f64, |m, (a, b)| m.max((a - b).abs())) / scale
}

fn scipy_reference() {
    let refs = Refs::parse(include_str!("../data/subspaces_reference.txt"));
    let cmat = |name: &str| {
        let m = refs.get(&format!("matrix {name}"));
        Array2::from_shape_vec((m[0][0] as usize, m[1][0] as usize), m[2].iter().zip(&m[3]).map(|(&r, &i)| C::new(r, i)).collect()).unwrap()
    };
    let rmat = |name: &str| cmat(name).mapv(|z| z.re);
    let cplx_diff = |got: &Array2<C>, key: &str| {
        let w = refs.get(key);
        let g: Vec<f64> = got.iter().map(|z| z.re).chain(got.iter().map(|z| z.im)).collect();
        let want: Vec<f64> = w[0].iter().chain(&w[1]).copied().collect();
        rel(g.into_iter(), &want)
    };
    let mut rows = Vec::new();
    let mut row = |what: String, d: f64| rows.push(vec![what, fmt_diff(d)]);
    for (name, label) in [("spd", "symmetric positive definite 12 × 12"), ("gen", "general 12 × 12"), ("cplx", "complex 10 × 10")] {
        let (s, l, f) = if name == "cplx" {
            let a = cmat(name);
            (sqrtm(&a).unwrap(), logm(&a).unwrap(), fractional_matrix_power(&a, 0.3).unwrap())
        } else {
            let a = rmat(name);
            (sqrtm(&a).unwrap(), logm(&a).unwrap(), fractional_matrix_power(&a, 0.3).unwrap())
        };
        row(format!("`sqrtm`, {label}"), cplx_diff(&s, &format!("sqrtm {name}")));
        row(format!("`logm`, {label}"), cplx_diff(&l, &format!("logm {name}")));
        row(format!("`fractional_matrix_power` (p = 0.3), {label}"), cplx_diff(&f, &format!("fmp {name}")));
    }
    let low = rmat("lowrank");
    row("`pinv`, 15 × 9 of rank 4".into(), rel(pinv(&low, None).unwrap().into_iter(), refs.values("pinv lowrank")));
    row(format!("`matrix_rank`, 15 × 9 of rank 4: {} (NumPy {})", matrix_rank(&low, None).unwrap(), refs.values("rank lowrank")[0]), 0.0);
    let ns = null_space(&low, None).unwrap();
    row(format!("`null_space`: projector onto the {}-dimensional null space", ns.ncols()), rel(ns.dot(&ns.t()).into_iter(), refs.values("null projector")));
    let o = orth(&low, None).unwrap();
    row(format!("`orth`: projector onto the {}-dimensional range", o.ncols()), rel(o.dot(&o.t()).into_iter(), refs.values("orth projector")));
    row("`pinv`, 20 × 7 of full rank".into(), rel(pinv(&rmat("tall"), None).unwrap().into_iter(), refs.values("pinv tall")));
    let ords = [Norm::Frobenius, Norm::Nuclear, Norm::One, Norm::MinusOne, Norm::Two, Norm::MinusTwo, Norm::Infinity, Norm::MinusInfinity];
    for name in ["gen", "lowrank", "tall"] {
        let a = rmat(name);
        let got: Vec<f64> = ords.iter().map(|&o| norm(&a, o).unwrap()).collect();
        let want = refs.values(&format!("norms {name}"));
        let worst = got.iter().zip(want).fold(0.0f64, |m, (g, w)| m.max((g - w).abs() / w.abs().max(1.0)));
        row(format!("`norm`, all eight orders, {name}"), worst);
    }
    for name in ["gen", "spd"] {
        let a = rmat(name);
        let got = [cond(&a, Norm::Two).unwrap(), cond(&a, Norm::MinusTwo).unwrap(), cond(&a, Norm::One).unwrap(), cond(&a, Norm::Infinity).unwrap(), cond(&a, Norm::Frobenius).unwrap()];
        let want = refs.values(&format!("cond {name}"));
        let worst = got.iter().zip(want).fold(0.0f64, |m, (g, w)| m.max((g - w).abs() / w.abs()));
        row(format!("`cond` in the 2, −2, 1, ∞ and Frobenius norms, {name}"), worst);
    }
    write(CHAPTER, "subspaces_scipy.md", &md_table(&["computation (SciPy / NumPy reference)", "largest difference (relative)"], &rows));
}

/// A rank-3 matrix plus noise of size 10⁻⁸: the pseudoinverse solution of `A x = b` for several
/// truncation tolerances.
fn truncation_table() {
    use lib_jsl_random::uniform_generator::UniformGenerator;
    use lib_jsl_random::xoshiro256plusplus::Xoshiro256PlusPlus;
    let mut rng = Xoshiro256PlusPlus::from_seed(4);
    let mut g = || rng.next_f64() * 2.0 - 1.0;
    let u = Array2::from_shape_fn((30, 3), |_| g());
    let v = Array2::from_shape_fn((3, 10), |_| g());
    let noise = Array2::from_shape_fn((30, 10), |_| 1e-8 * g());
    let a = u.dot(&v) + &noise;
    let x_true = Array2::from_shape_fn((10, 1), |_| g());
    // The reachable part of x_true: its projection onto the row space of U V.
    let p = pinv(&u.dot(&v), None).unwrap().dot(&u.dot(&v));
    let x_star = p.dot(&x_true);
    let b = u.dot(&v).dot(&x_true);
    let mut rows = Vec::new();
    for rtol in [1e-15, 1e-10, 1e-6, 1e-2] {
        let x = pinv(&a, Some(rtol)).unwrap().dot(&b);
        let err = (&x - &x_star).iter().map(|v| v * v).sum::<f64>().sqrt();
        let r = matrix_rank(&a, Some(rtol * norm(&a, Norm::Two).unwrap())).unwrap();
        rows.push(vec![format!("{rtol:.0e}"), format!("{r}"), format!("{err:.1e}")]);
    }
    write(CHAPTER, "pinv_truncation.md", &md_table(&["rtol", "singular values kept", "error of A⁺b against the minimum-norm solution"], &rows));
}
