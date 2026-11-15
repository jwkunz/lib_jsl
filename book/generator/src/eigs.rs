//! Tables and figures for release 2.2.0 (linear algebra chapter): `eigs` and
//! `eigs_shift_invert`, compared with SciPy's ARPACK and with the exact spectra.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_linalg::sparse::{CooMatrix, CsrMatrix, EigsOptions, EigsWhich, eigs, eigs_shift_invert};
use num_complex::Complex;
use std::f64::consts::PI;

type C = Complex<f64>;

const CHAPTER: &str = "linear_algebra";

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/eigs_reference.txt"));
    let mats = Matrices::new(&refs);
    table(&refs, &mats);
    spectrum_figure(&mats);
}

/// A pseudo-random sparse matrix with four off-diagonal entries per row (as in the tests).
fn scattered(n: usize) -> CsrMatrix {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, (i as f64 * 0.37).sin());
        for t in 1..=4 {
            coo.push(i, (i * 37 + 11 * t + 7 * t * t) % n, (i as f64 + 3.0 * t as f64 + 1.0).sin());
        }
    }
    coo.to_csr().unwrap()
}

/// A diagonal ramp plus a skew-symmetric tridiagonal part.
fn ramp(n: usize) -> CsrMatrix {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0 * i as f64 / n as f64);
        if i + 1 < n {
            coo.push(i, i + 1, 1.0);
            coo.push(i + 1, i, -1.0);
        }
    }
    coo.to_csr().unwrap()
}

/// The 5-point convection-diffusion operator `T ⊗ I + I ⊗ T`, `T = tridiag(−1 − γ, 2, −1 + γ)`.
fn convection(m: usize, gamma: f64) -> CsrMatrix {
    let mut coo = CooMatrix::new(m * m, m * m);
    for i in 0..m {
        for j in 0..m {
            let r = i * m + j;
            coo.push(r, r, 4.0);
            if i > 0 {
                coo.push(r, r - m, -1.0 - gamma);
            }
            if i + 1 < m {
                coo.push(r, r + m, -1.0 + gamma);
            }
            if j > 0 {
                coo.push(r, r - 1, -1.0 - gamma);
            }
            if j + 1 < m {
                coo.push(r, r + 1, -1.0 + gamma);
            }
        }
    }
    coo.to_csr().unwrap()
}

struct Matrices {
    list: Vec<(&'static str, &'static str, CsrMatrix, Vec<C>)>,
}

impl Matrices {
    fn new(refs: &Refs) -> Self {
        let spectrum = |key: &str| {
            let w = refs.get(key);
            w[0].iter().zip(&w[1]).map(|(&re, &im)| C::new(re, im)).collect::<Vec<C>>()
        };
        let (m, gamma) = (30usize, 0.2f64);
        let root = (1.0 - gamma * gamma).sqrt();
        let exact: Vec<C> = (1..=m)
            .flat_map(|i| (1..=m).map(move |j| C::new(4.0 - 2.0 * root * ((i as f64 * PI / (m + 1) as f64).cos() + (j as f64 * PI / (m + 1) as f64).cos()), 0.0)))
            .collect();
        Matrices {
            list: vec![
                ("scattered", "pseudo-random, n = 300", scattered(300), spectrum("spectrum scattered")),
                ("ramp", "ramp plus skew-symmetric, n = 400", ramp(400), spectrum("spectrum ramp")),
                ("convection", "convection-diffusion, γ = 0.2, n = 900", convection(m, gamma), exact),
            ],
        }
    }
    fn get(&self, key: &str) -> &(&'static str, &'static str, CsrMatrix, Vec<C>) {
        self.list.iter().find(|m| m.0 == key).unwrap()
    }
}

fn which_of(code: &str) -> (EigsWhich, &'static str) {
    match code {
        "LM" => (EigsWhich::LargestMagnitude, "largest \\|λ\\|"),
        "LR" => (EigsWhich::LargestReal, "largest Re λ"),
        "SR" => (EigsWhich::SmallestReal, "smallest Re λ"),
        "LI" => (EigsWhich::LargestImaginary, "largest \\|Im λ\\|"),
        _ => unreachable!(),
    }
}

/// The largest distance from a computed value to the exact spectrum, relative to its radius.
fn error(found: &[C], exact: &[C]) -> f64 {
    let radius = exact.iter().fold(0.0f64, |m, z| m.max(z.norm()));
    found.iter().map(|z| exact.iter().map(|w| (z - w).norm()).fold(f64::INFINITY, f64::min)).fold(0.0, f64::max) / radius
}

/// Checks that `found` are the six values most preferred by `key`, up to ties.
fn check_wanted(found: &[C], exact: &[C], key: impl Fn(C) -> f64, what: &str) {
    let mut want: Vec<f64> = exact.iter().map(|&z| key(z)).collect();
    want.sort_by(|a, b| b.total_cmp(a));
    let mut got: Vec<f64> = found.iter().map(|&z| key(z)).collect();
    got.sort_by(|a, b| b.total_cmp(a));
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-7, "{what}: wanted {:?}, found {:?}", &want[..6], got);
    }
}

fn preference(which: EigsWhich, z: C) -> f64 {
    match which {
        EigsWhich::LargestMagnitude => z.norm(),
        EigsWhich::LargestReal => z.re,
        EigsWhich::SmallestReal => -z.re,
        EigsWhich::LargestImaginary => z.im.abs(),
        _ => unreachable!(),
    }
}

fn table(refs: &Refs, mats: &Matrices) {
    let ones = |n: usize| EigsOptions { v0: Some(vec![1.0; n]), ..Default::default() };
    let mut rows = Vec::new();
    let cases = [("scattered", "LM"), ("scattered", "LR"), ("scattered", "SR"), ("scattered", "LI"), ("ramp", "LM"), ("ramp", "LR"), ("ramp", "LI"), ("convection", "LM"), ("convection", "SR")];
    for (key, code) in cases {
        let (_, label, a, exact) = mats.get(key);
        let (which, wlabel) = which_of(code);
        let r = eigs(a, 6, which, &ones(a.nrows)).unwrap();
        assert!(r.converged, "{key} {code}");
        let w = refs.get(&format!("eigs {key} {code}"));
        let scipy: Vec<C> = w[0].iter().zip(&w[1]).map(|(&re, &im)| C::new(re, im)).collect();
        check_wanted(&r.values, exact, |z| preference(which, z), &format!("library {key} {code}"));
        check_wanted(&scipy, exact, |z| preference(which, z), &format!("SciPy {key} {code}"));
        rows.push(vec![label.to_string(), wlabel.to_string(), fmt_diff(error(&r.values, exact)), fmt_diff(error(&scipy, exact)), format!("{} / {}", r.matvecs, w[2][0])]);
    }
    for (key, sigma) in [("scattered", 0.3), ("convection", 1.0)] {
        let (_, label, a, exact) = mats.get(key);
        let r = eigs_shift_invert(a, 6, sigma, &ones(a.nrows)).unwrap();
        assert!(r.converged, "shift {key}");
        let w = refs.get(&format!("shift {key} {sigma}"));
        let scipy: Vec<C> = w[0].iter().zip(&w[1]).map(|(&re, &im)| C::new(re, im)).collect();
        let near = |z: C| -(z - sigma).norm();
        check_wanted(&r.values, exact, near, &format!("library shift {key}"));
        check_wanted(&scipy, exact, near, &format!("SciPy shift {key}"));
        rows.push(vec![label.to_string(), format!("nearest σ = {sigma}"), fmt_diff(error(&r.values, exact)), fmt_diff(error(&scipy, exact)), format!("{} solves", r.matvecs)]);
    }
    write(CHAPTER, "sparse_eigs.md", &md_table(&["matrix", "six eigenvalues with", "error, library", "error, SciPy", "products with A (library / SciPy)"], &rows));
}

fn spectrum_figure(mats: &Matrices) {
    let (_, _, a, exact) = mats.get("scattered");
    let mut plot = Plot::new("Eigenvalues of a pseudo-random sparse matrix (n = 300) found by eigs", "Re λ", "Im λ");
    plot.legend = Legend::TopRight;
    plot.x_range = Some((-2.5, 2.5));
    plot.y_range = Some((-1.6, 2.6));
    plot.series.push(Series::new("whole spectrum", exact.iter().map(|z| (z.re, z.im)).collect(), Style::Markers, 7));
    let o = EigsOptions::default();
    for (k, code) in ["LR", "SR", "LI"].into_iter().enumerate() {
        let (which, label) = which_of(code);
        let label = label.replace('\\', "");
        let r = eigs(a, 6, which, &o).unwrap();
        plot.series.push(Series::new(&label, r.values.iter().map(|z| (z.re, z.im)).collect(), Style::Markers, k + 1));
    }
    let r = eigs_shift_invert(a, 6, 0.3, &o).unwrap();
    plot.series.push(Series::new("nearest σ = 0.3", r.values.iter().map(|z| (z.re, z.im)).collect(), Style::Markers, 4));
    write(CHAPTER, "sparse_eigs_spectrum.svg", &plot.render());
}
