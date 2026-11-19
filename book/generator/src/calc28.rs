//! Tables and figures for release 2.8.0: the Faddeeva, Voigt, Struve, Kelvin and Hankel functions
//! and Bessel zeros (special functions chapter), Halley's method and fixed points (root finding),
//! and isotonic regression, finite-difference gradients and brute-force search (optimisation),
//! compared with SciPy.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_optimization::brute::brute;
use lib_jsl_optimization::finite_differences::{approx_fprime, check_grad};
use lib_jsl_optimization::isotonic::isotonic_regression;
use lib_jsl_root_finding::{FixedPointMethod, fixed_point, halley};
use lib_jsl_special_functions::*;
use num_complex::Complex;
use std::f64::consts::PI;

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/calc28_reference.txt"));
    special_table(&refs);
    special_figures();
    roots_table(&refs);
    roots_figure();
    optim_table(&refs);
    optim_figure(&refs);
}

fn rel(got: f64, want: f64) -> f64 {
    if got == want { 0.0 } else { (got - want).abs() / want.abs() }
}

fn worst(it: impl Iterator<Item = f64>) -> f64 {
    it.fold(0.0, f64::max)
}

fn special_table(refs: &Refs) {
    let mut rows = Vec::new();
    let xs = [-8.0, -3.0, -0.5, 0.0, 0.7, 2.0, 5.5, 12.0];
    let (mut upper, mut real, mut lower) = (0.0f64, 0.0f64, 0.0f64);
    for y in [-2.0, -0.3, 0.0, 1e-3, 0.5, 2.0, 6.0, 20.0] {
        let w = refs.get(&format!("wofz {y}"));
        for (k, &x) in xs.iter().enumerate() {
            let got = wofz(Complex::new(x, y));
            let want = Complex::new(w[0][k], w[1][k]);
            let d = (got - want).norm() / want.norm();
            if y > 0.0 {
                upper = upper.max(d);
            } else if y == 0.0 {
                real = real.max(d);
            } else {
                lower = lower.max(d);
            }
        }
    }
    rows.push(vec!["`wofz`, upper half plane".into(), "40".into(), fmt_diff(upper), "Taylor series near the real axis, Weideman (\\|z\\| < 6), continued fraction beyond".into()]);
    rows.push(vec!["`wofz`, real axis".into(), "8".into(), fmt_diff(real), "from Dawson's integral".into()]);
    rows.push(vec!["`wofz`, lower half plane".into(), "16".into(), fmt_diff(lower), "reflection; relative to \\|w\\|, which grows like e^{y²}".into()]);
    let vx = [-6.0, -2.0, -0.5, 0.0, 0.3, 1.0, 3.0, 10.0];
    let mut v = 0.0f64;
    for (s, g) in [(1.0, 0.0), (0.0, 1.0), (1.0, 0.5), (0.3, 2.0), (2.0, 0.05)] {
        let w = refs.values(&format!("voigt {s} {g}"));
        v = v.max(worst(vx.iter().zip(w).map(|(&x, &r)| rel(voigt_profile(x, s, g), r))));
    }
    rows.push(vec!["`voigt_profile`".into(), "40".into(), fmt_diff(v), "including the pure Gaussian and Lorentzian".into()]);
    let sx = [0.1, 1.0, 3.0, 7.0, 12.0, 20.0, 30.0, 50.0, 100.0];
    let (mut h, mut l, mut l_closed) = (0.0f64, 0.0f64, 0.0f64);
    for nu in [0.0, 1.0, 2.5, -0.5, 5.0, 10.0] {
        let w = refs.get(&format!("struve {nu}"));
        for (k, &x) in sx.iter().enumerate() {
            h = h.max(rel(struve_h(nu, x), w[0][k]));
            if nu == -0.5 {
                // L_{−1/2}(x) = √(2/(πx)) sinh x: judge both against the closed form.
                let exact = (2.0 / (PI * x)).sqrt() * x.sinh();
                l_closed = l_closed.max(rel(w[1][k], exact));
                l = l.max(rel(struve_l(nu, x), exact));
            } else {
                l = l.max(rel(struve_l(nu, x), w[1][k]));
            }
        }
    }
    rows.push(vec!["`struve_h` (H_ν)".into(), "54".into(), fmt_diff(h), "best of four methods by error estimate".into()]);
    rows.push(vec![
        "`struve_l` (L_ν)".into(),
        "54".into(),
        fmt_diff(l),
        format!("at ν = −½ against √(2/(πx)) sinh x, from which SciPy differs by {}", fmt_diff(l_closed)),
    ]);
    let kx = [0.01, 0.1, 0.5, 1.0, 2.8, 5.0, 8.0, 10.0, 20.0, 30.0, 50.0];
    let names = ["ber", "bei", "ker", "kei", "berp", "beip", "kerp", "keip"];
    let w: Vec<&[f64]> = names.iter().map(|n| refs.values(&format!("kelvin {n}"))).collect();
    let (mut small, mut mid) = (0.0f64, 0.0f64);
    for (k, &x) in kx.iter().enumerate() {
        let (be, ke, bep, kep) = kelvin(x);
        let got = [be.re, be.im, ke.re, ke.im, bep.re, bep.im, kep.re, kep.im];
        for p in 0..4 {
            // Relative to the modulus of the complex pair: ber and bei have zeros.
            let want = Complex::new(w[2 * p][k], w[2 * p + 1][k]);
            let d = (Complex::new(got[2 * p], got[2 * p + 1]) - want).norm() / want.norm();
            if (8.0..20.0).contains(&x) { mid = mid.max(d) } else { small = small.max(d) }
        }
    }
    let series = kelvin_series_check();
    rows.push(vec!["`kelvin` (all eight), x ∉ [8, 20)".into(), "64".into(), fmt_diff(small), "trapezoid rules for I₀, I₁, K₀, K₁ at x e^{iπ/4}".into()]);
    rows.push(vec![
        "`kelvin`, x = 8 and 10".into(),
        "16".into(),
        fmt_diff(mid),
        format!("SciPy's expansion; the library agrees with the power series to {}", fmt_diff(series)),
    ]);
    let hx = [0.2, 2.0, 15.0, 60.0];
    let mut hk = 0.0f64;
    for nu in [0.0, 1.5, 7.3] {
        let w = refs.get(&format!("hankel {nu}"));
        for (k, &x) in hx.iter().enumerate() {
            let (a, b) = (hankel1(nu, x), hankel2(nu, x));
            hk = hk.max((a - Complex::new(w[0][k], w[1][k])).norm() / a.norm()).max((b - Complex::new(w[2][k], w[3][k])).norm() / b.norm());
        }
    }
    rows.push(vec!["`hankel1`, `hankel2`".into(), "24".into(), fmt_diff(hk), "J ± iY".into()]);
    let mut z = 0.0f64;
    for n in [0u32, 1, 5, 20] {
        let w = refs.get(&format!("zeros {n}"));
        let got = [jn_zeros(n, 10), jnp_zeros(n, 10), yn_zeros(n, 10), ynp_zeros(n, 10)];
        for (g, r) in got.iter().zip(w) {
            z = z.max(worst(g.iter().zip(r).map(|(a, b)| rel(*a, *b))));
        }
    }
    rows.push(vec!["`jn_zeros`, `jnp_zeros`, `yn_zeros`, `ynp_zeros`".into(), "160".into(), fmt_diff(z), "n = 0, 1, 5, 20; ten zeros each".into()]);
    write("special_functions", "calc28_special.md", &md_table(&["Function", "Values", "Largest relative difference from SciPy", "Notes"], &rows));
}

/// The Kelvin functions at 8 and 10 against their power series, summed in rationals' stead with
/// compensated (Kahan) summation of the alternating terms.
fn kelvin_series_check() -> f64 {
    let mut worst = 0.0f64;
    for x in [8.0f64, 10.0] {
        let q = (x / 2.0).powi(2);
        // ber + i bei = Σ (i q)^k / (k!)².
        let (mut re, mut im, mut c_re, mut c_im) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut t = 1.0f64;
        for k in 0..80 {
            if k > 0 {
                t *= q / (k as f64 * k as f64);
            }
            let (s, target) = match k % 4 {
                0 => (1.0, 0),
                1 => (1.0, 1),
                2 => (-1.0, 0),
                _ => (-1.0, 1),
            };
            let (sum, comp) = if target == 0 { (&mut re, &mut c_re) } else { (&mut im, &mut c_im) };
            let y = s * t - *comp;
            let nt = *sum + y;
            *comp = (nt - *sum) - y;
            *sum = nt;
        }
        let got = kelvin(x).0;
        worst = worst.max((got - Complex::new(re, im)).norm() / Complex::new(re, im).norm());
    }
    worst
}

fn special_figures() {
    let mut plot = Plot::new("Voigt profiles of unit area: Gaussian σ, Lorentzian γ", "x", "V(x; σ, γ)");
    plot.legend = Legend::TopRight;
    let xs: Vec<f64> = (0..=600).map(|i| -6.0 + 12.0 * i as f64 / 600.0).collect();
    for (k, (s, g, label)) in [(1.0, 0.0, "σ = 1, γ = 0 (Gaussian)"), (0.6, 0.4, "σ = 0.6, γ = 0.4"), (0.3, 0.8, "σ = 0.3, γ = 0.8"), (0.0, 1.0, "σ = 0, γ = 1 (Lorentzian)")]
        .into_iter()
        .enumerate()
    {
        plot.series.push(Series::new(label, xs.iter().map(|&x| (x, voigt_profile(x, s, g))).collect(), Style::Line, k));
    }
    write("special_functions", "voigt.svg", &plot.render());

    let mut plot = Plot::new("Kelvin functions and the Struve function H₀", "x", "");
    plot.legend = Legend::BottomLeft;
    plot.y_range = Some((-3.0, 3.0));
    let curve = |f: fn(f64) -> f64, a: f64| (0..=400).map(|i| a + (6.0 - a) * i as f64 / 400.0).map(|x| (x, f(x))).collect::<Vec<_>>();
    plot.series.push(Series::new("ber x", curve(ber, 0.0), Style::Line, 0));
    plot.series.push(Series::new("bei x", curve(bei, 0.0), Style::Line, 1));
    plot.series.push(Series::new("ker x", curve(ker, 0.12), Style::Dashed, 0));
    plot.series.push(Series::new("kei x", curve(kei, 0.0), Style::Dashed, 1));
    plot.series.push(Series::new("H₀(x)", curve(|x| struve_h(0.0, x), 0.0), Style::Line, 2));
    write("special_functions", "kelvin.svg", &plot.render());
}

fn roots_table(refs: &Refs) {
    let mut rows = Vec::new();
    type Fun = fn(f64) -> f64;
    let cases: [(&str, &str, Fun, Fun, Fun); 3] = [
        ("cbrt", "x³ = 10", |x| x * x * x - 10.0, |x| 3.0 * x * x, |x| 6.0 * x),
        ("kepler", "E − 0.9 sin E = 0.3", |e| e - 0.9 * e.sin() - 0.3, |e| 1.0 - 0.9 * e.cos(), |e| 0.9 * e.sin()),
        ("lambert", "w eʷ = 5", |w| w * w.exp() - 5.0, |w| (1.0 + w) * w.exp(), |w| (2.0 + w) * w.exp()),
    ];
    for (key, label, f, fp, fpp) in cases {
        let calls = std::cell::Cell::new(0);
        let r = halley(|x| { calls.set(calls.get() + 1); f(x) }, fp, fpp, 1.0, 1.48e-8).unwrap();
        let want = refs.values(&format!("halley {key}"))[0];
        rows.push(vec![format!("`halley`: {label}"), format!("{r:.15}"), fmt_diff((r - want).abs()), format!("{}", calls.get())]);
    }
    let fp: [(&str, &str, Fun, f64); 3] = [
        ("sqrt", "√(10/(x + 2))", |x| (10.0 / (x + 2.0)).sqrt(), 1.2),
        ("cos", "cos x", f64::cos, 0.5),
        ("logistic", "3.5 x (1 − x), repelling", |x| 3.5 * x * (1.0 - x), 0.7),
    ];
    for (key, label, g, x0) in fp {
        let calls = std::cell::Cell::new(0);
        let r = fixed_point(|x| { calls.set(calls.get() + 1); g(x) }, x0, 1e-8, 500, FixedPointMethod::Del2).unwrap();
        let want = refs.values(&format!("fixed_point {key}"))[0];
        rows.push(vec![format!("`fixed_point`: x = {label}"), format!("{r:.15}"), fmt_diff((r - want).abs()), format!("{}", calls.get())]);
    }
    write("root_finding", "calc28_roots.md", &md_table(&["Problem", "Root or fixed point", "Difference from SciPy", "Function evaluations"], &rows));
}

/// Error against iteration for plain iteration and Steffensen's method on x = cos x, and for
/// Newton's and Halley's methods on x³ = 10.
fn roots_figure() {
    let star = 0.739_085_133_215_160_6;
    let mut plot = Plot::new("Convergence: fixed-point iteration, Steffensen, Newton and Halley", "function evaluations", "error");
    plot.log_y = true;
    plot.legend = Legend::TopRight;
    plot.y_range = Some((1e-16, 10.0));
    let (mut x, mut pts) = (0.5f64, vec![(0.0, (0.5f64 - star).abs())]);
    for k in 1..=60 {
        x = x.cos();
        pts.push((k as f64, (x - star).abs().max(1e-16)));
    }
    plot.series.push(Series::new("iteration x ← cos x", pts, Style::LineMarkers, 0));
    let (mut p0, mut pts) = (0.5f64, vec![(0.0, (0.5f64 - star).abs())]);
    for k in 1..=5 {
        let p1 = p0.cos();
        let p2 = p1.cos();
        let d = p2 - 2.0 * p1 + p0;
        p0 = if d != 0.0 { p0 - (p1 - p0).powi(2) / d } else { p2 };
        pts.push(((2 * k) as f64, (p0 - star).abs().max(1e-16)));
    }
    plot.series.push(Series::new("Steffensen (Δ²) on x = cos x", pts, Style::LineMarkers, 1));
    let root = 10f64.cbrt();
    let (mut x, mut pts) = (1.0f64, vec![(0.0, (1.0f64 - root).abs())]);
    for k in 1..=8 {
        x -= (x * x * x - 10.0) / (3.0 * x * x);
        pts.push(((2 * k) as f64, (x - root).abs().max(1e-16)));
    }
    plot.series.push(Series::new("Newton on x³ = 10", pts, Style::LineMarkers, 2));
    let (mut x, mut pts) = (1.0f64, vec![(0.0, (1.0f64 - root).abs())]);
    for k in 1..=5 {
        let (f, d, d2) = (x * x * x - 10.0, 3.0 * x * x, 6.0 * x);
        let mut s = f / d;
        let adj = s * d2 / d / 2.0;
        if adj.abs() < 1.0 {
            s /= 1.0 - adj;
        }
        x -= s;
        pts.push(((3 * k) as f64, (x - root).abs().max(1e-16)));
    }
    plot.series.push(Series::new("Halley on x³ = 10", pts, Style::LineMarkers, 3));
    write("root_finding", "halley_convergence.svg", &plot.render());
}

fn optim_table(refs: &Refs) {
    let mut rows = Vec::new();
    let y = refs.values("isotonic y");
    let w = refs.values("isotonic w");
    for inc in [true, false] {
        let r = isotonic_regression(y, Some(w), inc).unwrap();
        let want = refs.get(&format!("isotonic {inc}"));
        let dx = worst(r.x.iter().zip(&want[0]).map(|(a, b)| (a - b).abs()));
        let dw = worst(r.weights.iter().zip(&want[1]).map(|(a, b)| (a - b).abs()));
        let same_blocks = r.blocks.iter().map(|&b| b as f64).eq(want[2].iter().copied());
        rows.push(vec![
            format!("`isotonic_regression`, {}", if inc { "increasing" } else { "decreasing" }),
            format!("60 points, {} block{}", r.blocks.len() - 1, if r.blocks.len() == 2 { "" } else { "s" }),
            format!("{} (values), {} (weights)", fmt_diff(dx), fmt_diff(dw)),
            if same_blocks { "identical blocks".into() } else { "blocks differ".into() },
        ]);
    }
    let rosen = |x: &[f64]| (0..x.len() - 1).map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2)).sum::<f64>();
    let rosen_der = |x: &[f64]| {
        let n = x.len();
        let mut g = vec![0.0; n];
        for i in 0..n - 1 {
            g[i] += -400.0 * x[i] * (x[i + 1] - x[i] * x[i]) - 2.0 * (1.0 - x[i]);
            g[i + 1] += 200.0 * (x[i + 1] - x[i] * x[i]);
        }
        g
    };
    let x0 = [-1.2, 1.0, 0.7, -0.4];
    let h = f64::EPSILON.sqrt();
    let a = approx_fprime(rosen, &x0, h);
    let da = worst(a.iter().zip(refs.values("approx_fprime")).map(|(p, q)| rel(*p, *q)));
    rows.push(vec!["`approx_fprime`, Rosenbrock in 4-D".into(), "4 components".into(), fmt_diff(da), "relative; same steps as SciPy".into()]);
    let c = check_grad(rosen, rosen_der, &x0, h);
    let want = refs.values("check_grad")[0];
    rows.push(vec!["`check_grad`, Rosenbrock in 4-D".into(), format!("{c:.6e}"), fmt_diff(rel(c, want)), "relative".into()]);
    let himmel = |x: &[f64]| (x[0] * x[0] + x[1] - 11.0).powi(2) + (x[0] + x[1] * x[1] - 7.0).powi(2);
    let g = brute(himmel, &[(-5.0, 5.0), (-5.0, 5.0)], 20, false).unwrap();
    let want = refs.get("brute grid");
    let dg = (g.x[0] - want[0][0]).abs().max((g.x[1] - want[0][1]).abs()).max((g.fun - want[1][0]).abs());
    rows.push(vec!["`brute`, Himmelblau, 20 × 20 grid".into(), format!("({:.6}, {:.6})", g.x[0], g.x[1]), fmt_diff(dg), "same grid point and value".into()]);
    let f = brute(himmel, &[(-5.0, 5.0), (-5.0, 5.0)], 20, true).unwrap();
    let want = refs.get("brute finish");
    let df = (f.x[0] - want[0][0]).abs().max((f.x[1] - want[0][1]).abs());
    let exact = (f.x[0] - 3.0).abs().max((f.x[1] - 2.0).abs());
    rows.push(vec![
        "`brute`, then Nelder-Mead".into(),
        format!("({:.10}, {:.10})", f.x[0], f.x[1]),
        fmt_diff(df),
        format!("SciPy's `fmin` stops at xtol = 10⁻⁴; the library is {} from (3, 2)", fmt_diff(exact)),
    ]);
    write("optimization", "calc28_optim.md", &md_table(&["Problem", "Result", "Difference from SciPy", "Notes"], &rows));
}

fn optim_figure(refs: &Refs) {
    let y = refs.values("isotonic y");
    let w = refs.values("isotonic w");
    let r = isotonic_regression(y, Some(w), true).unwrap();
    let mut plot = Plot::new("Isotonic regression of a noisy upward walk", "index", "");
    plot.legend = Legend::BottomRight;
    plot.series.push(Series::new("data", y.iter().enumerate().map(|(i, &v)| (i as f64, v)).collect(), Style::Markers, 7));
    let mut steps = Vec::new();
    for k in 0..r.blocks.len() - 1 {
        let (a, b) = (r.blocks[k], r.blocks[k + 1]);
        steps.push((a as f64 - 0.5, r.x[a]));
        steps.push((b as f64 - 0.5, r.x[a]));
    }
    plot.series.push(Series::new("non-decreasing fit (weighted PAVA)", steps, Style::Line, 0));
    write("optimization", "isotonic.svg", &plot.render());
}
