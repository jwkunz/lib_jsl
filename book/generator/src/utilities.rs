//! Tables and figures for release 1.16.0: the FFT helpers (FFT chapter), log-domain and inverse
//! special functions (special functions chapter), and robust lines, circular statistics and
//! distances (statistics chapter), each compared with SciPy.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_ffts::{TrigNormalization, TrigTransformType, dctn, dstn, fftfreq, irfftn, next_fast_len, rfftfreq, rfftn};
use lib_jsl_special_functions as sf;
use lib_jsl_statistics::distance::{Metric, cdist, pdist};
use lib_jsl_statistics::location_scale::{boxcox, boxcox_llf, circmean, circstd, circvar, mode, trim_mean, zscore};
use lib_jsl_statistics::regression::{linregress, siegelslopes, theilslopes};
use ndarray::{ArrayD, IxDyn};
use std::f64::consts::TAU;

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/utilities_reference.txt"));
    fft_table(&refs);
    special_table(&refs);
    log_ndtr_figure();
    statistics_table(&refs);
    robust_figure(&refs);
}

/// The largest elementwise relative difference (absolute where the reference is zero).
fn rel(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    got.iter().zip(want).fold(0.0f64, |m, (g, w)| {
        let d = if g == w { 0.0 } else { (g - w).abs() / if *w == 0.0 { 1.0 } else { w.abs() } };
        m.max(d)
    })
}

/// The largest difference relative to the largest reference value.
fn rel_scaled(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    got.iter().zip(want).fold(0.0f64, |m, (g, w)| m.max((g - w).abs())) / scale
}

fn fft_table(refs: &Refs) {
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row = |what: String, d: f64| rows.push(vec![what, fmt_diff(d)]);
    for n in [7, 8] {
        let d = rel(&fftfreq(n, 0.1), refs.values(&format!("fftfreq {n}"))).max(rel(&rfftfreq(n, 0.1), refs.values(&format!("rfftfreq {n}"))));
        row(format!("`fftfreq` and `rfftfreq`, n = {n}"), d);
    }
    let targets: Vec<usize> = refs.values("fast targets").iter().map(|&t| t as usize).collect();
    for (real, key) in [(true, "real"), (false, "complex")] {
        let got: Vec<f64> = targets.iter().map(|&t| next_fast_len(t, real) as f64).collect();
        let want = refs.values(&format!("next_fast_len {key}"));
        let shown: Vec<String> = got.iter().map(|v| format!("{v}")).collect();
        row(format!("`next_fast_len` ({key}) of 7, 13, 97, …, 999983: {}", shown.join(", ")), rel(&got, want));
    }
    for shape in [[4usize, 5, 6], [3, 4, 7]] {
        let x = ArrayD::from_shape_fn(IxDyn(&shape), |i| (0.3 * (i[0] * 35 + i[1] * 7 + i[2]) as f64).sin() + 0.1 * i[0] as f64);
        let r = rfftn(&x).unwrap();
        let w = refs.get(&format!("rfftn {}x{}x{}", shape[0], shape[1], shape[2]));
        let got: Vec<f64> = r.iter().map(|z| z.re).chain(r.iter().map(|z| z.im)).collect();
        let want: Vec<f64> = w[0].iter().chain(&w[1]).copied().collect();
        let back = irfftn(&r, &shape).unwrap();
        let round = back.iter().zip(x.iter()).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        row(format!("`rfftn`, {} × {} × {} (round trip through `irfftn`: {})", shape[0], shape[1], shape[2], fmt_diff(round)), rel_scaled(&got, &want));
    }
    let x2 = ArrayD::from_shape_fn(IxDyn(&[5, 6]), |i| (0.7 * (i[0] * 6 + i[1]) as f64).cos() + 0.1 * i[0] as f64);
    let kinds = [(TrigTransformType::I, 1), (TrigTransformType::II, 2), (TrigTransformType::III, 3), (TrigTransformType::IV, 4)];
    let (mut worst_c, mut worst_s) = (0.0f64, 0.0f64);
    for (kind, k) in kinds {
        for (norm, nn) in [(TrigNormalization::Backward, "backward"), (TrigNormalization::Ortho, "ortho")] {
            worst_c = worst_c.max(rel_scaled(&dctn(&x2, kind, norm).unwrap().iter().copied().collect::<Vec<_>>(), refs.values(&format!("dctn {k} {nn}"))));
            worst_s = worst_s.max(rel_scaled(&dstn(&x2, kind, norm).unwrap().iter().copied().collect::<Vec<_>>(), refs.values(&format!("dstn {k} {nn}"))));
        }
    }
    row("`dctn`, types I–IV, both normalisations, 5 × 6".into(), worst_c);
    row("`dstn`, types I–IV, both normalisations, 5 × 6".into(), worst_s);
    write("fft", "helpers_scipy.md", &md_table(&["computation (SciPy reference)", "largest difference (relative)"], &rows));
}

fn special_table(refs: &Refs) {
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row = |what: &str, d: f64| rows.push(vec![what.to_string(), fmt_diff(d)]);
    let lse = [sf::logsumexp(&[1000.0, 999.0, -1000.0]), sf::logsumexp(&[-1000.0, -1001.0]), sf::logsumexp(&(0..13).map(|i| -3.0 + 0.5 * i as f64).collect::<Vec<_>>())];
    let (l, s) = sf::logsumexp_weighted(&[0.0, 1.0, 2.0], &[1.0, -3.0, 1.0]);
    row("`logsumexp` (overflowing, underflowing and ordinary terms; signed weights)", rel(&lse, refs.values("logsumexp")).max(rel(&[l, s], refs.values("logsumexp signed"))));
    let grid: Vec<f64> = (0..61).map(|i| -50.0 + i as f64).collect();
    row("`log_ndtr` on −50, −49, …, 10", rel(&grid.iter().map(|&x| sf::log_ndtr(x)).collect::<Vec<_>>(), refs.values("log_ndtr")));
    let xs = [-800.0, -40.0, -5.0, -0.3, 0.0, 0.7, 5.0, 40.0, 800.0];
    row("`expit` and `log_expit` on ±800, ±40, ±5, …", rel(&xs.map(sf::expit), refs.values("expit")).max(rel(&xs.map(sf::log_expit), refs.values("log_expit"))));
    let ps = [1e-12, 0.01, 0.3, 0.5, 0.9, 1.0 - 1e-9];
    row("`logit` from 10⁻¹² to 1 − 10⁻⁹", rel(&ps.map(sf::logit), refs.values("logit")));
    let pos_x = [0.1, 0.5, 1.0, 3.7, 10.0, 55.5];
    let worst = (1..=4).map(|n| rel(&pos_x.map(|x| sf::polygamma(n, x)), &refs.values(&format!("polygamma {n}"))[..6])).fold(0.0, f64::max);
    row("`polygamma`, orders 1–4, x from 0.1 to 55.5", worst);
    let worst = (1..=4).map(|n| rel(&[-2.5, -0.3].map(|x| sf::polygamma(n, x)), &refs.values(&format!("polygamma {n}"))[6..])).fold(0.0, f64::max);
    row("`polygamma`, orders 1–4, x = −2.5 and −0.3 (cancellation in the shift; see the text)", worst);
    let mg = [(3.5, 1), (2.0, 2), (7.25, 3), (20.0, 10)].map(|(a, d)| sf::multigammaln(a, d));
    row("`multigammaln`, dimensions 1, 2, 3, 10", rel(&mg, refs.values("multigammaln")));
    row(&format!("`bernoulli`, B₀ … B₃₀, against exact rationals (SciPy's own error: {})", fmt_diff(rel(refs.values("bernoulli"), refs.values("bernoulli exact")))), rel(&sf::bernoulli(30), refs.values("bernoulli exact")));
    row(&format!("`euler`, E₀ … E₃₀, against exact integers (SciPy's own error: {})", fmt_diff(rel(refs.values("euler"), refs.values("euler exact")))), rel(&sf::euler(30), refs.values("euler exact")));
    let avals = [0.1, 0.5, 1.0, 2.5, 10.0, 100.0];
    let pvals = [1e-10, 0.01, 0.3, 0.5, 0.9, 0.999999];
    let gi: Vec<f64> = avals.iter().flat_map(|&a| pvals.iter().map(move |&p| sf::gammaincinv(a, p))).collect();
    row("`gammaincinv`, a from 0.1 to 100, p from 10⁻¹⁰ to 1 − 10⁻⁶", rel(&gi, refs.values("gammaincinv")));
    let gci: Vec<f64> = avals.iter().flat_map(|&a| pvals.iter().map(move |&p| sf::gammainccinv(a, p))).collect();
    row("`gammainccinv`, the same grid", rel(&gci, refs.values("gammainccinv")));
    let abvals = [(0.2, 0.3), (0.5, 0.5), (1.0, 3.0), (2.0, 5.0), (30.0, 0.7), (100.0, 200.0)];
    let bi: Vec<f64> = abvals.iter().flat_map(|&(a, b)| pvals.iter().map(move |&p| sf::betaincinv(a, b, p))).collect();
    row("`betaincinv`, six (a, b) pairs, the same p", rel(&bi, refs.values("betaincinv")));
    let bl = [-1.5, 0.0, 0.5, 2.0];
    let bc: Vec<f64> = [0.5, 1.0, 3.2, 10.0].iter().flat_map(|&x| bl.iter().map(move |&l| sf::boxcox(x, l))).collect();
    let bc1: Vec<f64> = [1e-10, 0.3, 5.0].iter().flat_map(|&x| bl.iter().map(move |&l| sf::boxcox1p(x, l))).collect();
    row("`boxcox` and `boxcox1p`", rel(&bc, refs.values("boxcox")).max(rel(&bc1, refs.values("boxcox1p"))));
    write("special_functions", "log_domain_scipy.md", &md_table(&["computation (SciPy reference)", "largest difference (relative)"], &rows));
}

fn log_ndtr_figure() {
    let mut plot = Plot::new("Relative error of ln Φ(x) computed as the log of the CDF", "x", "relative error (capped at 1)");
    plot.log_y = true;
    plot.legend = Legend::TopRight;
    let xs: Vec<f64> = (0..=228).map(|i| -45.0 + 0.25 * i as f64).collect();
    let naive: Vec<(f64, f64)> = xs
        .iter()
        .map(|&x| {
            let good = sf::log_ndtr(x);
            let bad = sf::normal_cdf(x, 0.0, 1.0).ln();
            let err = if bad.is_finite() { ((bad - good) / good).abs().min(1.0) } else { 1.0 };
            (x, err.max(1e-17))
        })
        .collect();
    plot.series.push(Series::new("ln(normal_cdf(x))", naive, Style::Line, 1));
    plot.series.push(Series::new("one ulp", vec![(-45.0, f64::EPSILON), (12.0, f64::EPSILON)], Style::Dashed, 7));
    plot.y_range = Some((1e-17, 10.0));
    write("special_functions", "log_ndtr.svg", &plot.render());
}

fn statistics_table(refs: &Refs) {
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row = |what: &str, d: f64| rows.push(vec![what.to_string(), fmt_diff(d)]);
    let xr: Vec<f64> = (0..40).map(|k| k as f64 / 3.0).collect();
    let yr = refs.values("input yr");
    let lr = linregress(&xr, yr).unwrap();
    row("`linregress`: slope, intercept, r, p-value and both standard errors", rel(&[lr.slope, lr.intercept, lr.rvalue, lr.pvalue, lr.stderr, lr.intercept_stderr], refs.values("linregress")));
    for alpha in [0.95, 0.9] {
        let t = theilslopes(yr, &xr, alpha).unwrap();
        row(&format!("`theilslopes` with its {:.0} % interval", alpha * 100.0), rel(&[t.slope, t.intercept, t.low_slope, t.high_slope], refs.values(&format!("theilslopes {alpha}"))));
    }
    let xt: Vec<f64> = (0..30).map(|k| (k / 3) as f64).collect();
    let t = theilslopes(refs.values("input yt"), &xt, 0.95).unwrap();
    row("`theilslopes`, tied x and y values", rel(&[t.slope, t.intercept, t.low_slope, t.high_slope], refs.values("theilslopes ties")));
    let s = siegelslopes(yr, &xr).unwrap();
    row("`siegelslopes`", rel(&[s.slope, s.intercept], refs.values("siegelslopes")));
    let sk = refs.values("input skewed");
    let tm: Vec<f64> = [0.0, 0.1, 0.25, 0.4].iter().map(|&p| trim_mean(sk, p).unwrap()).collect();
    row("`trim_mean`, cutting 0, 10, 25 and 40 %", rel(&tm, refs.values("trim_mean")));
    let z = refs.get("zscore");
    row("`zscore`, ddof 0 and 1", rel(&zscore(&sk[..10], 0).unwrap(), &z[0]).max(rel(&zscore(&sk[..10], 1).unwrap(), &z[1])));
    let rounded: Vec<f64> = sk.iter().map(|v| v.round_ties_even()).collect();
    let (m, c) = mode(&rounded).unwrap();
    row("`mode` of the rounded values", rel(&[m, c as f64], refs.values("mode")));
    let ang = refs.values("input angles");
    row("`circmean`, `circvar`, `circstd` in radians", rel(&[circmean(ang, 0.0, TAU).unwrap(), circvar(ang, 0.0, TAU).unwrap(), circstd(ang, 0.0, TAU).unwrap()], refs.values("circular")));
    let deg: Vec<f64> = ang.iter().map(|a| a.to_degrees()).collect();
    row("the same in degrees", rel(&[circmean(&deg, 0.0, 360.0).unwrap(), circvar(&deg, 0.0, 360.0).unwrap(), circstd(&deg, 0.0, 360.0).unwrap()], refs.values("circular degrees")));
    let pos = refs.values("input positive");
    let (bc, lam) = boxcox(pos).unwrap();
    let want = refs.values("boxcox_normmax");
    let gain = boxcox_llf(lam, pos) - boxcox_llf(want[0], pos);
    row(&format!("`boxcox`: λ = {lam:.9} against SciPy's {:.9}, absolute difference (log-likelihood {} higher than at SciPy's λ)", want[0], fmt_diff(gain)), (lam - want[0]).abs());
    row("`boxcox`: the transformed data", rel_scaled(&bc, refs.values("boxcox data")));
    let p = refs.get("input points");
    let pts: Vec<Vec<f64>> = p[0].chunks(4).map(|c| c.to_vec()).collect();
    let vi: Vec<Vec<f64>> = refs.values("input vi").chunks(4).map(|c| c.to_vec()).collect();
    let var: Vec<f64> = (0..4).map(|j| {
        let col: Vec<f64> = pts.iter().map(|r| r[j]).collect();
        lib_jsl_statistics::descriptive::variance(&col, 1).unwrap()
    }).collect();
    let metrics = [
        ("euclidean", Metric::Euclidean), ("sqeuclidean", Metric::SqEuclidean), ("cityblock", Metric::Cityblock), ("chebyshev", Metric::Chebyshev),
        ("minkowski", Metric::Minkowski(3.0)), ("seuclidean", Metric::SEuclidean(var)), ("mahalanobis", Metric::Mahalanobis(vi)), ("cosine", Metric::Cosine),
        ("correlation", Metric::Correlation), ("canberra", Metric::Canberra), ("braycurtis", Metric::BrayCurtis), ("hamming", Metric::Hamming), ("jaccard", Metric::Jaccard),
    ];
    let worst = metrics.iter().map(|(name, m)| rel(&pdist(&pts, m).unwrap(), refs.values(&format!("pdist {name}")))).fold(0.0, f64::max);
    row("`pdist`, all 13 metrics, 7 points in 4 dimensions", worst);
    let c: Vec<f64> = cdist(&pts[..3], &pts[3..], &Metric::Euclidean).unwrap().concat();
    row("`cdist`, 3 × 4 points", rel(&c, refs.values("cdist euclidean")));
    write("statistics", "robust_scipy.md", &md_table(&["computation (SciPy reference)", "largest difference (relative)"], &rows));
}

fn robust_figure(refs: &Refs) {
    let xr: Vec<f64> = (0..40).map(|k| k as f64 / 3.0).collect();
    let yr = refs.values("input yr");
    let mut plot = Plot::new("Straight lines through data with three outliers", "x", "y");
    plot.legend = Legend::TopLeft;
    plot.series.push(Series::new("data", xr.iter().zip(yr).map(|(&a, &b)| (a, b)).collect(), Style::Markers, 7));
    let line = |slope: f64, intercept: f64| vec![(0.0, intercept), (13.0, intercept + 13.0 * slope)];
    let lr = linregress(&xr, yr).unwrap();
    let t = theilslopes(yr, &xr, 0.95).unwrap();
    let s = siegelslopes(yr, &xr).unwrap();
    plot.series.push(Series::new("true line y = 2x + 1", line(2.0, 1.0), Style::Dashed, 6));
    plot.series.push(Series::new(&format!("least squares (slope {:.2})", lr.slope), line(lr.slope, lr.intercept), Style::Line, 1));
    plot.series.push(Series::new(&format!("Theil-Sen (slope {:.2})", t.slope), line(t.slope, t.intercept), Style::Line, 0));
    plot.series.push(Series::new(&format!("Siegel (slope {:.2})", s.slope), line(s.slope, s.intercept), Style::Line, 2));
    write("statistics", "robust_lines.svg", &plot.render());
}

