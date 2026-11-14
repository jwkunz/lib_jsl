//! Tables and figures for "Partial fractions, zoom FFT and waveforms" (spectral chapter): the
//! SciPy comparison and a zoom FFT against the plain FFT.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_dsp::continuous::lti::{ss2zpk, tf2ss};
use lib_jsl_dsp::discrete::partial_fractions::{RootType, deconvolve, invres, invresz, residue, residuez, tf2zpk, zpk2tf};
use lib_jsl_dsp::discrete::signal_tools::zoom_fft;
use lib_jsl_dsp::discrete::waveforms::{ChirpMethod, chirp, gausspulse, gausspulse_cutoff, sawtooth, square_wave, sweep_poly};
use num_complex::Complex;
use std::f64::consts::PI;

type C = Complex<f64>;
const CHAPTER: &str = "spectral";

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/signal2_reference.txt"));
    scipy_table(&refs);
    zoom_figure();
}

fn cref(refs: &Refs, key: &str) -> Vec<C> {
    let w = refs.get(key);
    if w[0].is_empty() {
        return Vec::new();
    }
    w[0].iter().zip(&w[1]).map(|(&r, &i)| C::new(r, i)).collect()
}

/// The largest difference relative to the largest reference magnitude, in the given order.
fn cdiff(got: &[C], want: &[C]) -> f64 {
    assert_eq!(got.len(), want.len(), "length mismatch {got:?} vs {want:?}");
    let scale = want.iter().fold(0.0f64, |m, z| m.max(z.norm())).max(1e-300);
    got.iter().zip(want).fold(0.0f64, |m, (a, b)| m.max((a - b).norm())) / scale
}

/// The same, after matching each reference value to the nearest unused computed value.
fn cdiff_matched(got: &[C], want: &[C]) -> f64 {
    assert_eq!(got.len(), want.len());
    let mut used = vec![false; got.len()];
    let mut worst = 0.0f64;
    let scale = want.iter().fold(0.0f64, |m, z| m.max(z.norm())).max(1e-300);
    for w in want {
        let (i, d) = got.iter().enumerate().filter(|(i, _)| !used[*i]).map(|(i, g)| (i, (g - w).norm())).min_by(|a, b| a.1.total_cmp(&b.1)).expect("same length");
        used[i] = true;
        worst = worst.max(d);
    }
    worst / scale
}

fn rdiff(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    got.iter().zip(want).fold(0.0f64, |m, (a, b)| m.max((a - b).abs())) / scale
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| a + (b - a) * i as f64 / (n - 1) as f64).collect()
}

fn poly_from_roots_real(roots: &[f64]) -> Vec<f64> {
    let mut p = vec![1.0];
    for &r in roots {
        let mut next = vec![0.0; p.len() + 1];
        for (i, &c) in p.iter().enumerate() {
            next[i] += c;
            next[i + 1] -= c * r;
        }
        p = next;
    }
    p
}

fn scipy_table(refs: &Refs) {
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row = |what: String, d: f64| rows.push(vec![what, fmt_diff(d)]);
    let complex_a = {
        // (s + 0.5)³ (s² + 2s + 5), expanded.
        let cubic = poly_from_roots_real(&[-0.5, -0.5, -0.5]);
        let quad = [1.0, 2.0, 5.0];
        let mut out = vec![0.0; cubic.len() + quad.len() - 1];
        for (i, a) in cubic.iter().enumerate() {
            for (j, b) in quad.iter().enumerate() {
                out[i + j] += a * b;
            }
        }
        out
    };
    let cases: Vec<(&str, Vec<f64>, Vec<f64>, &str)> = vec![
        ("simple", vec![1.0, 2.0], vec![1.0, 4.0, 3.0], "(s + 2)/(s² + 4s + 3)"),
        ("improper", vec![2.0, 3.0, 4.0, 5.0], vec![1.0, 3.0, 2.0], "an improper fraction (direct term k)"),
        ("repeated", vec![1.0, 0.0, 1.0], poly_from_roots_real(&[-1.0, -1.0, -2.0]), "a double pole"),
        ("complex", vec![1.0, 2.0, 3.0], complex_a, "a triple pole and a complex pair"),
    ];
    for (name, b, a, label) in cases {
        let pf = residue(&b, &a, 1e-3, RootType::Average).unwrap();
        let d = cdiff(&pf.r, &cref(refs, &format!("residue {name} r"))).max(cdiff(&pf.p, &cref(refs, &format!("residue {name} p")))).max(if pf.k.is_empty() { 0.0 } else { rdiff(&pf.k, refs.values(&format!("residue {name} k"))) });
        let (bb, aa) = invres(&pf.r, &pf.p, &pf.k, 1e-3, RootType::Average);
        let d2 = cdiff(&bb, &cref(refs, &format!("invres {name} b"))).max(cdiff(&aa, &cref(refs, &format!("invres {name} a"))));
        row(format!("`residue` and `invres`, {label}"), d.max(d2));
    }
    for (name, b, a, label) in [("simple", vec![1.0, -0.5, 0.25], vec![1.0, -0.9, 0.2], "second order"), ("repeated", vec![1.0, 2.0], vec![1.0, -1.0, 0.25], "a double pole at z = 0.5"), ("improper", vec![1.0, 2.0, 3.0, 4.0], vec![1.0, -0.5], "an FIR part (direct term k)")] {
        let pf = residuez(&b, &a, 1e-3, RootType::Average).unwrap();
        let d = cdiff(&pf.r, &cref(refs, &format!("residuez {name} r"))).max(cdiff(&pf.p, &cref(refs, &format!("residuez {name} p")))).max(if pf.k.is_empty() { 0.0 } else { rdiff(&pf.k, refs.values(&format!("residuez {name} k"))) });
        let (bb, aa) = invresz(&pf.r, &pf.p, &pf.k, 1e-3, RootType::Average);
        let d2 = cdiff(&bb, &cref(refs, &format!("invresz {name} b"))).max(cdiff(&aa, &cref(refs, &format!("invresz {name} a"))));
        row(format!("`residuez` and `invresz`, {label}"), d.max(d2));
    }
    let (q, r) = deconvolve(refs.values("input y2"), refs.values("input h")).unwrap();
    row("`deconvolve`, a noisy 44-sample convolution by a 5-tap filter: quotient and remainder".into(), rdiff(&q, refs.values("deconvolve q")).max(rdiff(&r, refs.values("deconvolve r"))));
    let (z, p, k) = tf2zpk(&[2.0, 3.0, 1.0], &[1.0, 0.5, 3.0, 1.0]).unwrap();
    row("`tf2zpk` (zeros and poles matched as sets)".into(), cdiff_matched(&z, &cref(refs, "tf2zpk z")).max(cdiff_matched(&p, &cref(refs, "tf2zpk p"))).max((k - refs.values("tf2zpk k")[0]).abs()));
    let (bb, aa) = zpk2tf(&z, &p, k);
    row("`zpk2tf` back".into(), rdiff(&bb, refs.values("zpk2tf b")).max(rdiff(&aa, refs.values("zpk2tf a"))));
    let sys = tf2ss(&[1.0, 3.0, 2.0], &[1.0, 2.0, 10.0, 4.0]).unwrap();
    let zpk = ss2zpk(&sys, 0).unwrap();
    let (z, p, k) = &zpk[0];
    row("`ss2zpk` (sets)".into(), cdiff_matched(z, &cref(refs, "ss2zpk z")).max(cdiff_matched(p, &cref(refs, "ss2zpk p"))).max((k - refs.values("ss2zpk k")[0]).abs()));
    row("Default time grid of `step` (7 time constants of the slowest mode)".into(), rdiff(&sys.default_response_times(100).unwrap(), refs.values("default times")));
    let fs = 1000.0;
    let x: Vec<f64> = (0..256).map(|i| (2.0 * PI * 100.3 * i as f64 / fs).cos() + 0.5 * (2.0 * PI * 101.7 * i as f64 / fs + 0.3).cos()).collect();
    for endpoint in [false, true] {
        let got = zoom_fft(&x, 95.0, 110.0, 128, fs, endpoint).unwrap();
        row(format!("`zoom_fft`, 95–110 Hz in 128 points, endpoint {endpoint}"), cdiff(&got, &cref(refs, &format!("zoom_fft {}", endpoint as u8))));
    }
    let t = linspace(-1.0, 1.0, 201);
    let arg: Vec<f64> = t.iter().map(|v| 2.0 * PI * 3.0 * v).collect();
    let saw = [1.0, 0.5, 0.25].iter().map(|&w| rdiff(&sawtooth(&arg, w), refs.values(&format!("sawtooth {w}")))).fold(0.0, f64::max);
    row("`sawtooth` (widths 1, 0.5, 0.25) and `square` (duty 0.3)".into(), saw.max(rdiff(&square_wave(&arg, 0.3), refs.values("square 0.3"))));
    let g = gausspulse(&t, 5.0, 0.5, -6.0).unwrap();
    let w = refs.get("gausspulse");
    row("`gausspulse`: in-phase, quadrature, envelope and cutoff time".into(), rdiff(&g.in_phase, &w[0]).max(rdiff(&g.quadrature, &w[1])).max(rdiff(&g.envelope, &w[2])).max(rdiff(&[gausspulse_cutoff(5.0, 0.5, -6.0, -60.0).unwrap()], refs.values("gausspulse cutoff"))));
    let tc = linspace(0.0, 2.0, 201);
    let methods = [("linear 1", ChirpMethod::Linear), ("quadratic 1", ChirpMethod::Quadratic { vertex_zero: true }), ("quadratic 0", ChirpMethod::Quadratic { vertex_zero: false }), ("logarithmic 1", ChirpMethod::Logarithmic), ("hyperbolic 1", ChirpMethod::Hyperbolic)];
    let worst = methods.iter().map(|(k, m)| rdiff(&chirp(&tc, 10.0, 2.0, 40.0, *m, 30.0).unwrap(), refs.values(&format!("chirp {k}")))).fold(0.0, f64::max);
    row("`chirp`, all four sweeps (and both quadratic vertices)".into(), worst);
    row("`sweep_poly`, a cubic frequency law".into(), rdiff(&sweep_poly(&tc, &[0.025, -0.36, 1.25, 2.0], 10.0), refs.values("sweep_poly")));
    write(CHAPTER, "partial_fractions_scipy.md", &md_table(&["computation (SciPy reference)", "largest difference (relative)"], &rows));
}

/// A zoom FFT of two tones 1.4 Hz apart against the 256-point FFT, whose bins are 3.9 Hz apart.
fn zoom_figure() {
    let fs = 1000.0;
    let n = 256;
    let x: Vec<f64> = (0..n).map(|i| (2.0 * PI * 100.3 * i as f64 / fs).cos() + 0.5 * (2.0 * PI * 101.7 * i as f64 / fs + 0.3).cos()).collect();
    let full = zoom_fft(&x, 0.0, fs, n, fs, false).unwrap();
    let bins: Vec<(f64, f64)> = (0..n).map(|k| (k as f64 * fs / n as f64, full[k].norm())).filter(|(f, _)| (88.0..=112.0).contains(f)).collect();
    let m = 400;
    let zoom = zoom_fft(&x, 88.0, 112.0, m, fs, true).unwrap();
    let curve: Vec<(f64, f64)> = (0..m).map(|k| (88.0 + 24.0 * k as f64 / (m - 1) as f64, zoom[k].norm())).collect();
    let mut x4: Vec<f64> = x.clone();
    x4.resize(4 * n, 0.0);
    let long: Vec<f64> = (0..4 * n).map(|i| (2.0 * PI * 100.3 * i as f64 / fs).cos() + 0.5 * (2.0 * PI * 101.7 * i as f64 / fs + 0.3).cos()).collect();
    let zoom_long = zoom_fft(&long, 88.0, 112.0, m, fs, true).unwrap();
    let curve_long: Vec<(f64, f64)> = (0..m).map(|k| (88.0 + 24.0 * k as f64 / (m - 1) as f64, zoom_long[k].norm() / 4.0)).collect();
    let mut plot = Plot::new("Two tones 1.4 Hz apart: the FFT, a zoom FFT, and a longer record", "frequency (Hz)", "|X(f)|");
    plot.legend = Legend::TopLeft;
    plot.series.push(Series::new("256-point FFT bins (3.9 Hz apart)", bins, Style::Stems, 1));
    plot.series.push(Series::new("zoom FFT of the same 256 samples", curve, Style::Line, 0));
    plot.series.push(Series::new("zoom FFT of 1024 samples (÷ 4)", curve_long, Style::Dashed, 2));
    plot.series.push(Series::new("true frequencies", vec![(100.3, 0.0), (101.7, 0.0)], Style::Markers, 6));
    write(CHAPTER, "zoom_fft.svg", &plot.render());
}
