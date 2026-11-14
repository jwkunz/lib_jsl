//! Figures and tables for "Signal-processing utilities" (the spectral chapter): the SciPy
//! comparison, a minimum-phase conversion, and coherence.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_dsp::discrete::signal_tools::*;
use lib_jsl_dsp::discrete::spectral::welch::{Detrend, SpectralScaling};

const CHAPTER: &str = "spectral";

pub fn generate() {
    scipy_reference();
    minimum_phase_figure();
}

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "lengths differ");
    let scale = b.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs())) / scale
}

fn scipy_reference() {
    let refs = Refs::parse(include_str!("../data/signal_tools_reference.txt"));
    let v = |k: &str| refs.values(k).to_vec();
    let mut rows = Vec::new();
    let mut row = |what: &str, d: f64| rows.push(vec![what.to_string(), fmt_diff(d)]);
    let w = v("data w");
    row("`group_delay`, Chebyshev I, order 6", max_abs(&group_delay(&v("data b"), &v("data a"), &w).unwrap(), &v("group_delay iir")));
    row("`group_delay`, linear-phase FIR, 41 taps", max_abs(&group_delay(&v("data fir"), &[1.0], &w).unwrap(), &v("group_delay fir")));
    row("`minimum_phase`, homomorphic, half (31 → 16 taps)", max_abs(&minimum_phase(&v("data remez"), true, None).unwrap(), &v("minimum_phase half")));
    row("`minimum_phase`, homomorphic, full length", max_abs(&minimum_phase(&v("data remez"), false, None).unwrap(), &v("minimum_phase full")));
    row("`firls`, lowpass, 41 taps", max_abs(&firls(41, &[0.0, 0.3, 0.4, 1.0], &[1.0, 1.0, 0.0, 0.0], None, None).unwrap(), &v("firls lowpass")));
    row("`firls`, weighted bandpass, 63 taps", max_abs(&firls(63, &[0.0, 0.1, 0.2, 0.4, 0.5, 1.0], &[0.0, 0.0, 1.0, 1.0, 0.0, 0.0], Some(&[10.0, 1.0, 10.0]), None).unwrap(), &v("firls bandpass")));
    row("`firls`, differentiator (sloped band), 31 taps", max_abs(&firls(31, &[0.0, 0.9], &[0.0, 0.9 * std::f64::consts::PI], None, Some(2.0)).unwrap(), &v("firls differentiator")));
    let x = v("data x");
    for q in [2usize, 5] {
        row(&format!("`decimate` by {q}, zero-phase IIR"), max_abs(&decimate(&x, q, true).unwrap(), &v(&format!("decimate {q}"))));
    }
    row("`decimate` by 3, causal IIR", max_abs(&decimate(&x, 3, false).unwrap(), &v("decimate 3 causal")));
    for num in [150usize, 401, 1000] {
        row(&format!("`resample`, 400 → {num} samples"), max_abs(&resample(&x, num).unwrap(), &v(&format!("resample {num}"))));
    }
    for k in [3usize, 7] {
        row(&format!("`medfilt`, kernel {k}, signal with spikes"), max_abs(&medfilt(&v("data spiky"), k).unwrap(), &v(&format!("medfilt {k}"))));
    }
    row("`envelope` (`abs(hilbert(x))`)", max_abs(&envelope(&x).unwrap(), &v("envelope")));
    let (y, win) = (v("data y"), v("data window"));
    let p = csd(&x, &y, 10.0, &win, 32, Detrend::Constant, SpectralScaling::Density).unwrap();
    let want = refs.get("csd");
    let got: Vec<f64> = p.values.iter().map(|z| z.re).chain(p.values.iter().map(|z| z.im)).collect();
    let wanted: Vec<f64> = want[0].iter().chain(&want[1]).copied().collect();
    row("`csd`, Hann window of 64, overlap 32", max_abs(&got, &wanted));
    let (_, c) = coherence(&x, &y, 10.0, &win, 32, Detrend::Constant).unwrap();
    row("`coherence`", max_abs(&c, &v("coherence")));
    write(CHAPTER, "signal_tools_scipy.md", &md_table(&["computation (SciPy reference)", "largest difference (relative to the largest value)"], &rows));
}

fn minimum_phase_figure() {
    let refs = Refs::parse(include_str!("../data/signal_tools_reference.txt"));
    let h = refs.values("data remez").to_vec();
    let hmin = minimum_phase(&h, true, None).unwrap();
    let mut taps = Plot::new("A linear-phase lowpass and its minimum-phase version", "tap n", "h[n]");
    taps.legend = Legend::TopRight;
    taps.series.push(Series::new("linear phase (31 taps, delay 15)", h.iter().enumerate().map(|(i, &v)| (i as f64, v)).collect(), Style::Stems, 0));
    taps.series.push(Series::new("minimum phase (16 taps)", hmin.iter().enumerate().map(|(i, &v)| (i as f64 + 0.25, v)).collect(), Style::Stems, 1));
    write(CHAPTER, "minimum_phase_taps.svg", &taps.render());
    let w: Vec<f64> = (1..=300).map(|i| i as f64 * std::f64::consts::PI / 301.0).collect();
    let mut gd = Plot::new("Group delay", "ω (radians per sample)", "delay (samples)");
    gd.legend = Legend::TopLeft;
    gd.y_range = Some((-2.0, 20.0));
    gd.x_range = Some((0.0, 0.2 * 2.0 * std::f64::consts::PI));
    gd.series.push(Series::new("linear phase", w.iter().copied().zip(group_delay(&h, &[1.0], &w).unwrap()).collect(), Style::Line, 0));
    gd.series.push(Series::new("minimum phase", w.iter().copied().zip(group_delay(&hmin, &[1.0], &w).unwrap()).collect(), Style::Line, 1));
    write(CHAPTER, "minimum_phase_delay.svg", &gd.render());
}
