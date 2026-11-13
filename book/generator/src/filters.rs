//! Figures and tables for the DSP chapter on filter design, FIR and IIR filters and adaptive
//! filters.

use lib_jsl_dsp::continuous::filter::bilinear::bilinear;
use lib_jsl_dsp::continuous::filter::design::IirFilterBandType as Band;
use lib_jsl_dsp::continuous::filter::bessel::BesselNorm;
use lib_jsl_dsp::continuous::spectral::freqs::{FreqsWorN, freqs};
use lib_jsl_dsp::discrete::convolve::{ConvolveMethod, ConvolveMode, convolve};
use lib_jsl_dsp::discrete::cross_correlation::{CorrelationMode, cross_correlate};
use lib_jsl_dsp::discrete::filters::bessel::bessel;
use lib_jsl_dsp::discrete::filters::butterworth::{ButterOutput, butter, buttord};
use lib_jsl_dsp::discrete::filters::chebyshev1::{cheb1ord, cheby1};
use lib_jsl_dsp::discrete::filters::chebyshev2::{cheb2ord, cheby2};
use lib_jsl_dsp::discrete::filters::discrete_linear_filter::DiscreteLinearFilter;
use lib_jsl_dsp::discrete::filters::elliptic::{ellip, ellipord};
use lib_jsl_dsp::discrete::filters::firwin::{FirwinPassZero, firwin};
use lib_jsl_dsp::discrete::filters::firwin2::firwin2;
use lib_jsl_dsp::discrete::filters::iir_comb::{IirCombFilterType, iircomb};
use lib_jsl_dsp::discrete::filters::iir_notch::iirnotch;
use lib_jsl_dsp::discrete::filters::iir_peak::iirpeak;
use lib_jsl_dsp::discrete::filters::remez::{RemezFilterType, remez};
use lib_jsl_dsp::discrete::spectral::freqz::{FreqzWorN, freqz};
use lib_jsl_dsp::discrete::windows::WindowType;
use crate::linear_algebra::time;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_dsp::continuous::filter::design::{Zpk, buttap, cheb1ap, cheb2ap, ellipap, besselap, lp2bp_zpk, lp2bs_zpk, lp2hp_zpk};
use lib_jsl_dsp::discrete::fft_convolution::fft_convolve;
use lib_jsl_dsp::discrete::fft_cross_correlation::fft_cross_correlate;
use lib_jsl_dsp::discrete::cross_correlation::cross_correlation_lags;
use lib_jsl_dsp::discrete::filters::butterworth::ButterOutputType;
use lib_jsl_dsp::discrete::filters::derivative_filter::generate_derivative_taps;
use lib_jsl_dsp::discrete::filters::fft_fir::FftFir;
use lib_jsl_dsp::discrete::filters::overlap_and_add_fir::OverlapAddFir;
use lib_jsl_dsp::discrete::filters::adaptive::lms::LmsFilter;
use lib_jsl_dsp::discrete::filters::adaptive::rls::RlsFilter;
use lib_jsl_dsp::discrete::filters::sos::{Section, sosfilt, sosfilt_zi, zpk2sos};
use lib_jsl_dsp::discrete::filters::zero_phase::{Padding, filtfilt, lfilter, lfilter_zi, sosfiltfilt};
use lib_jsl_dsp::discrete::stream_operator::StreamOperator;
use lib_jsl_dsp::discrete::windows::{kaiser_beta, kaiser_estimate_numtaps};
use lib_jsl_random::uniform_generator::UniformGenerator;
use lib_jsl_random::xoshiro256plusplus::Xoshiro256PlusPlus;
use num_complex::Complex;
use std::f64::consts::PI;

const CHAPTER: &str = "filters";

pub fn generate() {
    reference_table();
    convolution_figures();
    analog_figures();
    iir_figures();
    structure_figures();
    fir_figures();
    adaptive_figures();
    sos_figures();
}

fn md_table(header: &[&str], rows: &[Vec<String>]) -> String {
    let mut s = format!("| {} |\n|{}\n", header.join(" | "), header.iter().enumerate().map(|(i, _)| if i == 0 { "---|" } else { "---:|" }).collect::<String>());
    for r in rows {
        s += &format!("| {} |\n", r.join(" | "));
    }
    s
}

/// A numerator and denominator.
type Ba = (Vec<f64>, Vec<f64>);

/// Real numerator and denominator of a `ba` design.
pub(crate) fn ba(out: ButterOutput) -> Ba {
    match out {
        ButterOutput::Ba(r) => (r.beta.iter().map(|c| c.re).collect(), r.alpha.iter().map(|c| c.re).collect()),
        ButterOutput::Zpk(_) => panic!("expected ba output"),
    }
}


/// Second-order sections of a design, through its zeros, poles and gain.
pub(crate) fn sections(out: ButterOutput) -> Vec<Section> {
    match out {
        ButterOutput::Zpk(z) => zpk2sos(&z.z.to_vec(), &z.p.to_vec(), z.k.re).expect("valid design"),
        ButterOutput::Ba(_) => panic!("expected zpk output"),
    }
}

/// The test signal of the second-order-section and zero-phase reference cases.
fn sos_signal() -> Vec<f64> {
    (0..200).map(|i| (0.07 * i as f64).sin() + 0.3 * (0.9 * i as f64).cos() + ((i * 37) % 11) as f64 / 10.0).collect()
}

/// The library's value for each named reference case, as lists matching the reference file.
fn library_case(name: &str) -> Option<Vec<Vec<f64>>> {
    let two = |(b, a): (Vec<f64>, Vec<f64>)| vec![b, a];
    let order = |n: usize, wn: Vec<f64>| vec![vec![n as f64], wn];
    let x: Vec<f64> = (0..64).map(|i| (0.37 * i as f64).cos() + 0.5 * (1.3 * i as f64).sin()).collect();
    let y: Vec<f64> = (1..=8).map(|i| (i as f64).sqrt()).collect();
    Some(match name {
        "butter 4 lowpass 0.2" => two(ba(butter(4, &[0.2], Band::Lowpass, None, None).ok()?)),
        "butter 3 highpass 0.3" => two(ba(butter(3, &[0.3], Band::Highpass, None, None).ok()?)),
        "butter 4 bandpass 0.2-0.5" => two(ba(butter(4, &[0.2, 0.5], Band::Bandpass, None, None).ok()?)),
        "butter 2 bandstop 0.2-0.4" => two(ba(butter(2, &[0.2, 0.4], Band::Bandstop, None, None).ok()?)),
        "cheby1 5 1 dB lowpass 0.3" => two(ba(cheby1(5, 1.0, &[0.3], Band::Lowpass, None, None).ok()?)),
        "cheby1 4 0.5 dB bandpass 0.2-0.5" => two(ba(cheby1(4, 0.5, &[0.2, 0.5], Band::Bandpass, None, None).ok()?)),
        "cheby2 5 40 dB lowpass 0.3" => two(ba(cheby2(5, 40.0, &[0.3], Band::Lowpass, None, None).ok()?)),
        "cheby2 4 30 dB highpass 0.4" => two(ba(cheby2(4, 30.0, &[0.4], Band::Highpass, None, None).ok()?)),
        "ellip 5 1/50 dB lowpass 0.3" => two(ba(ellip(5, 1.0, 50.0, &[0.3], Band::Lowpass, None, None).ok()?)),
        "ellip 4 0.5/40 dB bandpass 0.2-0.5" => two(ba(ellip(4, 0.5, 40.0, &[0.2, 0.5], Band::Bandpass, None, None).ok()?)),
        "ellip 3 1/60 dB bandstop 0.2-0.6" => two(ba(ellip(3, 1.0, 60.0, &[0.2, 0.6], Band::Bandstop, None, None).ok()?)),
        "bessel 4 lowpass 0.2 (phase)" => two(ba(bessel(4, &[0.2], Band::Lowpass, None, Some(BesselNorm::Phase), None).ok()?)),
        "bessel 4 lowpass 0.2 (delay)" => two(ba(bessel(4, &[0.2], Band::Lowpass, None, Some(BesselNorm::Delay), None).ok()?)),
        "bessel 4 lowpass 0.2 (mag)" => two(ba(bessel(4, &[0.2], Band::Lowpass, None, Some(BesselNorm::Mag), None).ok()?)),
        n if n.ends_with("ord lowpass") || n.ends_with("ord bandpass") || n.ends_with("ord bandstop") => {
            let (wp, ws, band): (Vec<f64>, Vec<f64>, Band) = if n.ends_with("lowpass") {
                (vec![0.2], vec![0.3], Band::Lowpass)
            } else if n.ends_with("bandpass") {
                (vec![0.2, 0.5], vec![0.1, 0.6], Band::Bandpass)
            } else {
                (vec![0.1, 0.6], vec![0.2, 0.5], Band::Bandstop)
            };
            match &n[..n.find(' ').unwrap()] {
                "buttord" => buttord(&wp, &ws, 1.0, 40.0, band, None).map(|r| order(r.order, r.wn)).ok()?,
                "cheb1ord" => cheb1ord(&wp, &ws, 1.0, 40.0, band, None).map(|r| order(r.order, r.wn)).ok()?,
                "cheb2ord" => cheb2ord(&wp, &ws, 1.0, 40.0, band, None).map(|r| order(r.order, r.wn)).ok()?,
                _ => ellipord(&wp, &ws, 1.0, 40.0, band, None).map(|r| order(r.order, r.wn)).ok()?,
            }
        }
        "firwin 51 lowpass 0.3 hamming" => vec![firwin(51, &[0.3], None, WindowType::Hamming, FirwinPassZero::True, true, 2.0).ok()?],
        "firwin 51 bandpass 0.2-0.5" => vec![firwin(51, &[0.2, 0.5], None, WindowType::Hamming, FirwinPassZero::False, true, 2.0).ok()?],
        "firwin 61 highpass 0.4 kaiser 6" => vec![firwin(61, &[0.4], None, WindowType::Kaiser { beta: 6.0 }, FirwinPassZero::False, true, 2.0).ok()?],
        "firwin2 51" => vec![firwin2(51, &[0.0, 0.3, 0.5, 1.0], &[1.0, 1.0, 0.0, 0.0], None, Some(WindowType::Hamming), false, 2.0).ok()?],
        "remez 51 lowpass" => vec![remez(51, &[0.0, 0.2, 0.3, 0.5], &[1.0, 0.0], None, RemezFilterType::Bandpass, 25, 16, 1.0).ok()?],
        "remez 41 bandpass weighted" => vec![remez(41, &[0.0, 0.1, 0.15, 0.3, 0.35, 0.5], &[0.0, 1.0, 0.0], Some(&[10.0, 1.0, 10.0]), RemezFilterType::Bandpass, 25, 16, 1.0).ok()?],
        "remez 31 hilbert" => vec![remez(31, &[0.05, 0.45], &[1.0], None, RemezFilterType::Hilbert, 25, 16, 1.0).ok()?],
        "remez 30 differentiator" => vec![remez(30, &[0.0, 0.4], &[1.0], None, RemezFilterType::Differentiator, 25, 16, 1.0).ok()?],
        "iirnotch 0.25 Q 30" => {
            let r = iirnotch(0.25, 30.0, None).ok()?;
            vec![r.beta.iter().map(|c| c.re).collect(), r.alpha.iter().map(|c| c.re).collect()]
        }
        "iirpeak 0.25 Q 30" => {
            let r = iirpeak(0.25, 30.0, None).ok()?;
            vec![r.beta.iter().map(|c| c.re).collect(), r.alpha.iter().map(|c| c.re).collect()]
        }
        "iircomb 20 Hz Q 30 fs 200 notch" => {
            let r = iircomb(20.0, 30.0, Some(IirCombFilterType::Notch), Some(200.0), None).ok()?;
            vec![r.beta.iter().map(|c| c.re).collect(), r.alpha.iter().map(|c| c.re).collect()]
        }
        "bilinear 1/(s+1) fs 2" => {
            let r = bilinear(&[1.0], &[1.0, 1.0], 2.0).ok()?;
            vec![r.beta.iter().map(|c| c.re).collect(), r.alpha.iter().map(|c| c.re).collect()]
        }
        "freqz butter 4, 8 points" => {
            let (b, a) = ba(butter(4, &[0.2], Band::Lowpass, None, None).ok()?);
            let r = freqz(&b, Some(&a), Some(FreqzWorN::NumPoints(8)), None, None, Some(false)).ok()?;
            vec![r.h.iter().map(|c| c.re).collect(), r.h.iter().map(|c| c.im).collect()]
        }
        "freqs Butterworth prototype" => {
            let r = freqs(&[1.0], &[1.0, 2.6131259, 3.4142136, 2.6131259, 1.0], Some(FreqsWorN::Frequencies(vec![0.1, 0.5, 1.0, 2.0, 5.0]))).ok()?;
            vec![r.h.iter().map(|c| c.re).collect(), r.h.iter().map(|c| c.im).collect()]
        }
        "lfilter ellip 4 on 64 samples" => {
            let (b, a) = ba(ellip(4, 0.5, 40.0, &[0.3], Band::Lowpass, None, None).ok()?);
            let mut f = DiscreteLinearFilter::new(&b, &a);
            vec![x.iter().map(|v| f.step(v)).collect()]
        }
        "convolve full" => vec![convolve(&x[..20], &y, ConvolveMode::Full, ConvolveMethod::Direct).ok()?],
        "convolve same" => vec![convolve(&x[..20], &y, ConvolveMode::Same, ConvolveMethod::Direct).ok()?],
        "convolve valid" => vec![convolve(&x[..20], &y, ConvolveMode::Valid, ConvolveMethod::Direct).ok()?],
        "correlate full" => vec![cross_correlate(&x[..20], &y, CorrelationMode::Full, ConvolveMethod::Direct).ok()?],
        "zpk2sos butter 8 lowpass 0.15" => vec![sections(butter(8, &[0.15], Band::Lowpass, Some(ButterOutputType::Zpk), None).ok()?).concat()],
        "zpk2sos ellip 5 1/50 dB lowpass 0.3" => vec![sections(ellip(5, 1.0, 50.0, &[0.3], Band::Lowpass, Some(ButterOutputType::Zpk), None).ok()?).concat()],
        "zpk2sos cheby1 4 0.5 dB bandpass 0.2-0.5" => vec![sections(cheby1(4, 0.5, &[0.2, 0.5], Band::Bandpass, Some(ButterOutputType::Zpk), None).ok()?).concat()],
        "sosfilt_zi butter 8 lowpass 0.15" => vec![sosfilt_zi(&sections(butter(8, &[0.15], Band::Lowpass, Some(ButterOutputType::Zpk), None).ok()?)).ok()?.concat()],
        "sosfilt butter 8 lowpass 0.15" => vec![sosfilt(&sections(butter(8, &[0.15], Band::Lowpass, Some(ButterOutputType::Zpk), None).ok()?), &sos_signal(), None).ok()?.0],
        "sosfiltfilt butter 8 lowpass 0.15" => vec![sosfiltfilt(&sections(butter(8, &[0.15], Band::Lowpass, Some(ButterOutputType::Zpk), None).ok()?), &sos_signal(), Padding::Odd, None).ok()?],
        "lfilter_zi butter 4 lowpass 0.2" => {
            let (b, a) = ba(butter(4, &[0.2], Band::Lowpass, None, None).ok()?);
            vec![lfilter_zi(&b, &a).ok()?]
        }
        n if n.starts_with("filtfilt butter 4 lowpass 0.2") => {
            let (b, a) = ba(butter(4, &[0.2], Band::Lowpass, None, None).ok()?);
            let (padding, padlen) = if n.ends_with("(odd)") { (Padding::Odd, None) } else if n.ends_with("(even)") { (Padding::Even, None) } else { (Padding::Constant, Some(20)) };
            vec![filtfilt(&b, &a, &sos_signal(), padding, padlen).ok()?]
        }
        _ => return None,
    })
}

fn reference_table() {
    let reference = include_str!("../data/filter_reference.txt");
    let mut rows = Vec::new();
    for line in reference.lines() {
        let (name, values) = line.split_once('|').unwrap();
        let want: Vec<Vec<f64>> = values.split(';').map(|l| l.split(',').map(|v| v.parse().unwrap()).collect()).collect();
        let cell = match library_case(name) {
            None => "error".to_string(),
            Some(got) if got.len() != want.len() || got.iter().zip(&want).any(|(g, w)| g.len() != w.len()) => {
                format!("length {:?} against {:?}", got.iter().map(Vec::len).collect::<Vec<_>>(), want.iter().map(Vec::len).collect::<Vec<_>>())
            }
            Some(got) => {
                let scale = want.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
                let err = got.iter().flatten().zip(want.iter().flatten()).map(|(g, w)| (g - w).abs()).fold(0.0, f64::max) / scale;
                if name.contains("ord ") {
                    format!("order {} (SciPy {}), {err:.1e}", got[0][0], want[0][0])
                } else {
                    format!("{err:.1e}")
                }
            }
        };
        rows.push(vec![name.to_string(), cell]);
    }
    write(CHAPTER, "reference.md", &md_table(&["SciPy function and case", "largest difference ÷ largest coefficient"], &rows));
}

// ------------------------------------------------------------------ helpers

fn db(h: Complex<f64>) -> f64 {
    20.0 * h.norm().max(1e-300).log10()
}

/// The digital response at `n` frequencies on [0, 1] (in units of Nyquist).
fn response(b: &[f64], a: &[f64], n: usize) -> Vec<(f64, Complex<f64>)> {
    let w: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let r = freqz(b, Some(a), Some(FreqzWorN::Frequencies(w.iter().map(|f| f * PI).collect())), None, None, None).unwrap();
    w.into_iter().zip(r.h.iter().copied()).collect()
}

fn fir_response(h: &[f64], n: usize) -> Vec<(f64, Complex<f64>)> {
    response(h, &[1.0], n)
}

struct Rng(Xoshiro256PlusPlus);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(Xoshiro256PlusPlus::from_seed(seed))
    }
    fn uniform(&mut self) -> f64 {
        self.0.next_f64()
    }
    fn gauss(&mut self) -> f64 {
        let (u, v) = (1.0 - self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos()
    }
}

fn zpk_analog_response(zpk: &Zpk, w: &[f64]) -> Vec<Complex<f64>> {
    w.iter()
        .map(|&wi| {
            let s = Complex::new(0.0, wi);
            zpk.k * zpk.z.iter().map(|z| s - z).product::<Complex<f64>>() / zpk.p.iter().map(|p| s - p).product::<Complex<f64>>()
        })
        .collect()
}

/// Analogue group delay −dφ/dω from the poles and zeros: a factor `s − c` contributes
/// `Re c / |jω − c|²`, a pole the negative of that. (Exact, with no phase unwrapping.)
fn analog_group_delay(zpk: &Zpk, w: &[f64]) -> Vec<(f64, f64)> {
    let term = |c: &Complex<f64>, wi: f64| c.re / (Complex::new(0.0, wi) - c).norm_sqr();
    w.iter().map(|&wi| (wi, zpk.z.iter().map(|c| term(c, wi)).sum::<f64>() - zpk.p.iter().map(|c| term(c, wi)).sum::<f64>())).collect()
}

/// Digital group delay in samples from the poles and zeros of `k Π(1 − z_i z⁻¹) / Π(1 − p_i z⁻¹)`:
/// a factor `1 − c e^{−jω}` contributes `(|c|² − Re(c e^{−jω})) / |1 − c e^{−jω}|²`.
fn digital_group_delay(z: &[Complex<f64>], p: &[Complex<f64>], w: f64) -> f64 {
    let term = |c: &Complex<f64>| {
        let u = c * Complex::from_polar(1.0, -w);
        (c.norm_sqr() - u.re) / (Complex::new(1.0, 0.0) - u).norm_sqr()
    };
    z.iter().map(term).sum::<f64>() - p.iter().map(term).sum::<f64>()
}

// ------------------------------------------------------------------ convolution

fn convolution_figures() {
    let mut rng = Rng::new(1);
    let n = 20_000;
    let x: Vec<f64> = (0..n).map(|_| rng.gauss()).collect();
    let mut rows = Vec::new();
    for taps in [8usize, 32, 128, 512, 2048] {
        let h: Vec<f64> = (0..taps).map(|_| rng.gauss()).collect();
        let direct = time(|| {
            convolve(&x, &h, ConvolveMode::Full, ConvolveMethod::Direct).unwrap();
        });
        let fft = time(|| {
            fft_convolve(&x, &h, ConvolveMode::Full).unwrap();
        });
        let ola = time(|| {
            let mut f = OverlapAddFir::new(&h, 4 * taps.max(64)).unwrap();
            f.process(&x).unwrap();
            f.flush().unwrap();
        });
        let ols = time(|| {
            let mut f = FftFir::new(&h, 4 * taps.max(64)).unwrap();
            f.process(&x).unwrap();
            f.flush().unwrap();
        });
        rows.push(vec![format!("{taps}"), format!("{:.2}", direct * 1e3), format!("{:.2}", fft * 1e3), format!("{:.2}", ola * 1e3), format!("{:.2}", ols * 1e3)]);
    }
    write(CHAPTER, "convolution_time.md", &md_table(&["filter taps (20 000 samples)", "direct (ms)", "whole-signal FFT (ms)", "overlap-add stream (ms)", "overlap-save stream (ms)"], &rows));
    // Time-delay estimation by cross-correlation.
    let len = 400;
    let delay = 37;
    let clean: Vec<f64> = (0..len + delay).map(|_| rng.gauss()).collect();
    let reference: Vec<f64> = clean[delay..].to_vec();
    let received: Vec<f64> = (0..len).map(|i| clean[i] + 1.5 * rng.gauss()).collect();
    let c = fft_cross_correlate(&received, &reference, CorrelationMode::Full).unwrap();
    let lags = cross_correlation_lags(received.len(), reference.len(), CorrelationMode::Full);
    let mut plot = Plot::new("Cross-correlation of a noisy, delayed copy with the original (SNR −3.5 dB)", "lag (samples)", "correlation");
    plot.series.push(Series::new("", lags.iter().zip(&c).filter(|(l, _)| l.abs() <= 120).map(|(&l, &v)| (l as f64, v)).collect(), Style::Line, 0));
    write(CHAPTER, "xcorr_delay.svg", &plot.render());
}

// ------------------------------------------------------------------ analogue prototypes

fn prototypes() -> Vec<(&'static str, Zpk)> {
    vec![
        ("Butterworth", buttap(5).unwrap()),
        ("Chebyshev I (1 dB)", cheb1ap(5, 1.0).unwrap()),
        ("Chebyshev II (40 dB)", cheb2ap(5, 40.0).unwrap()),
        ("elliptic (1 dB, 40 dB)", ellipap(5, 1.0, 40.0).unwrap()),
        ("Bessel", besselap(5, Some(BesselNorm::Mag)).unwrap()),
    ]
}

fn analog_figures() {
    let w: Vec<f64> = (0..600).map(|i| 10f64.powf(-1.0 + 2.0 * i as f64 / 599.0)).collect();
    let mut mag = Plot::new("Fifth-order analogue prototypes", "ω (rad/s)", "gain (dB)");
    mag.log_x = true;
    mag.y_range = Some((-80.0, 5.0));
    let mut poles = Plot::new("Prototype poles (left half of the s-plane)", "Re s", "Im s");
    let wd: Vec<f64> = (0..400).map(|i| 0.02 + 2.98 * i as f64 / 399.0).collect();
    let mut delay = Plot::new("Group delay of the prototypes", "ω (rad/s)", "group delay (s)");
    delay.y_range = Some((0.0, 12.0));
    for (k, (name, zpk)) in prototypes().into_iter().enumerate() {
        let h = zpk_analog_response(&zpk, &w);
        mag.series.push(Series::new(name, w.iter().zip(&h).map(|(&x, &v)| (x, db(v))).collect(), Style::Line, k));
        poles.series.push(Series::new(name, zpk.p.iter().map(|p| (p.re, p.im)).collect(), Style::Markers, k));
        delay.series.push(Series::new(name, analog_group_delay(&zpk, &wd), Style::Line, k));
    }
    mag.legend = Legend::BottomLeft;
    poles.x_range = Some((-2.4, 1.2));
    poles.y_range = Some((-1.4, 1.4));
    poles.legend = Legend::BottomLeft;
    delay.legend = Legend::TopRight;
    write(CHAPTER, "analog_magnitude.svg", &mag.render());
    write(CHAPTER, "analog_poles.svg", &poles.render());
    write(CHAPTER, "analog_delay.svg", &delay.render());
    // Band transformations of a fourth-order Chebyshev I prototype.
    let proto = cheb1ap(4, 1.0).unwrap();
    let mut plot = Plot::new("Band transformations of a Chebyshev I prototype", "ω (rad/s)", "gain (dB)");
    plot.log_x = true;
    plot.y_range = Some((-60.0, 5.0));
    for (k, (name, z)) in [("lowpass (prototype)", proto.clone()), ("highpass, ω₀ = 1", lp2hp_zpk(&proto, 1.0).unwrap()), ("bandpass, ω₀ = 1, B = 0.5", lp2bp_zpk(&proto, 1.0, 0.5).unwrap()), ("bandstop, ω₀ = 1, B = 0.5", lp2bs_zpk(&proto, 1.0, 0.5).unwrap())].into_iter().enumerate() {
        let h = zpk_analog_response(&z, &w);
        plot.series.push(Series::new(name, w.iter().zip(&h).map(|(&x, &v)| (x, db(v))).collect(), Style::Line, k));
    }
    plot.legend = Legend::BottomLeft;
    write(CHAPTER, "analog_transforms.svg", &plot.render());
    // Orders needed for one specification.
    let rows: Vec<Vec<String>> = [
        ("Butterworth (buttord)", buttord(&[0.2], &[0.3], 1.0, 40.0, Band::Lowpass, None).unwrap().order),
        ("Chebyshev I (cheb1ord)", cheb1ord(&[0.2], &[0.3], 1.0, 40.0, Band::Lowpass, None).unwrap().order),
        ("Chebyshev II (cheb2ord)", cheb2ord(&[0.2], &[0.3], 1.0, 40.0, Band::Lowpass, None).unwrap().order),
        ("elliptic (ellipord)", ellipord(&[0.2], &[0.3], 1.0, 40.0, Band::Lowpass, None).unwrap().order),
    ]
    .iter()
    .map(|(n, o)| vec![n.to_string(), o.to_string()])
    .collect();
    write(CHAPTER, "orders.md", &md_table(&["family", "order for ≤ 1 dB ripple below 0.2 and ≥ 40 dB above 0.3 (× Nyquist)"], &rows));
}

// ------------------------------------------------------------------ digital IIR design

fn iir_figures() {
    // Frequency warping of the bilinear transform (fs = 2, so Nyquist is ω = 1 ... π rad/sample).
    let mut plot = Plot::new("Frequency warping of the bilinear transform", "digital frequency (× Nyquist)", "analogue frequency 2 fₛ tan(πf/fₛ) (× fₛ)");
    plot.series.push(Series::new("bilinear map", (0..200).map(|i| {
        let f = 0.99 * i as f64 / 199.0;
        (f, 2.0 * (PI * f / 2.0).tan() / 2.0)
    }).collect(), Style::Line, 0));
    plot.series.push(Series::new("no warping (ω_a = ω_d)", vec![(0.0, 0.0), (0.99, 0.99 * PI / 2.0)], Style::Dashed, 7));
    plot.y_range = Some((0.0, 8.0));
    plot.legend = Legend::TopLeft;
    write(CHAPTER, "bilinear_warping.svg", &plot.render());
    // The four families at their minimum orders for one specification.
    let designs: Vec<(&str, Ba)> = vec![
        ("Butterworth, order 12", ba(butter(12, &buttord(&[0.2], &[0.3], 1.0, 40.0, Band::Lowpass, None).unwrap().wn, Band::Lowpass, None, None).unwrap())),
        ("Chebyshev I, order 6", ba(cheby1(6, 1.0, &[0.2], Band::Lowpass, None, None).unwrap())),
        ("Chebyshev II, order 6", ba(cheby2(6, 40.0, &cheb2ord(&[0.2], &[0.3], 1.0, 40.0, Band::Lowpass, None).unwrap().wn, Band::Lowpass, None, None).unwrap())),
        ("elliptic, order 4", ba(ellip(4, 1.0, 40.0, &[0.2], Band::Lowpass, None, None).unwrap())),
    ];
    let mut plot = Plot::new("Minimum-order designs for one specification", "frequency (× Nyquist)", "gain (dB)");
    plot.y_range = Some((-80.0, 5.0));
    for (k, (name, (b, a))) in designs.iter().enumerate() {
        plot.series.push(Series::new(name, response(b, a, 800).into_iter().map(|(f, h)| (f, db(h))).collect(), Style::Line, k));
    }
    plot.series.push(Series::new("specification", vec![(0.0, -1.0), (0.2, -1.0), (0.2, -80.0)], Style::Dashed, 6));
    plot.series.push(Series::new("", vec![(0.3, 5.0), (0.3, -40.0), (1.0, -40.0)], Style::Dashed, 6));
    plot.legend = Legend::BottomLeft;
    write(CHAPTER, "iir_designs.svg", &plot.render());
    let mut zoom = Plot::new("Passband detail", "frequency (× Nyquist)", "gain (dB)");
    zoom.x_range = Some((0.0, 0.35));
    zoom.y_range = Some((-3.0, 0.5));
    for (k, (name, (b, a))) in designs.iter().enumerate() {
        zoom.series.push(Series::new(name, response(b, a, 2000).into_iter().filter(|(f, _)| *f <= 0.35).map(|(f, h)| (f, db(h))).collect(), Style::Line, k));
    }
    zoom.legend = Legend::BottomLeft;
    write(CHAPTER, "iir_passband.svg", &zoom.render());
    // Poles and zeros of the elliptic design in the z-plane.
    let ButterOutput::Zpk(z) = ellip(4, 1.0, 40.0, &[0.2], Band::Lowpass, Some(ButterOutputType::Zpk), None).unwrap() else { unreachable!() };
    let mut plot = Plot::new("Elliptic design: zeros on the unit circle, poles inside", "Re z", "Im z");
    plot.series.push(Series::new("unit circle", (0..=200).map(|i| {
        let t = 2.0 * PI * i as f64 / 200.0;
        (t.cos(), t.sin())
    }).collect(), Style::Line, 7));
    plot.series.push(Series::new("zeros", z.z.iter().map(|v| (v.re, v.im)).collect(), Style::Markers, 1));
    plot.series.push(Series::new("poles", z.p.iter().map(|v| (v.re, v.im)).collect(), Style::Markers, 0));
    plot.x_range = Some((-1.9, 1.9));
    plot.y_range = Some((-1.1, 1.1));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "iir_zplane.svg", &plot.render());
    // Group delay of the digital designs, from their poles and zeros.
    let mut plot = Plot::new("Group delay of the designs, and of a Bessel filter", "frequency (× Nyquist)", "group delay (samples)");
    let zpk = |o: ButterOutput| match o {
        ButterOutput::Zpk(z) => (z.z.to_vec(), z.p.to_vec()),
        ButterOutput::Ba(_) => unreachable!(),
    };
    let z = Some(ButterOutputType::Zpk);
    let all = [
        ("Butterworth, order 12", zpk(butter(12, &buttord(&[0.2], &[0.3], 1.0, 40.0, Band::Lowpass, None).unwrap().wn, Band::Lowpass, z, None).unwrap())),
        ("Chebyshev I, order 6", zpk(cheby1(6, 1.0, &[0.2], Band::Lowpass, z, None).unwrap())),
        ("Chebyshev II, order 6", zpk(cheby2(6, 40.0, &cheb2ord(&[0.2], &[0.3], 1.0, 40.0, Band::Lowpass, None).unwrap().wn, Band::Lowpass, z, None).unwrap())),
        ("elliptic, order 4", zpk(ellip(4, 1.0, 40.0, &[0.2], Band::Lowpass, z, None).unwrap())),
        ("Bessel, order 6", zpk(bessel(6, &[0.2], Band::Lowpass, z, Some(BesselNorm::Phase), None).unwrap())),
    ];
    for (k, (name, (zeros, poles))) in all.iter().enumerate() {
        plot.series.push(Series::new(name, (0..600).map(|i| {
            let f = 0.5 * i as f64 / 599.0;
            (f, digital_group_delay(zeros, poles, PI * f))
        }).collect(), Style::Line, k));
    }
    plot.y_range = Some((0.0, 35.0));
    plot.legend = Legend::TopLeft;
    write(CHAPTER, "iir_delay.svg", &plot.render());
}

// ------------------------------------------------------------------ structures, notch, peak, comb

fn structure_figures() {
    let mut plot = Plot::new("Notch and peak filters at 0.25 × Nyquist", "frequency (× Nyquist)", "gain (dB)");
    plot.y_range = Some((-50.0, 3.0));
    for (k, q) in [5.0, 30.0].into_iter().enumerate() {
        let n = iirnotch(0.25, q, None).unwrap();
        let (b, a): (Vec<f64>, Vec<f64>) = (n.beta.iter().map(|c| c.re).collect(), n.alpha.iter().map(|c| c.re).collect());
        plot.series.push(Series::new(&format!("notch, Q = {q}"), response(&b, &a, 1500).into_iter().map(|(f, h)| (f, db(h))).collect(), Style::Line, k));
        let p = iirpeak(0.25, q, None).unwrap();
        let (b, a): (Vec<f64>, Vec<f64>) = (p.beta.iter().map(|c| c.re).collect(), p.alpha.iter().map(|c| c.re).collect());
        plot.series.push(Series::new(&format!("peak, Q = {q}"), response(&b, &a, 1500).into_iter().map(|(f, h)| (f, db(h))).collect(), Style::Dashed, k));
    }
    plot.legend = Legend::BottomRight;
    write(CHAPTER, "notch_peak.svg", &plot.render());
    let c = iircomb(20.0, 30.0, Some(IirCombFilterType::Notch), Some(200.0), None).unwrap();
    let (b, a): (Vec<f64>, Vec<f64>) = (c.beta.iter().map(|v| v.re).collect(), c.alpha.iter().map(|v| v.re).collect());
    let mut plot = Plot::new("Comb notch filter: 20 Hz and its harmonics (fₛ = 200 Hz, Q = 30)", "frequency (Hz)", "gain (dB)");
    plot.series.push(Series::new("", response(&b, &a, 4000).into_iter().map(|(f, h)| (100.0 * f, db(h))).collect(), Style::Line, 0));
    plot.y_range = Some((-50.0, 3.0));
    write(CHAPTER, "comb.svg", &plot.render());
    // Coefficient sensitivity of the direct form: round the denominator to single precision.
    let mut rows = Vec::new();
    for (order, wn) in [(4usize, 0.2), (8, 0.2), (8, 0.05), (12, 0.05), (16, 0.05)] {
        let ButterOutput::Zpk(z) = butter(order, &[wn], Band::Lowpass, Some(ButterOutputType::Zpk), None).unwrap() else { unreachable!() };
        let (_, a) = ba(butter(order, &[wn], Band::Lowpass, None, None).unwrap());
        let rounded: Vec<f64> = a.iter().map(|&v| v as f32 as f64).collect();
        let r_exact = z.p.iter().map(|p| p.norm()).fold(0.0, f64::max);
        let r_rounded = polynomial_root_radius(&rounded);
        rows.push(vec![format!("{order}"), format!("{wn}"), format!("{r_exact:.6}"), format!("{r_rounded:.6}"), if r_rounded >= 1.0 { "unstable".into() } else { "stable".into() }]);
    }
    write(CHAPTER, "sensitivity.md", &md_table(&["Butterworth order", "cutoff (× Nyquist)", "largest pole radius", "after rounding a to f32", ""], &rows));
}

/// The largest root modulus of a real polynomial a[0] zᴺ + … + a[N], by the eigenvalues of its
/// companion matrix (Durand-Kerner iteration, adequate for this table).
fn polynomial_root_radius(a: &[f64]) -> f64 {
    let n = a.len() - 1;
    let c: Vec<Complex<f64>> = a.iter().map(|&v| Complex::new(v / a[0], 0.0)).collect();
    let mut roots: Vec<Complex<f64>> = (0..n).map(|k| Complex::from_polar(0.9, 0.4 + 2.0 * PI * k as f64 / n as f64)).collect();
    for _ in 0..2000 {
        for i in 0..n {
            let z = roots[i];
            let p = c.iter().fold(Complex::new(0.0, 0.0), |acc, &ci| acc * z + ci);
            let d: Complex<f64> = (0..n).filter(|&j| j != i).map(|j| z - roots[j]).product();
            roots[i] = z - p / d;
        }
    }
    roots.iter().map(|r| r.norm()).fold(0.0, f64::max)
}

// ------------------------------------------------------------------ FIR design

fn fir_figures() {
    let mut plot = Plot::new("Window-method lowpass filters, 51 taps, cutoff 0.3 × Nyquist", "frequency (× Nyquist)", "gain (dB)");
    plot.y_range = Some((-110.0, 5.0));
    for (k, (name, window)) in [("rectangular", WindowType::Boxcar), ("Hamming", WindowType::Hamming), ("Blackman", WindowType::Blackman), ("Kaiser, β = 8", WindowType::Kaiser { beta: 8.0 })].into_iter().enumerate() {
        let h = firwin(51, &[0.3], None, window, FirwinPassZero::True, true, 2.0).unwrap();
        plot.series.push(Series::new(name, fir_response(&h, 1200).into_iter().map(|(f, v)| (f, db(v))).collect(), Style::Line, k));
    }
    plot.legend = Legend::TopRight;
    write(CHAPTER, "fir_windows.svg", &plot.render());
    // Kaiser's design formulas against the attenuation achieved.
    let mut rows = Vec::new();
    for (atten, width) in [(40.0, 0.1), (60.0, 0.1), (80.0, 0.05), (100.0, 0.05)] {
        let n = kaiser_estimate_numtaps(atten, width);
        let n = n | 1;
        let beta = kaiser_beta(atten);
        let h = firwin(n, &[0.3], None, WindowType::Kaiser { beta }, FirwinPassZero::True, true, 2.0).unwrap();
        let r = fir_response(&h, 4000);
        let stop = r.iter().filter(|(f, _)| *f >= 0.3 + width / 2.0).map(|(_, v)| -db(*v)).fold(f64::INFINITY, f64::min);
        let pass = r.iter().filter(|(f, _)| *f <= 0.3 - width / 2.0).map(|(_, v)| (v.norm() - 1.0).abs()).fold(0.0, f64::max);
        rows.push(vec![format!("{atten}"), format!("{width}"), format!("{n}"), format!("{beta:.3}"), format!("{stop:.1}"), format!("{:.1}", -20.0 * pass.log10())]);
    }
    write(CHAPTER, "kaiser_design.md", &md_table(&["target attenuation (dB)", "transition width (× Nyquist)", "taps (kaiser_estimate_numtaps)", "β (kaiser_beta)", "stopband attenuation achieved (dB)", "passband ripple achieved (dB below 1)"], &rows));
    // Equiripple against the window method at the same length.
    let n = 41;
    let eq = remez(n, &[0.0, 0.2, 0.3, 1.0], &[1.0, 0.0], None, RemezFilterType::Bandpass, 25, 16, 2.0).unwrap();
    let kw = firwin(n, &[0.25], None, WindowType::Kaiser { beta: kaiser_beta(50.0) }, FirwinPassZero::True, true, 2.0).unwrap();
    let mut plot = Plot::new("41 taps, transition 0.2–0.3 × Nyquist", "frequency (× Nyquist)", "gain (dB)");
    plot.series.push(Series::new("Parks-McClellan (remez)", fir_response(&eq, 1200).into_iter().map(|(f, v)| (f, db(v))).collect(), Style::Line, 0));
    plot.series.push(Series::new("Kaiser window (firwin)", fir_response(&kw, 1200).into_iter().map(|(f, v)| (f, db(v))).collect(), Style::Line, 1));
    plot.y_range = Some((-90.0, 5.0));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "fir_remez.svg", &plot.render());
    // Equiripple attenuation against length.
    let mut rows = Vec::new();
    for n in [21usize, 31, 41, 61, 81] {
        let eq = remez(n, &[0.0, 0.2, 0.3, 1.0], &[1.0, 0.0], None, RemezFilterType::Bandpass, 25, 16, 2.0).unwrap();
        let r = fir_response(&eq, 4000);
        let stop = r.iter().filter(|(f, _)| *f >= 0.3).map(|(_, v)| -db(*v)).fold(f64::INFINITY, f64::min);
        let kw = firwin(n, &[0.25], None, WindowType::Kaiser { beta: kaiser_beta(stop) }, FirwinPassZero::True, true, 2.0).unwrap();
        let rk = fir_response(&kw, 4000);
        let stop_k = rk.iter().filter(|(f, _)| *f >= 0.3).map(|(_, v)| -db(*v)).fold(f64::INFINITY, f64::min);
        rows.push(vec![format!("{n}"), format!("{stop:.1}"), format!("{stop_k:.1}")]);
    }
    write(CHAPTER, "remez_vs_kaiser.md", &md_table(&["taps (transition 0.2–0.3 × Nyquist)", "Parks-McClellan stopband (dB)", "Kaiser window stopband (dB)"], &rows));
    // firwin2: an arbitrary piecewise-linear response.
    let freq = [0.0, 0.2, 0.4, 0.4, 0.7, 1.0];
    let gain = [0.0, 1.0, 0.5, 1.0, 1.0, 0.0];
    let mut plot = Plot::new("Frequency sampling (firwin2): a ramp, a step and a taper", "frequency (× Nyquist)", "gain");
    plot.series.push(Series::new("desired", freq.iter().zip(&gain).map(|(&f, &g)| (f, g)).collect(), Style::Dashed, 6));
    for (k, n) in [31usize, 101].into_iter().enumerate() {
        let h = firwin2(n, &freq, &gain, None, Some(WindowType::Hamming), false, 2.0).unwrap();
        plot.series.push(Series::new(&format!("{n} taps"), fir_response(&h, 1200).into_iter().map(|(f, v)| (f, v.norm())).collect(), Style::Line, k));
    }
    plot.legend = Legend::TopLeft;
    write(CHAPTER, "firwin2.svg", &plot.render());
    // Differentiators.
    let mut plot = Plot::new("FIR differentiators: |H| against the ideal |ω|", "frequency (× Nyquist)", "|H(ω)| (per sample)");
    plot.series.push(Series::new("ideal |ω|", vec![(0.0, 0.0), (1.0, PI)], Style::Dashed, 6));
    for (k, n) in [7usize, 31].into_iter().enumerate() {
        let d = generate_derivative_taps(n, 1, None).unwrap();
        plot.series.push(Series::new(&format!("generate_derivative_taps, {n} taps"), fir_response(&d, 600).into_iter().map(|(f, v)| (f, v.norm())).collect(), Style::Line, k));
    }
    // remez's slope is per cycle per sample, so |ω| = 2π f/fₛ needs a desired slope of 2π.
    let r = remez(31, &[0.0, 0.8], &[2.0 * PI], None, RemezFilterType::Differentiator, 25, 16, 2.0).unwrap();
    plot.series.push(Series::new("remez differentiator to 0.8, 31 taps", fir_response(&r, 600).into_iter().map(|(f, v)| (f, v.norm())).collect(), Style::Line, 2));
    plot.legend = Legend::TopLeft;
    write(CHAPTER, "differentiators.svg", &plot.render());
}

// ------------------------------------------------------------------ adaptive filters

fn adaptive_figures() {
    let plant = [0.8, -0.4, 0.25, 0.1, -0.05, 0.03, -0.02, 0.01];
    let m = plant.len();
    let runs = 50;
    let steps = 3000;
    for (colored, file, title) in [(false, "adaptive_white.svg", "Learning curves, white input"), (true, "adaptive_coloured.svg", "Learning curves, coloured input (eigenvalue spread ≈ 100)")] {
        let mut curves = vec![vec![0.0; steps]; 3];
        for run in 0..runs {
            let mut rng = Rng::new(100 + run as u64);
            let mut prev = 0.0;
            let x: Vec<Complex<f64>> = (0..steps)
                .map(|_| {
                    let w = rng.gauss();
                    // AR(1) colouring: x_n = 0.9 x_{n−1} + w_n, normalised to unit power.
                    let v = if colored { 0.9 * prev + w } else { w };
                    prev = v;
                    Complex::new(if colored { v * (1.0f64 - 0.81).sqrt() } else { v }, 0.0)
                })
                .collect();
            let d: Vec<Complex<f64>> = (0..steps)
                .map(|n| Complex::new((0..m).filter(|&k| k <= n).map(|k| plant[k] * x[n - k].re).sum::<f64>() + 0.01 * rng.gauss(), 0.0))
                .collect();
            let mut lms = LmsFilter::new(m, 0.02, false).unwrap();
            let mut nlms = LmsFilter::new(m, 0.5, true).unwrap();
            let mut rls = RlsFilter::new(m, 0.999, 100.0).unwrap();
            for n in 0..steps {
                curves[0][n] += lms.adapt(x[n], d[n]).1.norm_sqr() / runs as f64;
                curves[1][n] += nlms.adapt(x[n], d[n]).1.norm_sqr() / runs as f64;
                curves[2][n] += rls.adapt(x[n], d[n]).1.norm_sqr() / runs as f64;
            }
        }
        let mut plot = Plot::new(title, "samples", "mean squared error (dB)");
        for (k, name) in ["LMS, μ = 0.02", "NLMS, μ = 0.5", "RLS, λ = 0.999"].into_iter().enumerate() {
            plot.series.push(Series::new(name, curves[k].iter().enumerate().step_by(5).map(|(i, &v)| (i as f64, 10.0 * v.log10())).collect(), Style::Line, k));
        }
        plot.series.push(Series::new("noise floor", vec![(0.0, -40.0), (steps as f64, -40.0)], Style::Dashed, 6));
        plot.y_range = Some((-45.0, 5.0));
        plot.legend = Legend::TopRight;
        write(CHAPTER, file, &plot.render());
    }
    // Cost per sample.
    let mut rows = Vec::new();
    for taps in [8usize, 32, 128] {
        let mut rng = Rng::new(7);
        let x: Vec<Complex<f64>> = (0..5000).map(|_| Complex::new(rng.gauss(), 0.0)).collect();
        let t_lms = time(|| {
            let mut f = LmsFilter::new(taps, 0.01, true).unwrap();
            f.adapt_block(&x, &x).unwrap();
        });
        let t_rls = time(|| {
            let mut f = RlsFilter::new(taps, 0.999, 100.0).unwrap();
            f.adapt_block(&x, &x).unwrap();
        });
        rows.push(vec![format!("{taps}"), format!("{:.0}", t_lms / 5000.0 * 1e9), format!("{:.0}", t_rls / 5000.0 * 1e9)]);
    }
    write(CHAPTER, "adaptive_cost.md", &md_table(&["taps", "NLMS (ns per sample)", "RLS (ns per sample)"], &rows));
}

// ------------------------------------------------------------------ second-order sections and zero-phase filtering

/// Pseudo-random noise of standard deviation about `sigma` (a sum of four uniforms), reproducible.
fn noise(n: usize, sigma: f64, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut u = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    (0..n).map(|_| sigma * (u() + u() + u() + u()) * 3f64.sqrt()).collect()
}

pub(crate) fn sos_figures() {
    // Direct form against sections: the step response of narrow Butterworth low-pass filters.
    let mut rows = Vec::new();
    let step = vec![1.0; 6000];
    let mut plot = Plot::new("Step response of a 12th-order Butterworth low-pass, cutoff 0.02 × Nyquist", "sample", "output");
    for (order, wn) in [(4usize, 0.05), (8, 0.05), (12, 0.05), (12, 0.02), (16, 0.05), (20, 0.1)] {
        let (b, a) = ba(butter(order, &[wn], Band::Lowpass, None, None).unwrap());
        let sos = sections(butter(order, &[wn], Band::Lowpass, Some(ButterOutputType::Zpk), None).unwrap());
        let (direct, _) = lfilter(&b, &a, &step, None).unwrap();
        let (cascade, _) = sosfilt(&sos, &step, None).unwrap();
        let describe = |y: &[f64]| {
            let last = y[y.len() - 1];
            let peak = y.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            if !last.is_finite() || peak > 1e6 {
                format!("diverges (output reaches {peak:.0e})")
            } else {
                format!("{:.1e}", (last - 1.0).abs())
            }
        };
        rows.push(vec![format!("{order}"), format!("{wn}"), describe(&direct), describe(&cascade)]);
        if (order, wn) == (12, 0.02) {
            plot.series.push(Series::new("direct form (b, a)", direct[..1500].iter().enumerate().map(|(i, &v)| (i as f64, v)).collect(), Style::Line, 1));
            plot.series.push(Series::new("second-order sections", cascade[..1500].iter().enumerate().map(|(i, &v)| (i as f64, v)).collect(), Style::Line, 0));
        }
    }
    plot.y_range = Some((-1.0, 3.0));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "sos_step.svg", &plot.render());
    write(CHAPTER, "sos_vs_ba.md", &md_table(&["Butterworth order", "cutoff (× Nyquist)", "direct form: final error of the unit step", "sections: final error"], &rows));

    // One pass against two: a noisy pulse train through a 4th-order low-pass.
    let n = 600;
    let clean: Vec<f64> = (0..n).map(|i| {
        let t = i as f64;
        [120.0, 300.0, 460.0].iter().map(|c| (-((t - c) / 18.0).powi(2)).exp()).sum::<f64>() + if (200..260).contains(&i) { 0.6 } else { 0.0 }
    }).collect();
    let x: Vec<f64> = clean.iter().zip(noise(n, 0.15, 7)).map(|(c, e)| c + e).collect();
    let sos = sections(butter(4, &[0.04], Band::Lowpass, Some(ButterOutputType::Zpk), None).unwrap());
    let (once, _) = sosfilt(&sos, &x, None).unwrap();
    let twice = sosfiltfilt(&sos, &x, Padding::Odd, None).unwrap();
    let mut plot = Plot::new("A noisy signal low-pass filtered once and forwards-backwards", "sample", "value");
    plot.series.push(Series::new("noisy input", x.iter().enumerate().map(|(i, &v)| (i as f64, v)).collect(), Style::Line, 7));
    plot.series.push(Series::new("one pass (sosfilt)", once.iter().enumerate().map(|(i, &v)| (i as f64, v)).collect(), Style::Line, 1));
    plot.series.push(Series::new("zero phase (sosfiltfilt)", twice.iter().enumerate().map(|(i, &v)| (i as f64, v)).collect(), Style::Line, 0));
    plot.series.push(Series::new("clean signal", clean.iter().enumerate().map(|(i, &v)| (i as f64, v)).collect(), Style::Dashed, 6));
    plot.legend = Legend::TopRight;
    plot.y_range = Some((-0.6, 1.8));
    write(CHAPTER, "zero_phase.svg", &plot.render());

    // The ends of the record: padding choices on a ramp plus a sine, where the true low-pass output is known.
    let n = 400;
    let base: Vec<f64> = (0..n).map(|i| 1.0 + 0.004 * i as f64 + 0.5 * (2.0 * PI * i as f64 / 150.0).sin()).collect();
    let (b, a) = ba(butter(3, &[0.05], Band::Lowpass, None, None).unwrap());
    let mut rows = Vec::new();
    for (name, padding) in [("odd (default)", Padding::Odd), ("even", Padding::Even), ("constant", Padding::Constant), ("none", Padding::None)] {
        let y = filtfilt(&b, &a, &base, padding, None).unwrap();
        let edge = |range: std::ops::Range<usize>| range.map(|i| (y[i] - base[i]).abs()).fold(0.0f64, f64::max);
        rows.push(vec![name.to_string(), format!("{:.1e}", edge(0..30)), format!("{:.1e}", edge(185..215)), format!("{:.1e}", edge(n - 30..n))]);
    }
    // Starting from rest instead of the steady state (no initial conditions, odd padding).
    let edge_len = 3 * b.len().max(a.len());
    let mut ext: Vec<f64> = (1..=edge_len).rev().map(|k| 2.0 * base[0] - base[k]).collect();
    ext.extend_from_slice(&base);
    ext.extend((1..=edge_len).map(|k| 2.0 * base[n - 1] - base[n - 1 - k]));
    let (f, _) = lfilter(&b, &a, &ext, None).unwrap();
    let r: Vec<f64> = f.iter().rev().copied().collect();
    let (g, _) = lfilter(&b, &a, &r, None).unwrap();
    let y: Vec<f64> = g.iter().rev().copied().collect::<Vec<f64>>()[edge_len..edge_len + n].to_vec();
    let edge = |range: std::ops::Range<usize>| range.map(|i| (y[i] - base[i]).abs()).fold(0.0f64, f64::max);
    rows.push(vec!["odd, but each pass starting from rest".to_string(), format!("{:.1e}", edge(0..30)), format!("{:.1e}", edge(185..215)), format!("{:.1e}", edge(n - 30..n))]);
    write(CHAPTER, "filtfilt_edges.md", &md_table(&["padding", "largest error, first 30 samples", "in the middle", "last 30 samples"], &rows));
}
