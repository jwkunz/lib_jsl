//! Figures and tables for the DSP chapter on spectral estimation, windows, resampling and multirate
//! processing.

use crate::linear_algebra::time;
use crate::svg::{Heatmap, Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_dsp::continuous::spectral::fourier_integrals::{FourierDirection, fourier_integral};
use lib_jsl_dsp::discrete::resampling::cic::{CicDecimator, cic_magnitude_response};
use lib_jsl_dsp::discrete::resampling::half_band::{HalfBandDecimator, halfband_lowpass};
use lib_jsl_dsp::discrete::resampling::polyphase_arbitrary_resampling::PolyphaseArbitraryResampler;
use lib_jsl_dsp::discrete::spectral::critically_sampled_polyphase_filter_bank::{CriticallySampledPolyphaseAnalysisFilterBank, CriticallySampledPolyphaseSynthesisFilterBank};
use lib_jsl_dsp::discrete::spectral::goertzel::goertzel;
use lib_jsl_dsp::discrete::spectral::hilbert::instantaneous_attributes;
use lib_jsl_dsp::discrete::spectral::linear_predictive_coding::linear_predictive_coding;
use lib_jsl_dsp::discrete::spectral::maximum_entropy::{burg, maximum_entropy_spectrum};
use lib_jsl_dsp::discrete::spectral::music::music_real;
use lib_jsl_dsp::discrete::spectral::oversampled_polyphase_filter_bank::{OversampledPolyphaseAnalysisFilterBank, OversampledPolyphaseSynthesisFilterBank};
use lib_jsl_dsp::discrete::spectral::stft::{istft, spectrogram, stft};
use lib_jsl_dsp::discrete::spectral::wavelet_transform::{ContinuousWavelet, Wavelet, continuous_wavelet_transform, wavedec, waverec};
use lib_jsl_ffts::RealFft;
use lib_jsl_random::uniform_generator::UniformGenerator;
use lib_jsl_random::xoshiro256plusplus::Xoshiro256PlusPlus;
use lib_jsl_dsp::discrete::resampling::polyphase_integer_resampling::PolyphaseIntegerResampler;
use lib_jsl_dsp::discrete::spectral::chirpz::chirpz;
use lib_jsl_dsp::discrete::spectral::hilbert::analytic_signal;
use lib_jsl_dsp::discrete::spectral::linear_predictive_coding::levinson_durbin;
use lib_jsl_dsp::discrete::spectral::non_uniformly_sampled::lomb_scargle;
use lib_jsl_dsp::discrete::spectral::welch::{Detrend, SpectralScaling, periodogram, welch};
use lib_jsl_dsp::discrete::stream_operator::StreamOperator;
use lib_jsl_dsp::discrete::windows::{self, WindowType, generate_window};
use num_complex::Complex;
use std::f64::consts::PI;

const CHAPTER: &str = "spectral";

pub fn generate() {
    reference_table();
    window_figures();
    psd_figures();
    chosen_frequency_figures();
    hilbert_figures();
    parametric_figures();
    wavelet_figures();
    resampling_figures();
    filter_bank_figures();
}

pub(crate) fn md_table(header: &[&str], rows: &[Vec<String>]) -> String {
    let mut s = format!("| {} |\n|{}\n", header.join(" | "), header.iter().enumerate().map(|(i, _)| if i == 0 { "---|" } else { "---:|" }).collect::<String>());
    for r in rows {
        s += &format!("| {} |\n", r.join(" | "));
    }
    s
}

fn window_named(name: &str, m: usize, sym: bool) -> Option<Vec<f64>> {
    let t = match name {
        "boxcar" => WindowType::Boxcar,
        "triang" => WindowType::Triang,
        "parzen" => WindowType::Parzen,
        "bohman" => WindowType::Bohman,
        "blackman" => WindowType::Blackman,
        "nuttall" => WindowType::Nuttall,
        "blackmanharris" => WindowType::BlackmanHarris,
        "flattop" => WindowType::Flattop,
        "bartlett" => WindowType::Bartlett,
        "barthann" => WindowType::Barthann,
        "hamming" => WindowType::Hamming,
        "hann" => WindowType::Hann,
        "kaiser 8" => WindowType::Kaiser { beta: 8.0 },
        "gaussian 2.5" => WindowType::Gaussian { std: 2.5 },
        "general_gaussian 1.5 3" => WindowType::GeneralGaussian { p: 1.5, sig: 3.0 },
        "chebwin 80" => WindowType::Chebwin { at: 80.0 },
        "cosine" => WindowType::Cosine,
        "exponential tau 3" => WindowType::Exponential { center: None, tau: 3.0 },
        "tukey 0.4" => WindowType::Tukey { alpha: 0.4 },
        "taylor 4 30" => WindowType::Taylor { nbar: 4, sll: 30.0, norm: true },
        "dpss 3" => WindowType::Dpss { nw: 3.0 },
        "lanczos" => WindowType::Lanczos,
        "general_hamming 0.6" => return Some(windows::hamming::general_hamming(m, 0.6, sym)),
        _ => return None,
    };
    Some(generate_window(t, m, sym))
}

/// The library's values for each named reference case.
fn library_case(name: &str, data: &std::collections::HashMap<String, Vec<Vec<f64>>>) -> Option<Vec<Vec<f64>>> {
    let x = &data["signal"][0];
    let cplx = |v: Vec<Complex<f64>>| vec![v.iter().map(|c| c.re).collect(), v.iter().map(|c| c.im).collect()];
    if let Some(rest) = name.strip_prefix("window ") {
        let parts: Vec<&str> = rest.rsplitn(3, ' ').collect();
        if rest.starts_with("kaiser_bessel_derived") {
            return Some(vec![generate_window(WindowType::KaiserBesselDerived { beta: 8.0 }, 16, true)]);
        }
        let (tag, m, wname) = (parts[0], parts[1].parse::<usize>().ok()?, parts[2]);
        return window_named(wname, m, tag == "sym").map(|w| vec![w]);
    }
    Some(match name {
        "signal" | "lomb t" | "lomb y" => return None,
        "periodogram hann density" => vec![periodogram(x, 10.0, &windows::hann::hann(256, false), Detrend::Constant, SpectralScaling::Density).ok()?.power],
        "welch hann 64/32 linear spectrum" => vec![welch(x, 10.0, &windows::hann::hann(64, false), 32, Detrend::Linear, SpectralScaling::Spectrum).ok()?.power],
        "hilbert 64" => cplx(analytic_signal(&x[..64]).ok()?),
        "czt 20 points" => cplx(chirpz(&x[..64], Some(20), Some(Complex::from_polar(1.0, -2.0 * PI * 0.01)), Some(Complex::from_polar(1.0, 2.0 * PI * 0.05))).ok()?.to_vec()),
        "lombscargle unnormalized" => {
            let freqs: Vec<f64> = (0..30).map(|i| 0.1 + 5.9 * i as f64 / 29.0).collect();
            vec![lomb_scargle(&data["lomb t"][0], &data["lomb y"][0], &freqs, false).ok()?]
        }
        "upfirdn up 3" => {
            // The resampler multiplies its taps by the upsampling factor; SciPy's upfirdn does not.
            let h = [0.25 / 3.0, 0.5 / 3.0, 1.0 / 3.0, 0.5 / 3.0, 0.25 / 3.0];
            let mut r = PolyphaseIntegerResampler::new(3, 1, Some(&h)).ok()?;
            let mut y = r.process(&x[..20]).ok()?.unwrap_or_default();
            y.extend(r.flush().ok()?.unwrap_or_default());
            vec![y]
        }
        "upfirdn up 3 down 2" => {
            let h = lib_jsl_dsp::discrete::filters::fir::firwin::firwin(31, &[0.2], None, WindowType::Hamming, lib_jsl_dsp::discrete::filters::fir::firwin::FirwinPassZero::True, true, 2.0).ok()?;
            let mut r = PolyphaseIntegerResampler::new(3, 2, Some(&h)).ok()?;
            let mut y = r.process(&x[..40]).ok()?.unwrap_or_default();
            y.extend(r.flush().ok()?.unwrap_or_default());
            vec![y]
        }
        "stft hann 64/16" => {
            let z = lib_jsl_dsp::discrete::spectral::stft::stft(x, &windows::hann::hann(64, false), 16).ok()?;
            cplx(z.iter().copied().collect())
        }
        "spectrogram hann 64/16 density" => vec![lib_jsl_dsp::discrete::spectral::stft::spectrogram(x, 10.0, &windows::hann::hann(64, false), 16, SpectralScaling::Density).ok()?.power.iter().copied().collect()],
        "square period 8" => vec![lib_jsl_dsp::discrete::sources::square(8, 40).ok()?],
        "levinson order 8" => {
            let r: Vec<f64> = (0..9).map(|k| x[..256 - k].iter().zip(&x[k..]).map(|(a, b)| a * b).sum::<f64>() / 256.0).collect();
            // SciPy solves R a = r[1..]; Levinson returns A(z) = 1 + Σ a_k z^-k with the opposite sign.
            vec![levinson_durbin(&r, 8).ok()?.coefficients[1..].iter().map(|v| -v).collect()]
        }
        _ => return None,
    })
}

fn reference_table() {
    let reference = include_str!("../data/spectral_reference.txt");
    let data: std::collections::HashMap<String, Vec<Vec<f64>>> = reference
        .lines()
        .map(|line| {
            let (name, values) = line.split_once('|').unwrap();
            (name.to_string(), values.split(';').map(|l| l.split(',').map(|v| v.parse().unwrap()).collect()).collect())
        })
        .collect();
    let mut window_worst: Vec<(String, f64)> = Vec::new();
    let mut rows = Vec::new();
    for line in reference.lines() {
        let name = line.split_once('|').unwrap().0;
        let want = &data[name];
        let Some(got) = library_case(name, &data) else {
            if !matches!(name, "signal" | "lomb t" | "lomb y") {
                rows.push(vec![name.to_string(), "missing".to_string()]);
            }
            continue;
        };
        let err = if got.len() != want.len() || got.iter().zip(want).any(|(g, w)| g.len() != w.len()) {
            f64::INFINITY
        } else {
            let scale = want.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
            got.iter().flatten().zip(want.iter().flatten()).map(|(g, w)| (g - w).abs()).fold(0.0, f64::max) / scale
        };
        if let Some(rest) = name.strip_prefix("window ") {
            let wname = rest.rsplitn(3, ' ').last().unwrap().to_string();
            match window_worst.iter_mut().find(|(n, _)| *n == wname) {
                Some(entry) => entry.1 = entry.1.max(err),
                None => window_worst.push((wname, err)),
            }
        } else {
            rows.push(vec![name.to_string(), if err.is_finite() { format!("{err:.1e}") } else { "length differs".to_string() }]);
        }
    }
    let mut wrows: Vec<Vec<String>> = window_worst.into_iter().map(|(n, e)| vec![n, if e.is_finite() { format!("{e:.1e}") } else { "length differs".to_string() }]).collect();
    wrows.sort();
    write(CHAPTER, "window_reference.md", &md_table(&["window (lengths 16 and 17, symmetric and periodic)", "largest difference from SciPy"], &wrows));
    write(CHAPTER, "reference.md", &md_table(&["SciPy function and case", "largest difference ÷ largest value"], &rows));
}

// ------------------------------------------------------------------ helpers

struct Rng(Xoshiro256PlusPlus);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(Xoshiro256PlusPlus::from_seed(seed))
    }
    fn gauss(&mut self) -> f64 {
        let (u, v) = (1.0 - self.0.next_f64(), self.0.next_f64());
        (-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos()
    }
    fn uniform(&mut self) -> f64 {
        self.0.next_f64()
    }
}

/// |W(f)|² of a window, zero-padded to `n` points: `(bins of the window length, dB)` up to Nyquist.
fn window_spectrum(w: &[f64], n: usize) -> Vec<(f64, f64)> {
    let mut padded = w.to_vec();
    padded.resize(n, 0.0);
    let spectrum = RealFft::new(n).unwrap().forward(&padded).unwrap();
    let peak = spectrum[0].norm();
    spectrum.iter().enumerate().map(|(k, v)| (k as f64 * w.len() as f64 / n as f64, 20.0 * (v.norm() / peak).max(1e-12).log10())).collect()
}

fn db10(v: f64) -> f64 {
    10.0 * v.max(1e-300).log10()
}

// ------------------------------------------------------------------ windows

/// A window type, made on demand (`WindowType` is not `Copy`).
type MakeWindow = fn() -> WindowType;

fn window_figures() {
    let m = 64;
    let list: Vec<(&str, MakeWindow)> = vec![
        ("rectangular", || WindowType::Boxcar),
        ("Hann", || WindowType::Hann),
        ("Hamming", || WindowType::Hamming),
        ("Blackman", || WindowType::Blackman),
        ("Blackman-Harris", || WindowType::BlackmanHarris),
        ("flat top", || WindowType::Flattop),
        ("Kaiser β = 8", || WindowType::Kaiser { beta: 8.0 }),
        ("Dolph-Chebyshev 80 dB", || WindowType::Chebwin { at: 80.0 }),
        ("DPSS NW = 3", || WindowType::Dpss { nw: 3.0 }),
        ("Tukey α = 0.5", || WindowType::Tukey { alpha: 0.5 }),
        ("Gaussian σ = 8", || WindowType::Gaussian { std: 8.0 }),
        ("Taylor n̄ = 4, 30 dB", || WindowType::Taylor { nbar: 4, sll: 30.0, norm: true }),
    ];
    let mut shapes = Plot::new("Window shapes, 64 samples", "sample", "w[n]");
    for (k, (name, t)) in list.iter().take(7).enumerate() {
        shapes.series.push(Series::new(name, generate_window(t(), m, false).into_iter().enumerate().map(|(i, v)| (i as f64, v)).collect(), Style::Line, k));
    }
    shapes.legend = Legend::BottomRight;
    shapes.y_range = Some((-0.1, 1.1));
    write(CHAPTER, "window_shapes.svg", &shapes.render());
    for (file, title, chosen) in [
        ("window_spectra_cosine.svg", "Transforms of the cosine-sum windows", vec![0usize, 1, 3, 4]),
        ("window_spectra_param.svg", "Transforms of the parametric windows", vec![6usize, 7, 8, 10]),
    ] {
        let mut plot = Plot::new(title, "frequency (DFT bins of the 64-sample window)", "gain (dB)");
        for (k, &i) in chosen.iter().enumerate() {
            let (name, t) = list[i];
            plot.series.push(Series::new(name, window_spectrum(&generate_window(t(), m, false), 16384).into_iter().filter(|p| p.0 <= 16.0).collect(), Style::Line, k));
        }
        plot.y_range = Some((-140.0, 5.0));
        plot.legend = Legend::TopRight;
        write(CHAPTER, file, &plot.render());
    }
    // Figures of merit.
    let mut rows = Vec::new();
    for (name, t) in &list {
        let w = generate_window(t(), m, false);
        let sum: f64 = w.iter().sum();
        let sum2: f64 = w.iter().map(|v| v * v).sum();
        let enbw = m as f64 * sum2 / (sum * sum);
        let spec = window_spectrum(&w, 65536);
        let half = spec.iter().find(|p| p.1 < -3.0103).map_or(f64::NAN, |p| 2.0 * p.0);
        // The first null (or local minimum), then the highest level beyond it.
        let first_min = (1..spec.len() - 1).find(|&i| spec[i].1 < spec[i - 1].1 && spec[i].1 <= spec[i + 1].1).unwrap_or(1);
        let sidelobe = spec[first_min..].iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        let scallop = -spec.iter().min_by(|a, b| (a.0 - 0.5).abs().total_cmp(&(b.0 - 0.5).abs())).unwrap().1;
        rows.push(vec![name.to_string(), format!("{:.3}", sum / m as f64), format!("{enbw:.2}"), format!("{half:.2}"), format!("{sidelobe:.1}"), format!("{scallop:.2}"), format!("{:.2}", 2.0 * spec[first_min].0)]);
    }
    write(CHAPTER, "window_merits.md", &md_table(&["window (64 samples, periodic)", "coherent gain", "noise bandwidth (bins)", "3 dB width (bins)", "highest side lobe (dB)", "scalloping loss (dB)", "main lobe width to first minimum (bins)"], &rows));
}

// ------------------------------------------------------------------ power spectra

fn psd_figures() {
    let fs = 1000.0;
    let n = 4096;
    let mut rng = Rng::new(3);
    let x: Vec<f64> = (0..n).map(|i| {
        let t = i as f64 / fs;
        (2.0 * PI * 100.0 * t).sin() + 3e-4 * (2.0 * PI * 157.0 * t).sin() + 3e-5 * rng.gauss()
    }).collect();
    let mut plot = Plot::new("A tone at 100 Hz, one 70 dB weaker at 157 Hz, and white noise", "frequency (Hz)", "power spectral density (dB/Hz)");
    let rect = periodogram(&x, fs, &vec![1.0; n], Detrend::Constant, SpectralScaling::Density).unwrap();
    let hann = periodogram(&x, fs, &windows::hann::hann(n, false), Detrend::Constant, SpectralScaling::Density).unwrap();
    let w = welch(&x, fs, &windows::hann::hann(256, false), 128, Detrend::Constant, SpectralScaling::Density).unwrap();
    for (k, (name, p)) in [("periodogram, rectangular window", &rect), ("periodogram, Hann window", &hann), ("Welch, Hann, 256-sample segments", &w)].into_iter().enumerate() {
        plot.series.push(Series::new(name, p.frequencies.iter().zip(&p.power).map(|(&f, &v)| (f, db10(v))).collect(), Style::Line, k));
    }
    plot.y_range = Some((-150.0, 10.0));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "psd_estimates.svg", &plot.render());
    // Variance of the estimate for white noise.
    let mut rows = Vec::new();
    let noise: Vec<f64> = (0..16384).map(|_| rng.gauss()).collect();
    for seg in [16384usize, 4096, 1024, 256, 64] {
        let p = welch(&noise, 1.0, &windows::hann::hann(seg, false), seg / 2, Detrend::None, SpectralScaling::Density).unwrap();
        let v: Vec<f64> = p.power[1..p.power.len() - 1].to_vec();
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let sd = (v.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        rows.push(vec![format!("{seg}"), format!("{}", p.segments), format!("{:.4}", 1.0 / seg as f64), format!("{mean:.3}"), format!("{:.3}", sd / mean)]);
    }
    write(CHAPTER, "welch_variance.md", &md_table(&["segment length (Hann, 50% overlap)", "segments averaged", "bin width (× fs)", "mean density (true 2.000)", "standard deviation ÷ mean"], &rows));
    // Spectrogram of a chirp and a tone.
    let fs = 8000.0;
    let len = 16000;
    let sig: Vec<f64> = (0..len).map(|i| {
        let t = i as f64 / fs;
        (2.0 * PI * (200.0 * t + 0.5 * 1500.0 * t * t)).sin() + 0.5 * (2.0 * PI * 3000.0 * t).sin() * if t > 1.0 { 1.0 } else { 0.0 } + 0.05 * rng.gauss()
    }).collect();
    let sg = spectrogram(&sig, fs, &windows::hann::hann(256, false), 64, SpectralScaling::Density).unwrap();
    let (frames, bins) = sg.power.dim();
    let values: Vec<Vec<f64>> = (0..bins).map(|k| (0..frames).map(|t| db10(sg.power[[t, k]])).collect()).collect();
    let mut plot = Plot::new("Spectrogram: a chirp from 200 Hz rising 1500 Hz/s, and a 3 kHz tone after 1 s", "time (s)", "frequency (Hz)");
    plot.heatmap = Some(Heatmap { x: (sg.times[0] - 0.004, sg.times[frames - 1] + 0.004), y: (0.0, 4000.0 + 8000.0 / 256.0), values, range: (-90.0, -25.0) });
    plot.x_range = Some((0.0, 2.0));
    plot.y_range = Some((0.0, 4000.0));
    write(CHAPTER, "spectrogram.svg", &plot.render());
    // ISTFT round trip.
    let s = stft(&sig, &windows::hann::hann(256, false), 64).unwrap();
    let back = istft(&s, &windows::hann::hann(256, false), 64).unwrap();
    let err = back[256..back.len() - 256].iter().zip(&sig[256..]).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    write(CHAPTER, "istft.md", &format!("Largest reconstruction error of `istft(stft(x))` away from the ends: {err:.1e}.\n"));
    // Lomb-Scargle on uneven samples.
    let mut rng = Rng::new(8);
    let mut t: Vec<f64> = (0..120).map(|_| 30.0 * rng.uniform()).collect();
    t.sort_by(f64::total_cmp);
    let y: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * 0.73 * ti).sin() + 0.5 * (2.0 * PI * 1.9 * ti + 1.0).sin() + 0.4 * rng.gauss()).collect();
    let freqs: Vec<f64> = (1..=1200).map(|i| 0.0025 * i as f64).collect();
    let ls = lomb_scargle(&t, &y, &freqs.iter().map(|f| 2.0 * PI * f).collect::<Vec<_>>(), true).unwrap();
    let mut plot = Plot::new("Lomb-Scargle periodogram of 120 unevenly spaced samples (tones at 0.73 and 1.9)", "frequency (cycles per unit time)", "normalised power");
    plot.series.push(Series::new("", freqs.iter().zip(&ls).map(|(&f, &p)| (f, p)).collect(), Style::Line, 0));
    write(CHAPTER, "lomb_scargle.svg", &plot.render());
}

// ------------------------------------------------------------------ chosen frequencies

fn chosen_frequency_figures() {
    // Zooming with the chirp-z transform on two close tones.
    let n = 256;
    let x: Vec<f64> = (0..n).map(|i| (2.0 * PI * 0.200 * i as f64).sin() + (2.0 * PI * 0.207 * i as f64).sin()).collect();
    let fft = RealFft::new(n).unwrap().forward(&x).unwrap();
    let (f0, f1, points) = (0.18, 0.23, 400);
    let w = Complex::from_polar(1.0, -2.0 * PI * (f1 - f0) / points as f64);
    let a = Complex::from_polar(1.0, 2.0 * PI * f0);
    let zoom = chirpz(&x, Some(points), Some(w), Some(a)).unwrap();
    let mut plot = Plot::new("Two tones 0.007 cycles/sample apart, 256 samples", "frequency (cycles per sample)", "|X| (dB)");
    plot.series.push(Series::new("FFT bins", fft.iter().enumerate().map(|(k, v)| (k as f64 / n as f64, 20.0 * v.norm().log10())).filter(|p| p.0 >= f0 && p.0 <= f1).collect(), Style::Markers, 1));
    plot.series.push(Series::new("chirp-z zoom, 400 points", zoom.iter().enumerate().map(|(k, v)| (f0 + (f1 - f0) * k as f64 / points as f64, 20.0 * v.norm().log10())).collect(), Style::Line, 0));
    plot.legend = Legend::BottomRight;
    plot.y_range = Some((0.0, 50.0));
    write(CHAPTER, "chirpz_zoom.svg", &plot.render());
    // Goertzel against an FFT.
    let mut rows = Vec::new();
    for n in [256usize, 4096, 65536] {
        let sig: Vec<f64> = (0..n).map(|i| (0.3 * i as f64).sin()).collect();
        let tg = time(|| {
            goertzel(&sig, 12.0).unwrap();
        });
        let mut rf = RealFft::new(n).unwrap();
        let tf = time(|| {
            rf.forward(&sig).unwrap();
        });
        rows.push(vec![format!("{n}"), format!("{:.1}", tg * 1e6), format!("{:.1}", tf * 1e6), format!("{:.0}", tf / tg)]);
    }
    write(CHAPTER, "goertzel_time.md", &md_table(&["samples", "one Goertzel bin (µs)", "real FFT, all bins (µs)", "bins before the FFT is cheaper"], &rows));
    // Fourier integral of a Gaussian.
    let mut rows = Vec::new();
    for m in [16usize, 32, 64, 128] {
        let r = fourier_integral(|x| Complex::new((-0.5 * x * x).exp(), 0.0), -8.0, 8.0, 0.0, 4.0, m, FourierDirection::Forward).unwrap();
        let err = r.omega.iter().zip(r.values.iter()).map(|(&w, v)| (v - Complex::new((2.0 * PI).sqrt() * (-0.5 * w * w).exp(), 0.0)).norm()).fold(0.0, f64::max);
        rows.push(vec![format!("{m}"), format!("{err:.1e}")]);
    }
    write(CHAPTER, "fourier_integral.md", &md_table(&["samples on [−8, 8] (and frequencies on [0, 4])", "largest error against √(2π) e^(−ω²/2)"], &rows));
}

// ------------------------------------------------------------------ Hilbert transform

fn hilbert_figures() {
    let fs = 1000.0;
    let n = 1000;
    let am: Vec<f64> = (0..n).map(|i| {
        let t = i as f64 / fs;
        (1.0 + 0.6 * (2.0 * PI * 3.0 * t).cos()) * (2.0 * PI * 60.0 * t).cos()
    }).collect();
    let attr = instantaneous_attributes(&am, fs).unwrap();
    let mut plot = Plot::new("An AM signal and its envelope from the analytic signal", "time (s)", "amplitude");
    plot.series.push(Series::new("signal", am.iter().enumerate().map(|(i, &v)| (i as f64 / fs, v)).collect(), Style::Line, 7));
    plot.series.push(Series::new("envelope |x + iH{x}|", attr.amplitude.iter().enumerate().map(|(i, &v)| (i as f64 / fs, v)).collect(), Style::Line, 0));
    plot.series.push(Series::new("true envelope", (0..n).map(|i| (i as f64 / fs, 1.0 + 0.6 * (2.0 * PI * 3.0 * i as f64 / fs).cos())).collect(), Style::Dashed, 6));
    plot.legend = Legend::BottomRight;
    write(CHAPTER, "hilbert_envelope.svg", &plot.render());
    let chirp: Vec<f64> = (0..n).map(|i| {
        let t = i as f64 / fs;
        (2.0 * PI * (50.0 * t + 100.0 * t * t)).sin()
    }).collect();
    let attr = instantaneous_attributes(&chirp, fs).unwrap();
    let mut plot = Plot::new("Instantaneous frequency of a chirp 50 + 200t Hz", "time (s)", "frequency (Hz)");
    plot.series.push(Series::new("from the analytic signal", attr.frequency.iter().enumerate().map(|(i, &v)| ((i as f64 + 0.5) / fs, v)).collect(), Style::Line, 0));
    plot.series.push(Series::new("true", vec![(0.0, 50.0), (1.0, 250.0)], Style::Dashed, 6));
    plot.legend = Legend::TopLeft;
    plot.y_range = Some((0.0, 300.0));
    write(CHAPTER, "hilbert_frequency.svg", &plot.render());
}

// ------------------------------------------------------------------ parametric methods

fn parametric_figures() {
    // Two close tones in a short record.
    let fs = 1.0;
    let n = 64;
    let mut rng = Rng::new(21);
    let x: Vec<f64> = (0..n).map(|i| (2.0 * PI * 0.20 * i as f64).sin() + (2.0 * PI * 0.215 * i as f64 + 0.7).sin() + 0.1 * rng.gauss()).collect();
    let mut plot = Plot::new("Two tones at 0.200 and 0.215 cycles/sample, 64 samples", "frequency (cycles per sample)", "normalised spectrum (dB)");
    let p = periodogram(&x, fs, &windows::hann::hann(n, false), Detrend::Constant, SpectralScaling::Density).unwrap();
    let mut padded = x.iter().zip(windows::hann::hann(n, false)).map(|(a, b)| a * b).collect::<Vec<_>>();
    padded.resize(4096, 0.0);
    let zp = RealFft::new(4096).unwrap().forward(&padded).unwrap();
    let peak = zp.iter().map(|v| v.norm_sqr()).fold(0.0, f64::max);
    plot.series.push(Series::new("periodogram, Hann (zero-padded)", zp.iter().enumerate().map(|(k, v)| (k as f64 / 4096.0, db10(v.norm_sqr() / peak))).collect(), Style::Line, 7));
    let _ = p;
    let (f, me) = maximum_entropy_spectrum(&x, 16, 2049, fs).unwrap();
    let mp = me.iter().fold(0.0, |a: f64, &b| a.max(b));
    plot.series.push(Series::new("Burg AR(16) (maximum entropy)", f.iter().zip(&me).map(|(&a, &b)| (a, db10(b / mp))).collect(), Style::Line, 0));
    let mu = music_real(&x, 2, 24, 2049, fs).unwrap();
    let mm = mu.pseudospectrum.iter().fold(0.0, |a: f64, &b| a.max(b));
    plot.series.push(Series::new("MUSIC, 2 sinusoids, 24×24 covariance", mu.frequencies.iter().zip(&mu.pseudospectrum).map(|(&a, &b)| (a, db10(b / mm))).collect(), Style::Line, 1));
    plot.series.push(Series::new("true frequencies", vec![(0.2, 5.0), (0.2, -60.0)], Style::Dashed, 6));
    plot.series.push(Series::new("", vec![(0.215, 5.0), (0.215, -60.0)], Style::Dashed, 6));
    plot.x_range = Some((0.1, 0.3));
    plot.y_range = Some((-60.0, 5.0));
    plot.legend = Legend::BottomLeft;
    write(CHAPTER, "parametric_resolution.svg", &plot.render());
    let peaks = mu.peak_frequencies(2);
    write(CHAPTER, "music_peaks.md", &format!("MUSIC's two peaks: {:.4} and {:.4} cycles per sample (true 0.2000 and 0.2150).\n", peaks[0], peaks[1]));
    // AR coefficient recovery: Levinson (autocorrelation method) against Burg.
    let a_true = [1.0, -2.7607, 3.8106, -2.6535, 0.9238];
    let mut rows = Vec::new();
    for len in [64usize, 256, 1024] {
        let (mut err_l, mut err_b) = (0.0, 0.0);
        let trials = 50;
        for trial in 0..trials {
            let mut rng = Rng::new(1000 + trial);
            let mut y = vec![0.0; len + 200];
            for i in 4..y.len() {
                y[i] = rng.gauss() - (1..5).map(|k| a_true[k] * y[i - k]).sum::<f64>();
            }
            let y = &y[200..];
            let l = linear_predictive_coding(y, 4).unwrap();
            let b = burg(y, 4).unwrap();
            err_l += (1..5).map(|k| (l.coefficients[k] - a_true[k]).powi(2)).sum::<f64>().sqrt() / trials as f64;
            err_b += (1..5).map(|k| (b.coefficients[k] - a_true[k]).powi(2)).sum::<f64>().sqrt() / trials as f64;
        }
        rows.push(vec![format!("{len}"), format!("{err_l:.3}"), format!("{err_b:.3}")]);
    }
    write(CHAPTER, "ar_estimation.md", &md_table(&["samples of a sharply resonant AR(4) process", "Levinson-Durbin (autocorrelation method) error", "Burg error"], &rows));
}

// ------------------------------------------------------------------ wavelets

fn wavelet_figures() {
    // Denoising a piecewise signal by soft-thresholding the detail coefficients.
    let n = 1024;
    let clean: Vec<f64> = (0..n).map(|i| {
        let t = i as f64 / n as f64;
        if t < 0.3 { 1.0 } else if t < 0.5 { -0.5 + 3.0 * (t - 0.3) } else if t < 0.7 { (2.0 * PI * 8.0 * t).sin() } else { 0.0 }
    }).collect();
    let mut rng = Rng::new(5);
    let sigma = 0.2;
    let noisy: Vec<f64> = clean.iter().map(|v| v + sigma * rng.gauss()).collect();
    let mut rows = Vec::new();
    let mut best: Vec<f64> = Vec::new();
    for (name, wavelet) in [("Haar", Wavelet::Haar), ("Daubechies 4", Wavelet::Daubechies(4)), ("Daubechies 8", Wavelet::Daubechies(8))] {
        let mut d = wavedec(&noisy, wavelet, 5).unwrap();
        let threshold = sigma * (2.0 * (n as f64).ln()).sqrt();
        for level in d.details.iter_mut() {
            level.iter_mut().for_each(|c| *c = c.signum() * (c.abs() - threshold).max(0.0));
        }
        let den = waverec(&d, wavelet).unwrap();
        let rms = |a: &[f64]| (a.iter().zip(&clean).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / n as f64).sqrt();
        rows.push(vec![name.to_string(), format!("{:.3}", rms(&noisy)), format!("{:.3}", rms(&den[..n]))]);
        if name == "Daubechies 4" {
            best = den;
        }
    }
    write(CHAPTER, "wavelet_denoise.md", &md_table(&["wavelet (5 levels, universal soft threshold)", "RMS error, noisy", "RMS error, denoised"], &rows));
    let mut plot = Plot::new("Wavelet denoising (Daubechies 4, soft threshold)", "sample", "value");
    plot.series.push(Series::new("noisy", noisy.iter().enumerate().map(|(i, &v)| (i as f64, v)).collect(), Style::Line, 7));
    plot.series.push(Series::new("denoised", best.iter().take(n).enumerate().map(|(i, &v)| (i as f64, v)).collect(), Style::Line, 0));
    plot.series.push(Series::new("clean", clean.iter().enumerate().map(|(i, &v)| (i as f64, v)).collect(), Style::Dashed, 6));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "wavelet_denoise.svg", &plot.render());
    // Perfect reconstruction.
    let mut rows = Vec::new();
    for (name, wavelet) in [("Haar", Wavelet::Haar), ("Daubechies 2", Wavelet::Daubechies(2)), ("Daubechies 6", Wavelet::Daubechies(6)), ("Daubechies 10", Wavelet::Daubechies(10))] {
        let d = wavedec(&noisy, wavelet, 4).unwrap();
        let back = waverec(&d, wavelet).unwrap();
        let err = back.iter().zip(&noisy).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let h = wavelet.scaling_filter().unwrap();
        let orth = (0..h.len() / 2).map(|s| (h.iter().zip(h.iter().skip(2 * s)).map(|(a, b)| a * b).sum::<f64>() - if s == 0 { 1.0 } else { 0.0 }).abs()).fold(0.0, f64::max);
        rows.push(vec![name.to_string(), format!("{}", h.len()), format!("{orth:.1e}"), format!("{err:.1e}")]);
    }
    write(CHAPTER, "wavelet_checks.md", &md_table(&["wavelet", "filter length", "largest deviation from orthonormality Σ h[n]h[n+2s] = δ_s", "reconstruction error after 4 levels"], &rows));
    // Scalogram of a chirp.
    let len = 1024;
    let sig: Vec<f64> = (0..len).map(|i| {
        let t = i as f64;
        (2.0 * PI * (0.01 * t + 0.5 * 0.0003 * t * t)).sin() + if (400..440).contains(&i) { 2.0 * (2.0 * PI * 0.35 * t).sin() } else { 0.0 }
    }).collect();
    let freqs: Vec<f64> = (0..80).map(|k| 0.005 * (0.45f64 / 0.005).powf(k as f64 / 79.0)).collect();
    let wavelet = ContinuousWavelet::Morlet { omega0: 6.0 };
    let scales: Vec<f64> = freqs.iter().map(|&f| wavelet.scale_for_frequency(f)).collect();
    let cwt = continuous_wavelet_transform(&sig, &scales, wavelet).unwrap();
    let values: Vec<Vec<f64>> = (0..freqs.len()).map(|k| (0..len).step_by(4).map(|t| cwt[[k, t]].norm()).collect()).collect();
    let mut plot = Plot::new("Morlet scalogram: a chirp and a 40-sample burst at 0.35 cycles/sample", "sample", "log₁₀ frequency (cycles per sample)");
    plot.heatmap = Some(Heatmap { x: (0.0, len as f64), y: (freqs[0].log10(), freqs[79].log10() + (freqs[79].log10() - freqs[0].log10()) / 79.0), values, range: (0.0, 2.5) });
    plot.x_range = Some((0.0, len as f64));
    plot.y_range = Some((freqs[0].log10(), freqs[79].log10()));
    write(CHAPTER, "scalogram.svg", &plot.render());
}

// ------------------------------------------------------------------ resampling

fn resampling_figures() {
    // Interpolation by 4 of a short burst.
    let x: Vec<f64> = (0..24).map(|i| (2.0 * PI * 0.11 * i as f64).sin() * (-0.01 * (i as f64 - 12.0).powi(2)).exp()).collect();
    let mut up = PolyphaseIntegerResampler::new(4, 1, None).unwrap();
    let mut y = up.process(&x).unwrap().unwrap();
    y.extend(up.flush().unwrap().unwrap_or_default());
    let mut probe = PolyphaseIntegerResampler::<f64>::new(4, 1, None).unwrap();
    let mut impulse = vec![0.0; 200];
    impulse[0] = 1.0;
    let mut hresp = probe.process(&impulse).unwrap().unwrap();
    hresp.extend(probe.flush().unwrap().unwrap_or_default());
    let delay = hresp.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as f64 / 4.0;
    let mut plot = Plot::new("Interpolation by 4 with the default polyphase filter", "time (input samples)", "value");
    plot.series.push(Series::new("input", x.iter().enumerate().map(|(i, &v)| (i as f64, v)).collect(), Style::Markers, 1));
    plot.series.push(Series::new("output (delay removed)", y.iter().enumerate().map(|(i, &v)| (i as f64 / 4.0 - delay, v)).collect(), Style::Line, 0));
    plot.x_range = Some((0.0, 23.0));
    plot.legend = Legend::BottomRight;
    write(CHAPTER, "interpolation.svg", &plot.render());
    // Arbitrary-ratio timing error against up_rate.
    let sig: Vec<f64> = (0..3000).map(|n| (0.07 * n as f64).sin()).collect();
    let mut rows = Vec::new();
    for upr in [4usize, 16, 64, 256] {
        let down = upr as f64 * 1.37;
        // An explicit Kaiser low-pass (cutoff at the output Nyquist), so the delay is known.
        let max_rate = down.ceil();
        let len = 20 * max_rate as usize + 1;
        let taps = lib_jsl_dsp::discrete::filters::fir::firwin::firwin(len, &[0.5 / max_rate], None, WindowType::Kaiser { beta: 8.0 }, lib_jsl_dsp::discrete::filters::fir::firwin::FirwinPassZero::True, true, 1.0).unwrap();
        let d = (len - 1) as f64 / 2.0;
        let mut r = PolyphaseArbitraryResampler::new(upr, down, Some(&taps)).unwrap();
        let out = r.process(&sig).unwrap().unwrap();
        let err = |exact: bool| out.iter().enumerate().skip(50).take(out.len().saturating_sub(100)).map(|(m, v)| {
            let p = if exact { m as f64 * down } else { (m as f64 * down).floor() };
            (v - (0.07 * ((p - d) / upr as f64)).sin()).abs()
        }).fold(0.0, f64::max);
        rows.push(vec![format!("{upr}"), format!("{down:.2}"), format!("{:.1e}", err(false)), format!("{:.1e}", err(true))]);
    }
    write(CHAPTER, "arbitrary_resampling.md", &md_table(&["up_rate", "down_rate (ratio 1/1.37)", "error against the rounded output times", "error against the exact output times"], &rows));
    // Half-band filter response and the CIC droop.
    let hb = halfband_lowpass(31, 8.0).unwrap();
    let mut plot = Plot::new("Half-band low-pass, 31 taps (14 of them zero), Kaiser β = 8", "frequency (× input Nyquist)", "gain (dB)");
    let n = 4096;
    let mut padded = hb.clone();
    padded.resize(n, 0.0);
    let spec = RealFft::new(n).unwrap().forward(&padded).unwrap();
    plot.series.push(Series::new("", spec.iter().enumerate().map(|(k, v)| (2.0 * k as f64 / n as f64, 20.0 * v.norm().max(1e-12).log10())).collect(), Style::Line, 0));
    plot.y_range = Some((-100.0, 5.0));
    write(CHAPTER, "halfband.svg", &plot.render());
    let zeros = hb.iter().filter(|v| **v == 0.0).count();
    let mut dec = HalfBandDecimator::new(&hb).unwrap();
    let tone: Vec<f64> = (0..400).map(|i| (2.0 * PI * 0.05 * i as f64).sin()).collect();
    let out = dec.process(&tone);
    write(CHAPTER, "halfband_facts.md", &format!("The 31-tap half-band filter has {zeros} zero taps; decimating 400 samples gives {} outputs.\n", out.len()));
    let mut plot = Plot::new("CIC decimator by R = 8 (M = 1): response in units of the output rate", "frequency (cycles per output sample)", "gain (dB)");
    for (k, stages) in [1usize, 2, 3, 4].into_iter().enumerate() {
        plot.series.push(Series::new(&format!("N = {stages}"), (1..800).map(|i| {
            let f_out = 4.0 * i as f64 / 800.0;
            (f_out, 20.0 * cic_magnitude_response(8, stages, 1, f_out / 8.0).max(1e-10).log10())
        }).collect(), Style::Line, k));
    }
    plot.y_range = Some((-100.0, 5.0));
    plot.legend = Legend::BottomLeft;
    write(CHAPTER, "cic_response.svg", &plot.render());
    // CIC: impulse response matches the formula, and gain.
    let mut cic = CicDecimator::new(1, 3, 4).unwrap();
    let mut impulse = vec![0i64; 40];
    impulse[0] = 1;
    let h = cic.process(&impulse);
    let n2 = 512;
    let mut padded: Vec<f64> = h.iter().map(|&v| v as f64).collect();
    padded.resize(n2, 0.0);
    let spec = RealFft::new(n2).unwrap().forward(&padded).unwrap();
    let err = spec.iter().enumerate().map(|(k, v)| (v.norm() / 64.0 - cic_magnitude_response(1, 3, 4, k as f64 / n2 as f64)).abs()).fold(0.0, f64::max);
    let mut big = CicDecimator::new(64, 5, 1).unwrap();
    let ones = vec![1000i64; 64 * 20];
    let out = big.process(&ones);
    write(CHAPTER, "cic_facts.md", &format!("The impulse response of a 3-stage CIC with RM = 4 matches |sin(4πf)/sin(πf)|³ to {err:.1e}. A 5-stage decimator by 64 turns a constant input of 1000 into {} (gain 64⁵ = {:.0}), with integrator registers that wrap around without corrupting the output.\n", out.last().unwrap(), big.gain()));
}

// ------------------------------------------------------------------ filter banks

fn filter_bank_figures() {
    let m = 16;
    let mut plot = Plot::new("Analysis filter bank, 16 channels: response of channels 0–3 to a swept tone", "frequency (cycles per sample)", "channel power (dB)");
    for (k, &os) in [1usize, 2].iter().enumerate() {
        let freqs: Vec<f64> = (0..240).map(|i| -0.05 + 0.3 * i as f64 / 239.0).collect();
        let mut responses = vec![Vec::new(); 4];
        for &f in &freqs {
            let x: Vec<Complex<f64>> = (0..2048).map(|n| Complex::from_polar(1.0, 2.0 * PI * f * n as f64)).collect();
            let frames = if os == 1 {
                CriticallySampledPolyphaseAnalysisFilterBank::new(m, None).unwrap().process(&x).unwrap().unwrap()
            } else {
                OversampledPolyphaseAnalysisFilterBank::new(m, os, None).unwrap().process(&x).unwrap().unwrap()
            };
            for (c, r) in responses.iter_mut().enumerate() {
                let p = frames[frames.len() / 2..].iter().map(|fr| fr[c].norm_sqr()).sum::<f64>() / (frames.len() - frames.len() / 2) as f64;
                r.push((f, p));
            }
        }
        if os == 1 {
            for (c, r) in responses.into_iter().enumerate() {
                let peak = r.iter().map(|p| p.1).fold(0.0, f64::max);
                plot.series.push(Series::new(&format!("channel {c}"), r.into_iter().map(|(f, p)| (f, db10(p / peak))).collect(), Style::Line, c));
            }
        }
        let _ = k;
    }
    plot.y_range = Some((-90.0, 5.0));
    plot.legend = Legend::BottomRight;
    write(CHAPTER, "filter_bank_channels.svg", &plot.render());
    // Reconstruction through analysis and synthesis.
    let mut rng = Rng::new(9);
    let x: Vec<Complex<f64>> = (0..4096).map(|_| Complex::new(rng.gauss(), rng.gauss())).collect();
    let mut rows = Vec::new();
    for (name, os, prototype) in [("critically sampled", 1usize, "one tap per phase (rectangular)"), ("critically sampled", 1, "default (Kaiser, 60 dB)"), ("oversampled ×2", 2, "one tap per phase (rectangular)"), ("oversampled ×2", 2, "default (Kaiser, 60 dB)"), ("oversampled ×4", 4, "default (Kaiser, 60 dB)")] {
        let rect = vec![1.0; m];
        let taps = if prototype.starts_with("one") { Some(rect.as_slice()) } else { None };
        let y: Vec<Complex<f64>> = if os == 1 {
            let mut a = CriticallySampledPolyphaseAnalysisFilterBank::new(m, taps).unwrap();
            let mut s = CriticallySampledPolyphaseSynthesisFilterBank::new(m, taps).unwrap();
            let mut fr = a.process(&x).unwrap().unwrap();
            fr.extend(a.flush().unwrap().unwrap_or_default());
            let mut y = s.process(&fr).unwrap().unwrap();
            y.extend(s.flush().unwrap().unwrap_or_default());
            y
        } else {
            let mut a = OversampledPolyphaseAnalysisFilterBank::new(m, os, taps).unwrap();
            let mut s = OversampledPolyphaseSynthesisFilterBank::new(m, os, taps).unwrap();
            let mut fr = a.process(&x).unwrap().unwrap();
            fr.extend(a.flush().unwrap().unwrap_or_default());
            let mut y = s.process(&fr).unwrap().unwrap();
            y.extend(s.flush().unwrap().unwrap_or_default());
            y
        };
        // Best delay and complex gain.
        let mut best = (f64::INFINITY, 0usize, Complex::new(0.0, 0.0));
        for d in 0..400 {
            if d + 3500 > y.len() {
                break;
            }
            let num: Complex<f64> = (500..3500).map(|i| y[i + d] * x[i].conj()).sum();
            let den: f64 = (500..3500).map(|i| x[i].norm_sqr()).sum();
            let g = num / den;
            let e = (500..3500).map(|i| (y[i + d] - g * x[i]).norm_sqr()).sum::<f64>() / (500..3500).map(|i| (g * x[i]).norm_sqr()).sum::<f64>();
            if e < best.0 {
                best = (e, d, g);
            }
        }
        rows.push(vec![name.to_string(), prototype.to_string(), format!("{}", best.1), format!("{:.4}", best.2.norm()), format!("{:.1e}", best.0.sqrt())]);
    }
    write(CHAPTER, "filter_bank_reconstruction.md", &md_table(&["bank (16 channels)", "prototype filter", "best delay (samples)", "best gain", "relative RMS error after removing delay and gain"], &rows));
}
