//! Figures and tables for the fast Fourier transform chapter.

use crate::linear_algebra::time;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_ffts::fft_engine_trait::{FftDirection, FftEngine1D, FftOrdering, FftScaleFactor};
use lib_jsl_ffts::{BestFft, BluesteinFft, Fft, FractionalFft, RealFft, SimdFft, TrigNormalization, TrigTransformType, dct};
use lib_jsl_ffts_reference::{optimized_radix2::OptimizedRadix2FFT, optimized_split_radix::OptimizedSplitRadixFFT, simple_cooley_tukey::SimpleCooleyTukeyFFT};
use num_complex::Complex;
use std::f64::consts::PI;

const CHAPTER: &str = "fft";

type C = Complex<f64>;

/// A sum of four complex exponentials with integer frequencies, whose DFT is known exactly.
fn test_signal(n: usize) -> (Vec<C>, Vec<C>) {
    let freqs = [(1usize, 1.0, 0.5), (n / 3, -0.7, 0.2), (n / 2 + 5, 0.3, -1.1), (n - 2, 0.9, 0.0)];
    let x = (0..n)
        .map(|k| {
            freqs.iter().map(|&(f, re, im)| {
                // The angle is reduced exactly, so the signal itself is accurate to rounding.
                let a = 2.0 * PI * ((f % n * k) % n) as f64 / n as f64;
                C::new(re, im) * C::new(a.cos(), a.sin())
            }).sum()
        })
        .collect();
    let mut spectrum = vec![C::new(0.0, 0.0); n];
    for &(f, re, im) in &freqs {
        spectrum[f % n] += C::new(re, im) * n as f64;
    }
    (x, spectrum)
}

fn relative_error(engine: &mut dyn FftEngine1D, n: usize) -> f64 {
    let (x, want) = test_signal(n);
    engine.plan(n, FftScaleFactor::None, FftDirection::Forward, FftOrdering::Standard).unwrap();
    let got = engine.execute(&x).unwrap();
    let err = got.iter().zip(&want).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>().sqrt();
    (err / want.iter().map(|b| b.norm_sqr()).sum::<f64>().sqrt()).max(1e-17)
}

/// The direct O(n²) DFT, with exactly reduced angles.
fn direct_dft(x: &[C]) -> Vec<C> {
    let n = x.len();
    (0..n)
        .map(|k| x.iter().enumerate().map(|(m, v)| {
            let a = -2.0 * PI * ((k * m) % n) as f64 / n as f64;
            v * C::new(a.cos(), a.sin())
        }).sum())
        .collect()
}

fn accuracy_figure() {
    let mut plot = Plot::new("Accuracy of the FFT engines (relative 2-norm error)", "length n", "relative error");
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::TopLeft;
    plot.y_range = Some((1e-17, 1e-9));
    type Make = fn() -> Box<dyn FftEngine1D>;
    let engines: [(&str, Make, usize); 5] = [
        ("reference: radix 2, twiddles by recurrence", || Box::new(SimpleCooleyTukeyFFT::new()), 0),
        ("reference: radix 2, exact twiddles", || Box::new(OptimizedRadix2FFT::new()), 1),
        ("reference: split radix", || Box::new(OptimizedSplitRadixFFT::new()), 2),
        ("production: BestFft", || Box::new(BestFft::new()), 3),
        ("production: Bluestein (any length)", || Box::new(BluesteinFft::new()), 4),
    ];
    for (name, make, color) in engines {
        let pts = (3..=20).map(|p| {
            let n = 1usize << p;
            (n as f64, relative_error(make().as_mut(), n))
        }).collect();
        plot.series.push(Series::new(name, pts, Style::LineMarkers, color));
    }
    let primes = [7usize, 31, 127, 1021, 8191, 65537, 524287];
    plot.series.push(Series::new("production: Fft at prime lengths", primes.iter().map(|&n| (n as f64, relative_error(&mut Fft::new(), n))).collect(), Style::Markers, 6));
    plot.series.push(Series::new("ε log₂ n", (3..=20).map(|p| ((1usize << p) as f64, f64::EPSILON * p as f64)).collect(), Style::Dashed, 7));
    write(CHAPTER, "accuracy.svg", &plot.render());
}

fn speed_figure() {
    let mut plot = Plot::new("FFT speed (one core; 5 n log₂ n flops)", "length n", "GFLOP/s");
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::BottomRight;
    plot.y_range = Some((0.001, 50.0));
    let flops = |n: usize| 5.0 * n as f64 * (n as f64).log2();
    type Make = fn() -> Box<dyn FftEngine1D>;
    let engines: [(&str, Make, usize, Style); 6] = [
        ("reference: radix 2, twiddles by recurrence", || Box::new(SimpleCooleyTukeyFFT::new()), 0, Style::Dashed),
        ("reference: radix 2, exact twiddles", || Box::new(OptimizedRadix2FFT::new()), 1, Style::Dashed),
        ("reference: split radix (recursive)", || Box::new(OptimizedSplitRadixFFT::new()), 2, Style::Dashed),
        ("production: SimdFft", || Box::new(SimdFft::new()), 5, Style::LineMarkers),
        ("production: BestFft", || Box::new(BestFft::new()), 3, Style::LineMarkers),
        ("production: Bluestein at the same lengths", || Box::new(BluesteinFft::new()), 4, Style::LineMarkers),
    ];
    for (name, make, color, style) in engines {
        let pts = [6usize, 8, 10, 12, 14, 16, 18, 20]
            .iter()
            .map(|&p| {
                let n = 1usize << p;
                let mut e = make();
                e.plan(n, FftScaleFactor::None, FftDirection::Forward, FftOrdering::Standard).unwrap();
                let (x, _) = test_signal(n);
                (n as f64, flops(n) / time(|| { std::hint::black_box(e.execute(&x).unwrap()); }) / 1e9)
            })
            .collect();
        plot.series.push(Series::new(name, pts, style, color));
    }
    let direct = [64usize, 256, 1024, 4096]
        .iter()
        .map(|&n| {
            let (x, _) = test_signal(n);
            (n as f64, flops(n) / time(|| { std::hint::black_box(direct_dft(&x)); }) / 1e9)
        })
        .collect();
    plot.series.push(Series::new("direct O(n²) DFT (same flop count used)", direct, Style::LineMarkers, 6));
    write(CHAPTER, "speed.svg", &plot.render());
}

fn real_fft_table() {
    let mut table = String::from("| length $n$ | complex `Fft` of real data | `RealFft` | ratio |\n|---:|---:|---:|---:|\n");
    for n in [256usize, 1024, 4096, 65536, 1 << 20] {
        let x: Vec<f64> = (0..n).map(|k| (k as f64 * 0.01).sin()).collect();
        let xc: Vec<C> = x.iter().map(|&v| C::new(v, 0.0)).collect();
        let mut fft = Fft::new();
        fft.plan(n, FftScaleFactor::None, FftDirection::Forward, FftOrdering::Standard).unwrap();
        let mut rfft = RealFft::new(n).unwrap();
        // Repeat short transforms so that each timing lasts tens of milliseconds.
        let reps = (20_000_000 / n).max(1);
        let tc = time(|| { for _ in 0..reps { std::hint::black_box(fft.execute(&xc).unwrap()); } }) / reps as f64;
        let tr = time(|| { for _ in 0..reps { std::hint::black_box(rfft.forward(&x).unwrap()); } }) / reps as f64;
        let fmt = |t: f64| if t < 1e-3 { format!("{:.1} µs", t * 1e6) } else { format!("{:.2} ms", t * 1e3) };
        table.push_str(&format!("| {n} | {} | {} | {:.2} |\n", fmt(tc), fmt(tr), tr / tc));
    }
    write(CHAPTER, "real_fft.md", &table);
}

fn dct_figure() {
    // A smooth, non-periodic signal: its periodic extension jumps, its even extension does not.
    let n = 256;
    let x: Vec<f64> = (0..n).map(|k| { let t = (k as f64 + 0.5) / n as f64; (1.5 * t).exp() + 0.3 * t * t }).collect();
    let xc: Vec<C> = x.iter().map(|&v| C::new(v, 0.0)).collect();
    let mut fft = Fft::new();
    fft.plan(n, FftScaleFactor::None, FftDirection::Forward, FftOrdering::Standard).unwrap();
    let dft = fft.execute(&xc).unwrap();
    let c = dct(&x, TrigTransformType::II, TrigNormalization::Backward).unwrap();
    let mut plot = Plot::new("Coefficient decay for the smooth signal e^(1.5t) + 0.3t² on [0, 1]", "index k", "|coefficient| / |coefficient 0|");
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::BottomLeft;
    plot.series.push(Series::new("DFT (periodic extension jumps): ~ 1/k", (1..n / 2).map(|k| (k as f64, dft[k].norm() / dft[0].norm())).collect(), Style::Line, 0));
    plot.series.push(Series::new("DCT-II (even extension is continuous): ~ 1/k²", (1..n).map(|k| (k as f64, c[k].abs().max(1e-17) / c[0].abs())).collect(), Style::Line, 1));
    write(CHAPTER, "dct_compaction.svg", &plot.render());
}

fn fractional_figure() {
    // A rectangular pulse of width 32 samples centred on t = 0 (index 0, wrapping), N = 256: the
    // transform samples time at spacing 1/√N = 1/16, so the pulse is two units wide.
    let n = 256;
    let centred = |k: usize| if k < n / 2 { k as f64 } else { k as f64 - n as f64 };
    let x: Vec<C> = (0..n).map(|k| C::new(if centred(k).abs() < 16.0 { 1.0 } else { 0.0 }, 0.0)).collect();
    let mut frft = FractionalFft::new();
    let mut plot = Plot::new("Fractional Fourier transforms of a rectangular pulse, N = 256", "u = (sample index, centred) / √N", "magnitude");
    plot.legend = Legend::TopRight;
    for (k, alpha) in [0.0, 0.25, 0.5, 0.75, 1.0].into_iter().enumerate() {
        let y = frft.execute(&x, alpha).unwrap();
        let mut pts: Vec<(f64, f64)> = (0..n).map(|i| (centred(i) / 16.0, y[i].norm())).collect();
        pts.sort_by(|a, b| a.0.total_cmp(&b.0));
        plot.series.push(Series::new(&format!("α = {alpha}"), pts, Style::Line, k));
    }
    write(CHAPTER, "fractional.svg", &plot.render());
}

pub fn generate() {
    accuracy_figure();
    dct_figure();
    fractional_figure();
    real_fft_table();
    speed_figure();
}
