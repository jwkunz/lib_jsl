//! Figures and tables for the DSP chapter on signal sources, communications, synchronisation,
//! radar and control.

use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_core::{C1D, C2D};
use lib_jsl_dsp::discrete::controls::kalman::KalmanFilter;
use lib_jsl_dsp::discrete::controls::pid::PidController;
use lib_jsl_dsp::discrete::digital_communications::barker_codes::{barker_autocorrelation, barker_code};
use lib_jsl_dsp::discrete::digital_communications::constellations::Constellation;
use lib_jsl_dsp::discrete::digital_communications::decision_feedback_equalizer::DecisionFeedbackEqualizer;
use lib_jsl_dsp::discrete::digital_communications::farrow_filter::farrow_filter;
use lib_jsl_dsp::discrete::digital_communications::fm::{carson_bandwidth, fm_demodulate, fm_modulate};
use lib_jsl_dsp::discrete::digital_communications::mueller_timing_detector::MuellerTimingRecovery;
use lib_jsl_dsp::discrete::digital_communications::ofdm::{Ofdm, OfdmParams};
use lib_jsl_dsp::discrete::digital_communications::pulse_correlation_tracker::PulseCorrelationTracker;
use lib_jsl_dsp::discrete::radar::pulse_compression::{linear_chirp, peak_sidelobe_level_db, pulse_compress};
use lib_jsl_dsp::discrete::sources::cordic::Cordic32;
use lib_jsl_dsp::discrete::sources::nco::NcoSource;
use lib_jsl_dsp::discrete::sources::sinusoid::SineWave;
use lib_jsl_dsp::discrete::spectral::welch::{Detrend, SpectralScaling, periodogram};
use lib_jsl_dsp::discrete::stream_operator::StreamOperator;
use lib_jsl_dsp::discrete::synchronization::costas_loop::{CostasLoop, CostasModulation};
use lib_jsl_dsp::discrete::synchronization::phase_locked_loop::PhaseLockedLoop;
use lib_jsl_dsp::discrete::windows::{WindowType, generate_window};
use lib_jsl_random::uniform_generator::UniformGenerator;
use lib_jsl_random::xoshiro256plusplus::Xoshiro256PlusPlus;
use lib_jsl_special_functions::error_function::erfc;
use ndarray::{Array1, Array2, arr2};
use num_complex::Complex;
use std::f64::consts::PI;

const CHAPTER: &str = "comms";
type C64 = Complex<f64>;

pub fn generate() {
    source_figures();
    modulation_figures();
    synchronisation_figures();
    equalisation_figures();
    radar_figures();
    control_figures();
    controls_scipy_reference();
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
    fn cgauss(&mut self, sigma: f64) -> C64 {
        C64::new(self.gauss(), self.gauss()) * (sigma / 2f64.sqrt())
    }
    fn bits(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| u8::from(self.uniform() < 0.5)).collect()
    }
}

fn db10(v: f64) -> f64 {
    10.0 * v.max(1e-300).log10()
}

// ------------------------------------------------------------------ sources

fn source_figures() {
    // CORDIC accuracy against iterations.
    let mut rows = Vec::new();
    for iterations in [8u32, 12, 16, 20, 24, 28] {
        let c = Cordic32::new(iterations).unwrap();
        let err = (0..10000).map(|k| {
            let a = 2.0 * PI * k as f64 / 10000.0;
            let (s, co) = c.sin_cos(a);
            (s - a.sin()).abs().max((co - a.cos()).abs())
        }).fold(0.0, f64::max);
        rows.push(vec![format!("{iterations}"), format!("{err:.1e}"), format!("{:.1}", -20.0 * err.log10() / 6.02)]);
    }
    write(CHAPTER, "cordic_accuracy.md", &md_table(&["CORDIC iterations (32-bit)", "largest error in sin and cos", "equivalent bits"], &rows));
    // Spurs of a sine from a short CORDIC.
    let n = 8192;
    let fs = 1.0;
    let f0 = 1031.0 / n as f64 + 0.123 / n as f64;
    let mut plot = Plot::new("Spectrum of an NCO sine: exact against an 8-iteration CORDIC", "frequency (cycles per sample)", "power (dB)");
    for (k, (name, mut src)) in [("sin() kernel", NcoSource::<SineWave>::sinusoid(f0, fs).unwrap()), ("CORDIC, 8 iterations", NcoSource::<SineWave>::sinusoid_cordic(f0, fs, 8).unwrap())].into_iter().enumerate() {
        let x = src.generate(n);
        let p = periodogram(&x, fs, &generate_window(WindowType::BlackmanHarris, n, false), Detrend::None, SpectralScaling::Spectrum).unwrap();
        plot.series.push(Series::new(name, p.frequencies.iter().zip(&p.power).map(|(&f, &v)| (f, db10(v / 0.5))).collect(), Style::Line, k));
    }
    plot.y_range = Some((-160.0, 5.0));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "nco_spurs.svg", &plot.render());
}

// ------------------------------------------------------------------ modulation

fn modulation_figures() {
    // Constellations.
    let mut plot = Plot::new("QPSK, 8-PSK and 16-QAM (unit average power)", "in-phase", "quadrature");
    for (k, (name, c)) in [("QPSK", Constellation::qpsk()), ("8-PSK", Constellation::psk(8).unwrap()), ("16-QAM", Constellation::qam(16).unwrap())].into_iter().enumerate() {
        plot.series.push(Series::new(name, c.points().iter().map(|p| (p.re, p.im)).collect(), Style::Markers, k));
    }
    plot.x_range = Some((-1.8, 1.8));
    plot.y_range = Some((-1.2, 1.2));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "constellations.svg", &plot.render());
    // Gray labelling check and minimum distances.
    let mut rows = Vec::new();
    for (name, c) in [("BPSK", Constellation::bpsk()), ("QPSK", Constellation::qpsk()), ("8-PSK", Constellation::psk(8).unwrap()), ("16-QAM", Constellation::qam(16).unwrap()), ("64-QAM", Constellation::qam(64).unwrap())] {
        let dmin = c.min_distance();
        let pts = c.points();
        let mut nearest_pairs = 0;
        let mut one_bit = 0;
        for i in 0..pts.len() {
            for j in i + 1..pts.len() {
                if ((pts[i] - pts[j]).norm() - dmin).abs() < 1e-9 {
                    nearest_pairs += 1;
                    if (i ^ j).count_ones() == 1 {
                        one_bit += 1;
                    }
                }
            }
        }
        rows.push(vec![name.to_string(), format!("{}", c.bits_per_symbol()), format!("{:.4}", c.average_power()), format!("{dmin:.4}"), format!("{one_bit} of {nearest_pairs}")]);
    }
    write(CHAPTER, "constellation_table.md", &md_table(&["constellation", "bits per symbol", "average power", "minimum distance", "nearest neighbours differing in one bit"], &rows));
    // Bit error rate over AWGN.
    let mut plot = Plot::new("Bit error rate over AWGN: simulation (markers) and theory (lines)", "Eb/N0 (dB)", "bit error rate");
    plot.log_y = true;
    let mut rng = Rng::new(5);
    for (k, (name, c)) in [("BPSK", Constellation::bpsk()), ("QPSK", Constellation::qpsk()), ("16-QAM", Constellation::qam(16).unwrap()), ("64-QAM", Constellation::qam(64).unwrap())].into_iter().enumerate() {
        let bps = c.bits_per_symbol();
        let mut sim = Vec::new();
        for i in 0..=12 {
            let ebn0_db = i as f64;
            let ebn0 = 10f64.powf(ebn0_db / 10.0);
            let n0 = c.average_power() / (bps as f64 * ebn0);
            let nbits = 120_000 / bps * bps;
            let bits = rng.bits(nbits);
            let tx = c.map_bits(&bits).unwrap();
            let rx: Vec<C64> = tx.iter().map(|&s| s + rng.cgauss(n0.sqrt())).collect();
            let errors = c.demap_hard(&rx).iter().zip(&bits).filter(|(a, b)| a != b).count();
            if errors > 20 {
                sim.push((ebn0_db, errors as f64 / nbits as f64));
            }
        }
        plot.series.push(Series::new(name, sim, Style::Markers, k));
        let m = (1usize << bps) as f64;
        let theory: Vec<(f64, f64)> = (0..=120).map(|i| {
            let ebn0_db = 0.1 * i as f64;
            let ebn0 = 10f64.powf(ebn0_db / 10.0);
            let q = |x: f64| 0.5 * erfc(x / 2f64.sqrt());
            let p = if bps <= 2 { q((2.0 * ebn0).sqrt()) } else { 4.0 / bps as f64 * (1.0 - 1.0 / m.sqrt()) * q((3.0 * bps as f64 * ebn0 / (m - 1.0)).sqrt()) };
            (ebn0_db, p)
        }).collect();
        plot.series.push(Series::new("", theory, Style::Line, k));
    }
    plot.y_range = Some((1e-5, 0.5));
    plot.legend = Legend::BottomLeft;
    write(CHAPTER, "ber.svg", &plot.render());
    // OFDM over a multipath channel.
    let params = OfdmParams::wifi_like();
    let mut ofdm = Ofdm::new(params.clone()).unwrap();
    let qam = Constellation::qam(16).unwrap();
    let nd = params.num_data();
    let mut rng = Rng::new(9);
    let symbols_n = 40;
    let bits = rng.bits(nd * symbols_n * 4);
    let data = qam.map_bits(&bits).unwrap();
    let tx = ofdm.modulate(&data).unwrap();
    let channel = [C64::new(1.0, 0.0), C64::new(0.0, 0.5), C64::new(-0.3, 0.2), C64::new(0.1, 0.0)];
    let rx_clean: Vec<C64> = (0..tx.len()).map(|n| (0..channel.len()).filter(|&k| k <= n).map(|k| channel[k] * tx[n - k]).sum()).collect();
    let noise = 10f64.powf(-25.0 / 20.0);
    let rx: Vec<C64> = rx_clean.iter().map(|&v| v + rng.cgauss(noise * (tx.iter().map(|t| t.norm_sqr()).sum::<f64>() / tx.len() as f64).sqrt())).collect();
    let dem = ofdm.demodulate(&rx).unwrap();
    let evm = (dem.data.iter().zip(&data).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>() / data.len() as f64).sqrt();
    let errs = qam.demap_hard(&dem.data).iter().zip(&bits).filter(|(a, b)| a != b).count();
    // The first OFDM symbol as a known training symbol, then a fixed channel estimate.
    let len = params.symbol_len();
    let h_est = ofdm.estimate_channel(&rx[..len], &data[..nd]).unwrap();
    let trained = ofdm.demodulate_with_channel(&rx[len..], &h_est).unwrap();
    let evm_t = (trained.iter().zip(&data[nd..]).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>() / trained.len() as f64).sqrt();
    let errs_t = qam.demap_hard(&trained).iter().zip(&bits[nd * 4..]).filter(|(a, b)| a != b).count();
    let raw = ofdm.demodulate(&tx).unwrap();
    let raw_err = (raw.data.iter().zip(&data).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max)).max(1e-300);
    let mut plot = Plot::new("16-QAM OFDM through a 4-tap multipath channel at 25 dB SNR", "in-phase", "quadrature");
    plot.series.push(Series::new("pilot interpolation", dem.data.iter().take(1500).map(|p| (p.re, p.im)).collect(), Style::Markers, 7));
    plot.series.push(Series::new("training-symbol estimate", trained.iter().take(1500).map(|p| (p.re, p.im)).collect(), Style::Markers, 0));
    plot.legend = Legend::TopRight;
    plot.series.push(Series::new("", qam.points().iter().map(|p| (p.re, p.im)).collect(), Style::Markers, 6));
    plot.x_range = Some((-2.0, 2.0));
    plot.y_range = Some((-1.4, 1.4));
    write(CHAPTER, "ofdm_constellation.svg", &plot.render());
    write(CHAPTER, "ofdm_table.md", &md_table(&["channel estimate (25 dB SNR, 4-tap channel)", "error vector magnitude", "bit errors"], &[
        vec!["linear interpolation between the 4 pilots".into(), format!("{:.1}%", 100.0 * evm), format!("{errs} of {}", bits.len())],
        vec!["one known training symbol".into(), format!("{:.1}%", 100.0 * evm_t), format!("{errs_t} of {}", bits.len() - nd * 4)],
    ]));
    write(CHAPTER, "ofdm_facts.md", &format!("Back-to-back, modulation and demodulation agree to {raw_err:.1e}.\n"));
    // FM.
    let fs = 48000.0;
    let message: Vec<f64> = (0..4800).map(|n| (2.0 * PI * 1000.0 * n as f64 / fs).sin() + 0.5 * (2.0 * PI * 2500.0 * n as f64 / fs).sin()).collect();
    let deviation = 5000.0;
    let bb = fm_modulate(&message, deviation, fs).unwrap();
    let demod = fm_demodulate(&bb, deviation, fs).unwrap();
    let err = demod.iter().zip(&message).skip(10).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    let mut plot = Plot::new("FM baseband spectrum: 5 kHz deviation, tones at 1 and 2.5 kHz", "frequency (kHz)", "power (dB)");
    let p = lib_jsl_dsp::discrete::spectral::welch::welch(&bb.iter().map(|c| c.re).collect::<Vec<_>>(), fs, &generate_window(WindowType::Hann, 1024, false), 512, Detrend::None, SpectralScaling::Density).unwrap();
    let peak = p.power.iter().fold(0.0, |a: f64, &b| a.max(b));
    plot.series.push(Series::new("", p.frequencies.iter().zip(&p.power).map(|(&f, &v)| (f / 1000.0, db10(v / peak))).collect(), Style::Line, 0));
    let carson = carson_bandwidth(1.5 * deviation, 2500.0) / 2000.0;
    plot.series.push(Series::new("half the Carson bandwidth", vec![(carson, 5.0), (carson, -100.0)], Style::Dashed, 6));
    plot.x_range = Some((0.0, 24.0));
    plot.y_range = Some((-100.0, 5.0));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "fm_spectrum.svg", &plot.render());
    write(CHAPTER, "fm_facts.md", &format!("The discriminator recovers the two-tone message to {err:.1e}, and Carson's rule gives a bandwidth of {:.1} kHz for the peak deviation of 7.5 kHz.\n", carson_bandwidth(1.5 * deviation, 2500.0) / 1000.0));
}

// ------------------------------------------------------------------ synchronisation

fn synchronisation_figures() {
    // PLL acquisition of a tone with a frequency offset, for two bandwidths.
    let mut plot = Plot::new("PLL phase error locking to a tone 0.002 cycles/sample off", "sample", "phase error (radians)");
    for (k, bn) in [0.01, 0.03].into_iter().enumerate() {
        let mut pll = PhaseLockedLoop::new(bn, 0.707).unwrap();
        let errs: Vec<(f64, f64)> = (0..1500).map(|n| (n as f64, pll.step(C64::from_polar(1.0, 2.0 * PI * 0.002 * n as f64 + 2.0)).1)).collect();
        plot.series.push(Series::new(&format!("BnT = {bn}"), errs, Style::Line, k));
    }
    plot.legend = Legend::TopRight;
    write(CHAPTER, "pll_lock.svg", &plot.render());
    // Costas loop on QPSK.
    let mut rng = Rng::new(3);
    let q = Constellation::qpsk();
    let bits = rng.bits(4000);
    let sym = q.map_bits(&bits).unwrap();
    let rx: Vec<C64> = sym.iter().enumerate().map(|(n, &s)| s * C64::from_polar(1.0, 0.7 + 0.004 * n as f64) + rng.cgauss(0.1)).collect();
    let mut costas = CostasLoop::new(CostasModulation::Qpsk, 0.02, 0.707).unwrap();
    let out = costas.process(&rx);
    let mut plot = Plot::new("QPSK with a residual carrier: before and after a Costas loop", "in-phase", "quadrature");
    plot.series.push(Series::new("received (rotating)", rx.iter().take(600).map(|c| (c.re, c.im)).collect(), Style::Markers, 7));
    plot.series.push(Series::new("after the loop (last 1000)", out.iter().skip(1000).take(1000).map(|c| (c.re, c.im)).collect(), Style::Markers, 0));
    plot.x_range = Some((-2.0, 2.0));
    plot.y_range = Some((-1.4, 1.4));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "costas.svg", &plot.render());
    write(CHAPTER, "costas_facts.md", &format!("The loop's frequency estimate settles at {:.5} radians per symbol (true 0.00400).\n", costas.frequency()));
    // Timing recovery with a fractional offset.
    let sps = 4.0;
    let bits = rng.bits(4000);
    let sym = q.map_bits(&bits).unwrap();
    // Raised-cosine-like pulse shaping (Hann-windowed sinc) at 4 samples per symbol, then a 0.37-sample delay.
    let span = 8;
    let pulse: Vec<f64> = (-(span * 4)..=(span * 4)).map(|k| {
        let t = k as f64 / sps;
        let sinc = if t == 0.0 { 1.0 } else { (PI * t).sin() / (PI * t) };
        let w = 0.5 + 0.5 * (PI * k as f64 / (span * 4) as f64).cos();
        sinc * w
    }).collect();
    let mut up = vec![C64::new(0.0, 0.0); sym.len() * 4];
    for (i, s) in sym.iter().enumerate() {
        up[4 * i] = *s;
    }
    let shaped: Vec<C64> = (0..up.len()).map(|n| (0..pulse.len()).filter(|&k| k <= n).map(|k| up[n - k] * pulse[k]).sum()).collect();
    let delayed = farrow_filter(&shaped, 0.37, 3).unwrap();
    let mut mm = MuellerTimingRecovery::new(sps, q.clone()).unwrap();
    let rec = mm.process(&delayed).unwrap().unwrap();
    let mut plot = Plot::new("Mueller-Müller timing recovery at 4 samples per symbol (offset 0.37 samples)", "in-phase", "quadrature");
    plot.series.push(Series::new("first 100 symbols", rec.iter().take(100).map(|c| (c.re, c.im)).collect(), Style::Markers, 7));
    plot.series.push(Series::new("after 500 symbols", rec.iter().skip(500).map(|c| (c.re, c.im)).collect(), Style::Markers, 0));
    plot.x_range = Some((-2.0, 2.0));
    plot.y_range = Some((-1.4, 1.4));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "timing_recovery.svg", &plot.render());
    let tail: Vec<C64> = rec.iter().skip(500).copied().collect();
    let evm = (tail.iter().map(|y| (y - q.slice(*y)).norm_sqr()).sum::<f64>() / tail.len() as f64).sqrt();
    write(CHAPTER, "timing_facts.md", &format!("After acquisition the symbols' error vector magnitude is {:.2}% and the estimated symbol period {:.4} samples (nominal 4).\n", 100.0 * evm, mm.symbol_period()));
    // Farrow fractional delay accuracy against order.
    let mut rows = Vec::new();
    for order in [1usize, 3, 5, 7] {
        let mut worst = Vec::new();
        for f in [0.05, 0.15, 0.3] {
            let x: Vec<f64> = (0..400).map(|n| (2.0 * PI * f * n as f64).sin()).collect();
            let y = farrow_filter(&x, 0.4, order).unwrap();
            let err = y.iter().enumerate().skip(20).take(300).map(|(n, &v)| (v - (2.0 * PI * f * (n as f64 - 0.4)).sin()).abs()).fold(0.0, f64::max);
            worst.push(format!("{err:.1e}"));
        }
        rows.push(vec![format!("{order}"), worst[0].clone(), worst[1].clone(), worst[2].clone()]);
    }
    write(CHAPTER, "farrow.md", &md_table(&["Lagrange order", "error at 0.05 cycles/sample", "at 0.15", "at 0.30"], &rows));
}

// ------------------------------------------------------------------ equalisation

fn equalisation_figures() {
    let mut rng = Rng::new(11);
    let q = Constellation::qpsk();
    let bits = rng.bits(6000);
    let sym = q.map_bits(&bits).unwrap();
    let h = [C64::new(1.0, 0.0), C64::new(0.45, 0.3), C64::new(-0.25, 0.1), C64::new(0.1, -0.05)];
    let rx: Vec<C64> = (0..sym.len()).map(|n| (0..h.len()).filter(|&k| k <= n).map(|k| h[k] * sym[n - k]).sum::<C64>() + rng.cgauss(0.05)).collect();
    let mut curves = Vec::new();
    let mut eq_out = Vec::new();
    for (name, mut eq) in [("LMS, μ = 0.05", DecisionFeedbackEqualizer::new(q.clone(), 8, 4, 0.05).unwrap()), ("RLS, λ = 0.995", DecisionFeedbackEqualizer::new_rls(q.clone(), 8, 4, 0.995).unwrap())] {
        let mut mse = Vec::new();
        let mut out = Vec::new();
        for (i, &x) in rx.iter().enumerate() {
            let target = if i < 300 { Some(sym[i]) } else { None };
            let (y, _) = eq.equalize_sample(x, target);
            out.push(y);
            mse.push((y - sym[i]).norm_sqr());
        }
        // Smooth over 25 symbols for display.
        let smooth: Vec<(f64, f64)> = (0..mse.len() / 25).map(|b| ((b * 25) as f64, db10(mse[b * 25..(b + 1) * 25].iter().sum::<f64>() / 25.0))).collect();
        curves.push((name, smooth));
        eq_out.push(out);
    }
    let mut plot = Plot::new("DFE learning curves: 300 training symbols, then decision-directed", "symbol", "squared error (dB)");
    for (k, (name, c)) in curves.into_iter().enumerate() {
        plot.series.push(Series::new(name, c.into_iter().filter(|p| p.0 < 2000.0).collect(), Style::Line, k));
    }
    plot.legend = Legend::TopRight;
    write(CHAPTER, "dfe_learning.svg", &plot.render());
    let mut plot = Plot::new("QPSK through an ISI channel: received and equalised (RLS DFE)", "in-phase", "quadrature");
    plot.series.push(Series::new("received", rx.iter().skip(1000).take(800).map(|c| (c.re, c.im)).collect(), Style::Markers, 7));
    plot.series.push(Series::new("equalised", eq_out[1].iter().skip(1000).take(800).map(|c| (c.re, c.im)).collect(), Style::Markers, 0));
    plot.x_range = Some((-2.6, 2.6));
    plot.y_range = Some((-1.8, 1.8));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "dfe_constellation.svg", &plot.render());
    let ser = |out: &[C64]| out.iter().zip(&sym).skip(1000).filter(|(y, s)| (q.slice(**y) - **s).norm() > 1e-9).count() as f64 / (sym.len() - 1000) as f64;
    let unequalised = rx.iter().zip(&sym).skip(1000).filter(|(y, s)| (q.slice(**y) - **s).norm() > 1e-9).count() as f64 / (sym.len() - 1000) as f64;
    write(CHAPTER, "dfe_table.md", &md_table(&["receiver", "symbol error rate after symbol 1000"], &[vec!["no equaliser".into(), format!("{unequalised:.4}")], vec!["DFE, LMS".into(), format!("{:.4}", ser(&eq_out[0]))], vec!["DFE, RLS".into(), format!("{:.4}", ser(&eq_out[1]))]]));
}

// ------------------------------------------------------------------ radar

fn radar_figures() {
    let fs = 100e6;
    let pulse = linear_chirp(20e6, 10e-6, fs).unwrap();
    let mut rx = vec![C64::new(0.0, 0.0); 3000];
    for (k, &p) in pulse.iter().enumerate() {
        rx[500 + k] += p;
        rx[560 + k] += 0.1 * p;
    }
    let hamming = generate_window(WindowType::Hamming, pulse.len(), true);
    let plain = pulse_compress(&rx, &pulse, None).unwrap();
    let weighted = pulse_compress(&rx, &pulse, Some(&hamming)).unwrap();
    let mut plot = Plot::new("Pulse compression of a 10 µs, 20 MHz chirp: targets at 500 and 560 samples (−20 dB)", "delay (samples)", "output (dB)");
    for (k, (name, c)) in [("matched filter", &plain), ("Hamming-weighted", &weighted)].into_iter().enumerate() {
        let peak = c.iter().map(|v| v.norm()).fold(0.0, f64::max);
        plot.series.push(Series::new(name, c.iter().enumerate().map(|(i, v)| (i as f64, 20.0 * (v.norm() / peak).max(1e-6).log10())).filter(|p| p.0 >= 400.0 && p.0 <= 700.0).collect(), Style::Line, k));
    }
    plot.y_range = Some((-80.0, 5.0));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "pulse_compression.svg", &plot.render());
    // Isolated target sidelobes.
    let mut single = vec![C64::new(0.0, 0.0); 3000];
    for (k, &p) in pulse.iter().enumerate() {
        single[500 + k] = p;
    }
    let mut rows = Vec::new();
    for (name, w) in [("matched (none)", None), ("Hamming", Some(hamming.clone())), ("Taylor n̄ = 5, −35 dB", Some(generate_window(WindowType::Taylor { nbar: 5, sll: 35.0, norm: true }, pulse.len(), true)))] {
        let c = pulse_compress(&single, &pulse, w.as_deref()).unwrap();
        let peak = c.iter().map(|v| v.norm()).fold(0.0, f64::max);
        let width = c.iter().filter(|v| v.norm() >= peak / 2f64.sqrt()).count();
        rows.push(vec![name.to_string(), format!("{:.1}", peak_sidelobe_level_db(&c)), format!("{width}")]);
    }
    for len in [7usize, 11, 13] {
        let code = barker_code(len).unwrap();
        let ac = barker_autocorrelation(&code);
        let peak = *ac.iter().max().unwrap() as f64;
        let side = ac.iter().filter(|&&v| (v as f64) < peak).map(|v| v.abs()).max().unwrap() as f64;
        rows.push(vec![format!("Barker {len} (phase code)"), format!("{:.1}", 20.0 * (side / peak).log10()), "1 chip".to_string()]);
    }
    write(CHAPTER, "sidelobes.md", &md_table(&["waveform and weighting", "peak sidelobe level (dB)", "−3 dB width (samples)"], &rows));
    // A pulse drifting through successive frames, tracked.
    let template: Vec<C64> = (0..64).map(|k| {
        let c = k as f64 - 32.0;
        C64::from_polar(1.0, PI * 0.006 * c * c)
    }).collect();
    let mut rng = Rng::new(4);
    let mut tracker = PulseCorrelationTracker::new(&template, 6, 0.5, 0.1).unwrap();
    let mut truth = Vec::new();
    let mut est = Vec::new();
    for frame_i in 0..80 {
        let delay = 100.0 + 1.3 * frame_i as f64 + 6.0 * (frame_i as f64 / 12.0).sin();
        let mut frame = vec![C64::new(0.0, 0.0); 400];
        for (n, v) in frame.iter_mut().enumerate() {
            let t = n as f64 - delay;
            if (0.0..64.0).contains(&t) {
                let c = t - 32.0;
                *v = C64::from_polar(1.0, PI * 0.006 * c * c);
            }
            *v += rng.cgauss(1.0);
        }
        let s = if frame_i == 0 { tracker.acquire(&frame).unwrap() } else { tracker.update(&frame).unwrap() };
        truth.push((frame_i as f64, delay));
        est.push((frame_i as f64, s.delay));
    }
    let rms = (truth.iter().zip(&est).skip(5).map(|(a, b)| (a.1 - b.1).powi(2)).sum::<f64>() / 75.0).sqrt();
    let mut plot = Plot::new(&format!("α-β tracking of a pulse in noise (SNR 0 dB per sample): RMS error {rms:.2} samples"), "frame", "pulse delay (samples)");
    plot.series.push(Series::new("true delay", truth, Style::Line, 7));
    plot.series.push(Series::new("tracked", est, Style::Markers, 0));
    plot.legend = Legend::TopLeft;
    write(CHAPTER, "pulse_tracking.svg", &plot.render());
}

// ------------------------------------------------------------------ control

fn control_figures() {
    // Kalman filter for a constant-velocity target, position measured.
    let dt = 1.0;
    let q = 0.01;
    let r = 4.0;
    let f = arr2(&[[1.0, dt], [0.0, 1.0]]).mapv(|v| C64::new(v, 0.0));
    let h = arr2(&[[1.0, 0.0]]).mapv(|v| C64::new(v, 0.0));
    let qm = arr2(&[[dt.powi(3) / 3.0, dt.powi(2) / 2.0], [dt.powi(2) / 2.0, dt]]).mapv(|v| C64::new(q * v, 0.0));
    let rm: C2D = Array2::from_elem((1, 1), C64::new(r, 0.0));
    let p0: C2D = Array2::from_diag(&Array1::from(vec![C64::new(100.0, 0.0), C64::new(10.0, 0.0)]));
    let x0: C1D = Array1::from(vec![C64::new(0.0, 0.0), C64::new(0.0, 0.0)]);
    let mut kf = KalmanFilter::new(f, h, qm, rm, p0, x0, None, None).unwrap();
    let mut rng = Rng::new(17);
    let (mut pos, mut vel) = (0.0, 1.0);
    let (mut truth, mut meas, mut est, mut vest) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut nees = 0.0;
    let steps = 200;
    for k in 0..steps {
        // Continuous white-acceleration model, discretised.
        let a = rng.gauss() * q.sqrt();
        pos += vel * dt + 0.5 * a * dt;
        vel += a;
        let z = pos + r.sqrt() * rng.gauss();
        let out = kf.process(&[Array1::from(vec![C64::new(z, 0.0)])]).unwrap().unwrap();
        let xh = out[0].next_state.clone();
        let p = kf.estimate_covariance().clone();
        let e = [pos - xh[0].re, vel - xh[1].re];
        let det = p[[0, 0]].re * p[[1, 1]].re - p[[0, 1]].re * p[[1, 0]].re;
        if k >= 20 {
            nees += (p[[1, 1]].re * e[0] * e[0] - (p[[0, 1]].re + p[[1, 0]].re) * e[0] * e[1] + p[[0, 0]].re * e[1] * e[1]) / det;
        }
        truth.push((k as f64, pos));
        meas.push((k as f64, z));
        est.push((k as f64, xh[0].re));
        vest.push((k as f64, xh[1].re - vel));
    }
    let mut plot = Plot::new("Kalman filter: position of a constant-velocity target (measurement noise σ = 2)", "step", "position");
    plot.series.push(Series::new("measurements", meas.clone(), Style::Markers, 7));
    plot.series.push(Series::new("true", truth.clone(), Style::Line, 6));
    plot.series.push(Series::new("estimate", est.clone(), Style::Line, 0));
    plot.x_range = Some((100.0, 160.0));
    plot.y_range = Some((95.0, 175.0));
    plot.legend = Legend::TopLeft;
    write(CHAPTER, "kalman_track.svg", &plot.render());
    let rms_meas = (meas.iter().zip(&truth).skip(20).map(|(a, b)| (a.1 - b.1).powi(2)).sum::<f64>() / (steps - 20) as f64).sqrt();
    let rms_est = (est.iter().zip(&truth).skip(20).map(|(a, b)| (a.1 - b.1).powi(2)).sum::<f64>() / (steps - 20) as f64).sqrt();
    write(CHAPTER, "kalman_facts.md", &format!("After 20 steps the position error is {rms_est:.2} RMS against {rms_meas:.2} for the raw measurements, and the normalised estimation error squared averages {:.2} (2 for a consistent filter with a two-dimensional state).\n", nees / (steps - 20) as f64));
    // PID step responses on a first-order plant with dead time, with saturation.
    let simulate = |kp: f64, ki: Option<f64>, kd: Option<f64>, limits: Option<(f64, f64)>, anti: bool| {
        let mut pid = PidController::new_with_dt(Some(1.0), Some(kp), ki, kd, limits, Some(anti), 0.1).unwrap();
        let mut y = 0.0;
        let delay = 5;
        let mut u_hist = vec![0.0; delay];
        let mut out = Vec::new();
        for n in 0..600 {
            let u = pid.process(&[y]).unwrap().unwrap()[0].control;
            u_hist.push(u);
            let u_delayed = u_hist[u_hist.len() - 1 - delay];
            // Plant: τ dy/dt = −y + u, τ = 2 s, dt = 0.1 s.
            y += 0.1 / 2.0 * (-y + u_delayed);
            out.push((n as f64 * 0.1, y));
        }
        out
    };
    let mut plot = Plot::new("Step responses of P, PI and PID control (plant 1/(2s + 1) with 0.5 s dead time)", "time (s)", "output");
    plot.series.push(Series::new("P, Kp = 2", simulate(2.0, None, None, None, false), Style::Line, 0));
    plot.series.push(Series::new("PI, Kp = 2, Ki = 1", simulate(2.0, Some(1.0), None, None, false), Style::Line, 1));
    plot.series.push(Series::new("PID, Kp = 2, Ki = 1, Kd = 0.5", simulate(2.0, Some(1.0), Some(0.5), None, false), Style::Line, 2));
    plot.series.push(Series::new("setpoint", vec![(0.0, 1.0), (60.0, 1.0)], Style::Dashed, 6));
    plot.x_range = Some((0.0, 30.0));
    plot.legend = Legend::BottomRight;
    write(CHAPTER, "pid_steps.svg", &plot.render());
    let mut plot = Plot::new("Integrator windup with the output limited to [−1.2, 1.2] (PI, Kp = 1, Ki = 2)", "time (s)", "output");
    plot.series.push(Series::new("no anti-windup", simulate(1.0, Some(2.0), None, Some((-1.2, 1.2)), false), Style::Line, 1));
    plot.series.push(Series::new("anti-windup", simulate(1.0, Some(2.0), None, Some((-1.2, 1.2)), true), Style::Line, 0));
    plot.series.push(Series::new("setpoint", vec![(0.0, 1.0), (60.0, 1.0)], Style::Dashed, 6));
    plot.x_range = Some((0.0, 40.0));
    plot.legend = Legend::BottomRight;
    write(CHAPTER, "pid_windup.svg", &plot.render());
}

// ------------------------------------------------------------------ SciPy references

fn controls_scipy_reference() {
    use crate::refs::{Refs, fmt_diff, max_abs_diff};
    use lib_jsl_dsp::discrete::controls::state_space::StateSpace;
    use ndarray::{Array1, Array2};
    use num_complex::Complex;

    let refs = Refs::parse(include_str!("../data/controls_reference.txt"));
    let c = |rows: &[&[f64]]| Array2::from_shape_fn((rows.len(), rows[0].len()), |(i, j)| Complex::new(rows[i][j], 0.0));
    let a: C2D = c(&[&[0.9, 0.2, 0.0], &[-0.1, 0.8, 0.3], &[0.0, -0.2, 0.7]]);
    let b: C2D = c(&[&[1.0, 0.0], &[0.5, 0.2], &[0.0, 1.0]]);
    let cm: C2D = c(&[&[1.0, 0.0, -1.0], &[0.3, 1.0, 0.0]]);
    let d: C2D = c(&[&[0.0, 0.1], &[0.2, 0.0]]);
    let x0: C1D = Array1::from(vec![Complex::new(0.5, 0.0), Complex::new(-0.2, 0.0), Complex::new(0.1, 0.0)]);
    let mut sys = StateSpace::new(a, b, cm, d, x0).unwrap();
    let mut y = Vec::new();
    for k in 0..60 {
        let u: C1D = Array1::from(vec![Complex::new((0.2 * k as f64).sin(), 0.0), Complex::new(if k % 20 < 10 { 1.0 } else { -1.0 }, 0.0)]);
        y.extend(sys.step(u).unwrap().iter().map(|v| v.re));
    }
    let poles: Vec<Complex<f64>> = sys.get_poles().unwrap().to_vec();
    let want = refs.get("poles");
    let pole_diff = crate::refs::max_matched_distance(&poles, &want[0], &want[1]);
    let rows = vec![
        vec!["response of a 3-state, 2-input, 2-output system to 60 input samples (`signal.dlsim`)".to_string(), fmt_diff(max_abs_diff(&y, refs.values("dlsim")))],
        vec!["its poles, the eigenvalues of A (`linalg.eigvals`)".to_string(), fmt_diff(pole_diff)],
    ];
    write(CHAPTER, "scipy_controls.md", &md_table(&["computation (SciPy reference)", "largest difference"], &rows));
}
