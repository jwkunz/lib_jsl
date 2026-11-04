//! Figures and tables for the chapter on error-correcting codes.

use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_information::forward_error_correction::{ConvolutionalCode, HammingCode, HammingStatus, LdpcCode, PolarCode, ReedSolomon};
use lib_jsl_random::uniform_generator::UniformGenerator;
use lib_jsl_random::xoshiro256plusplus::Xoshiro256PlusPlus;
use lib_jsl_special_functions::error_function::erfc;
use std::f64::consts::PI;
use std::time::Instant;

const CHAPTER: &str = "coding";

pub fn generate() {
    let mut summary: Vec<Series> = Vec::new();
    hamming_figures(&mut summary);
    reed_solomon_figures();
    convolutional_figures(&mut summary);
    ldpc_figures(&mut summary);
    polar_figures(&mut summary);
    let mut plot = Plot::new("Bit error rates of the codes of this chapter (BPSK, AWGN)", "Eb/N0 (dB)", "bit error rate");
    plot.series = summary;
    plot.log_y = true;
    // Room above the curves for the legend.
    plot.y_range = Some((1e-6, 100.0));
    plot.x_range = Some((0.0, 10.0));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "summary_ber.svg", &plot.render());
}

fn q(x: f64) -> f64 {
    0.5 * erfc(x / 2f64.sqrt())
}

fn uncoded(color: usize) -> Series {
    Series::new("uncoded BPSK", (0..=20).map(|k| k as f64 * 0.5).map(|e| (e, q((2.0 * 10f64.powf(e / 10.0)).sqrt()))).collect(), Style::Dashed, color)
}

/// BPSK over AWGN at code rate `rate`: channel LLRs for the bits.
fn channel(bits: &[u8], ebn0_db: f64, rate: f64, rng: &mut Rng) -> Vec<f64> {
    let sigma2 = 1.0 / (2.0 * rate * 10f64.powf(ebn0_db / 10.0));
    let sigma = sigma2.sqrt();
    bits.iter().map(|&b| {
        let y = (1.0 - 2.0 * b as f64) + sigma * rng.gauss();
        2.0 * y / sigma2
    }).collect()
}

fn random_bits(n: usize, rng: &mut Rng) -> Vec<u8> {
    (0..n).map(|_| (rng.uniform() < 0.5) as u8).collect()
}

/// Simulates frames until `min_frame_errors` frames contain errors or `max_bits` information bits
/// have been sent, returning the bit error rate. Errors come in bursts within a frame, so counting
/// failed frames gives steadier estimates than counting bits.
fn simulate(mut frame: impl FnMut(&mut Rng) -> (usize, usize), rng: &mut Rng, min_frame_errors: usize, max_bits: usize) -> f64 {
    let (mut errors, mut bits, mut failed) = (0, 0, 0);
    while failed < min_frame_errors && bits < max_bits {
        let (e, b) = frame(rng);
        errors += e;
        bits += b;
        failed += (e > 0) as usize;
    }
    errors as f64 / bits as f64
}

fn count_errors(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

// ------------------------------------------------------------------ Hamming

fn hamming_figures(summary: &mut Vec<Series>) {
    // Exhaustive outcomes of the (8,4) extended code for every error pattern of weight 1..3.
    let code = HammingCode::new_extended(3).unwrap();
    let plain = HammingCode::new(3).unwrap();
    let mut rows = Vec::new();
    for (name, c) in [("(7,4) Hamming", plain), ("(8,4) extended Hamming", code)] {
        let n = c.n();
        for w in 1..=3usize {
            let (mut correct, mut detected, mut wrong, mut total) = (0, 0, 0, 0);
            for data_v in 0..16usize {
                let data: Vec<u8> = (0..4).map(|i| (data_v >> i & 1) as u8).collect();
                let cw = c.encode(&data).unwrap();
                for pattern in 0..(1usize << n) {
                    if pattern.count_ones() as usize != w {
                        continue;
                    }
                    let rx: Vec<u8> = cw.iter().enumerate().map(|(i, &b)| b ^ (pattern >> i & 1) as u8).collect();
                    let d = c.decode(&rx).unwrap();
                    total += 1;
                    if d.status == HammingStatus::DoubleErrorDetected {
                        detected += 1;
                    } else if d.data == data {
                        correct += 1;
                    } else {
                        wrong += 1;
                    }
                }
            }
            let pct = |x: usize| format!("{:.0}%", 100.0 * x as f64 / total as f64);
            rows.push(vec![name.to_string(), format!("{w}"), pct(correct), pct(detected), pct(wrong)]);
        }
    }
    write(CHAPTER, "hamming_outcomes.md", &md_table(&["code", "bit errors", "decoded correctly", "detected, not corrected", "decoded wrongly"], &rows));
    // Bit error rates.
    let mut rng = Rng::new(1);
    let mut plot = Plot::new("Hamming codes with hard decisions (BPSK, AWGN)", "Eb/N0 (dB)", "bit error rate");
    plot.series.push(uncoded(7));
    let ebn0: Vec<f64> = (0..=20).map(|k| k as f64 * 0.5).collect();
    for (k, r) in [3u32, 5].into_iter().enumerate() {
        let c = HammingCode::new(r).unwrap();
        let rate = c.k() as f64 / c.n() as f64;
        let pts: Vec<(f64, f64)> = ebn0
            .iter()
            .map(|&e| {
                let ber = simulate(
                    |rng| {
                        let mut errors = 0;
                        for _ in 0..200 {
                            let data = random_bits(c.k(), rng);
                            let cw = c.encode(&data).unwrap();
                            let rx: Vec<u8> = channel(&cw, e, rate, rng).iter().map(|&l| (l < 0.0) as u8).collect();
                            errors += count_errors(&c.decode(&rx).unwrap().data, &data);
                        }
                        (errors, 200 * c.k())
                    },
                    &mut rng,
                    60,
                    4_000_000,
                );
                (e, ber)
            })
            .filter(|p| p.1 > 0.0)
            .collect();
        let s = Series::new(&format!("({}, {}) Hamming", c.n(), c.k()), pts, Style::LineMarkers, k);
        if r == 3 {
            summary.push(uncoded(7));
            summary.push(Series::new("(7, 4) Hamming, hard", s.points.clone(), Style::LineMarkers, 0));
        }
        plot.series.push(s);
    }
    plot.log_y = true;
    plot.y_range = Some((1e-6, 0.5));
    plot.legend = Legend::BottomLeft;
    write(CHAPTER, "hamming_ber.svg", &plot.render());
}

// ------------------------------------------------------------------ Reed-Solomon

fn reed_solomon_figures() {
    let mut rows = Vec::new();
    for line in include_str!("../data/coding_reference.txt").lines() {
        let (nk, parity) = line.split_once('|').unwrap();
        let (n, k) = nk.split_once(' ').unwrap();
        let (n, k): (usize, usize) = (n.parse().unwrap(), k.parse().unwrap());
        let want: Vec<u8> = parity.split(',').map(|v| v.parse().unwrap()).collect();
        let rs = ReedSolomon::new(n, k).unwrap();
        let msg: Vec<u8> = (0..k).map(|i| ((i + 1) % 256) as u8).collect();
        let got = rs.parity(&msg).unwrap();
        // Corrupt t bytes at spread positions and decode.
        let t = rs.error_capacity();
        let cw = rs.encode(&msg).unwrap();
        let mut rx = cw.clone();
        for j in 0..t {
            rx[(j * 37 + 5) % n] ^= (j as u8).wrapping_mul(29).wrapping_add(1);
        }
        let ok = rs.decode(&rx).map(|(m, c)| m == msg && c == t).unwrap_or(false);
        rows.push(vec![format!("({n}, {k})"), format!("{}", n - k), if got == want { "identical".to_string() } else { "DIFFERENT".to_string() }, format!("{t}"), if ok { "all corrected".to_string() } else { "FAILED".to_string() }]);
    }
    write(CHAPTER, "rs_reference.md", &md_table(&["code (n, k)", "parity bytes", "parity vs `reedsolo`", "errors t = (n − k)/2 injected", "decoding"], &rows));
    // Codeword failure against symbol error probability, RS(255, 223).
    let rs = ReedSolomon::new(255, 223).unwrap();
    let mut rng = Rng::new(2);
    let mut plot = Plot::new("RS(255, 223): probability that a codeword is not recovered", "byte error probability", "codeword failure probability");
    let ps: Vec<f64> = (0..=14).map(|k| 0.02 + 0.005 * k as f64).collect();
    let theory: Vec<(f64, f64)> = ps.iter().map(|&p| (p, (17..=255).map(|e| binom_pmf(255, e, p)).sum::<f64>())).collect();
    let (mut miscorrected, mut failures_total) = (0, 0);
    let sim: Vec<(f64, f64)> = ps
        .iter()
        .map(|&p| {
            let trials = 2000;
            let mut fails = 0;
            for _ in 0..trials {
                let msg: Vec<u8> = (0..223).map(|_| (rng.uniform() * 256.0) as u8).collect();
                let cw = rs.encode(&msg).unwrap();
                let rx: Vec<u8> = cw.iter().map(|&b| if rng.uniform() < p { b ^ (1 + (rng.uniform() * 255.0) as u8) } else { b }).collect();
                match rs.decode(&rx) {
                    Ok((m, _)) if m == msg => {}
                    Ok(_) => {
                        fails += 1;
                        miscorrected += 1;
                        failures_total += 1;
                    }
                    Err(_) => {
                        fails += 1;
                        failures_total += 1;
                    }
                }
            }
            (p, fails as f64 / trials as f64)
        })
        .collect();
    plot.series.push(Series::new("simulated", sim.into_iter().filter(|p| p.1 > 0.0).collect(), Style::Markers, 0));
    plot.series.push(Series::new("P(more than 16 byte errors)", theory, Style::Line, 1));
    plot.log_y = true;
    plot.y_range = Some((1e-4, 1.0));
    plot.legend = Legend::BottomRight;
    write(CHAPTER, "rs_failure.svg", &plot.render());
    write(CHAPTER, "rs_facts.md", &format!("Of the {failures_total} simulated codewords that were not recovered, the decoder reported {} as uncorrectable and returned a wrong message for {miscorrected}.", failures_total - miscorrected));
}

fn binom_pmf(n: u64, k: u64, p: f64) -> f64 {
    let ln_c: f64 = (1..=k).map(|i| ((n - k + i) as f64 / i as f64).ln()).sum();
    (ln_c + k as f64 * p.ln() + (n - k) as f64 * (1.0 - p).ln()).exp()
}

// ------------------------------------------------------------------ convolutional

fn convolutional_figures(summary: &mut Vec<Series>) {
    let mut rng = Rng::new(3);
    let mut plot = Plot::new("Viterbi decoding of rate-1/2 convolutional codes", "Eb/N0 (dB)", "bit error rate");
    plot.series.push(uncoded(7));
    let codes = [("K = 3 (7, 5)", ConvolutionalCode::new(3, &[0o7, 0o5]).unwrap()), ("K = 7 (171, 133)", ConvolutionalCode::nasa_k7())];
    let ebn0: Vec<f64> = (0..=16).map(|k| k as f64 * 0.5).collect();
    let mut color = 0;
    for (name, code) in &codes {
        for soft in [false, true] {
            let pts: Vec<(f64, f64)> = ebn0
                .iter()
                .map(|&e| {
                    let ber = simulate(
                        |rng| {
                            let data = random_bits(1000, rng);
                            let cw = code.encode(&data, true);
                            let llr = channel(&cw, e, 0.5, rng);
                            let out = if soft {
                                code.decode_soft(&llr, true).unwrap()
                            } else {
                                code.decode_hard(&llr.iter().map(|&l| (l < 0.0) as u8).collect::<Vec<_>>(), true).unwrap()
                            };
                            (count_errors(&out, &data), data.len())
                        },
                        &mut rng,
                        60,
                        4_000_000,
                    );
                    (e, ber)
                })
                .filter(|p| p.1 > 0.0)
                .collect();
            let label = format!("{name}, {}", if soft { "soft" } else { "hard" });
            if name.starts_with("K = 7") {
                summary.push(Series::new(&label, pts.clone(), Style::LineMarkers, if soft { 2 } else { 1 }));
            }
            plot.series.push(Series::new(&label, pts, if soft { Style::LineMarkers } else { Style::Line }, color));
            color += 1;
        }
    }
    plot.log_y = true;
    plot.y_range = Some((1e-6, 0.5));
    plot.legend = Legend::BottomLeft;
    write(CHAPTER, "viterbi_ber.svg", &plot.render());
    let t0 = Instant::now();
    let code = ConvolutionalCode::nasa_k7();
    let data = random_bits(100_000, &mut rng);
    let llr = channel(&code.encode(&data, true), 4.0, 0.5, &mut rng);
    let _ = code.decode_soft(&llr, true).unwrap();
    write(CHAPTER, "viterbi_facts.md", &format!("Soft Viterbi decoding of the K = 7 code (64 states) ran at {:.2} Mbit/s here.", 0.1 / t0.elapsed().as_secs_f64()));
}

// ------------------------------------------------------------------ LDPC

fn ldpc_figures(summary: &mut Vec<Series>) {
    let code = LdpcCode::gallager(1008, 3, 6, 7).unwrap();
    let (n, k) = (code.n(), code.k());
    let rate = k as f64 / n as f64;
    let mut rng = Rng::new(4);
    let ebn0: Vec<f64> = (0..=12).map(|j| 0.5 + 0.25 * j as f64).collect();
    let mut iters_at: Vec<(f64, f64)> = Vec::new();
    let mut fer: Vec<(f64, f64)> = Vec::new();
    let pts: Vec<(f64, f64)> = ebn0
        .iter()
        .map(|&e| {
            let (mut frames, mut frame_errors, mut iterations) = (0usize, 0usize, 0usize);
            let ber = simulate(
                |rng| {
                    let info = random_bits(k, rng);
                    let cw = code.encode(&info).unwrap();
                    let d = code.decode(&channel(&cw, e, rate, rng), 50).unwrap();
                    let errs = count_errors(&code.extract_info(&d.codeword).unwrap(), &info);
                    frames += 1;
                    iterations += d.iterations;
                    frame_errors += (errs > 0) as usize;
                    (errs, k)
                },
                &mut rng,
                60,
                4000 * k,
            );
            iters_at.push((e, iterations as f64 / frames as f64));
            fer.push((e, frame_errors as f64 / frames as f64));
            (e, ber)
        })
        .collect();
    let mut plot = Plot::new(&format!("Gallager (3, 6) LDPC code, n = {n}, k = {k}: sum-product decoding, 50 iterations"), "Eb/N0 (dB)", "error rate");
    plot.series.push(uncoded(7));
    plot.series.push(Series::new("bit error rate", pts.iter().copied().filter(|p| p.1 > 0.0).collect(), Style::LineMarkers, 0));
    plot.series.push(Series::new("frame error rate", fer.iter().copied().filter(|p| p.1 > 0.0).collect(), Style::LineMarkers, 1));
    plot.log_y = true;
    plot.y_range = Some((1e-6, 1.0));
    plot.x_range = Some((0.0, 5.0));
    plot.legend = Legend::BottomLeft;
    write(CHAPTER, "ldpc_ber.svg", &plot.render());
    summary.push(Series::new(&format!("LDPC ({n}, {k}), sum-product"), pts.into_iter().filter(|p| p.1 > 0.0).collect(), Style::LineMarkers, 3));
    let mut plot = Plot::new("Average sum-product iterations per frame", "Eb/N0 (dB)", "iterations (limit 50)");
    plot.series.push(Series::new("LDPC (1008, k)", iters_at, Style::LineMarkers, 0));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "ldpc_iterations.svg", &plot.render());
    write(CHAPTER, "ldpc_facts.md", &format!("The (3, 6) Gallager code of length {n} has {} checks of rank {}, so k = {k} (rate {rate:.3}); the Shannon limit for rate 1/2 on the BPSK-input AWGN channel is about 0.19 dB.", code.num_checks(), n - k));
}

// ------------------------------------------------------------------ polar

/// Bhattacharyya parameters of the polarized channels, in the library's index order.
fn bhattacharyya(log2_n: u32, e: f64) -> Vec<f64> {
    let mut z = vec![e];
    for _ in 0..log2_n {
        z = z.iter().flat_map(|&t| [2.0 * t - t * t, t * t]).collect();
    }
    z
}

fn polar_figures(summary: &mut Vec<Series>) {
    let z = bhattacharyya(10, 0.5);
    let mut plot = Plot::new("Channel polarization: N = 1024 bit channels from a BEC(0.5)", "bit channel index", "erasure probability (Bhattacharyya parameter)");
    plot.series.push(Series::new("bit channels", z.iter().enumerate().map(|(i, &v)| (i as f64, v)).collect(), Style::Markers, 0));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "polarization.svg", &plot.render());
    let good = z.iter().filter(|&&v| v < 1e-3).count();
    let bad = z.iter().filter(|&&v| v > 1.0 - 1e-3).count();
    write(CHAPTER, "polar_facts.md", &format!("Of the 1024 bit channels, {good} have erasure probability below 0.001 and {bad} above 0.999; the capacity of the BEC(0.5) is 512 bits per 1024 uses."));
    let mut rng = Rng::new(5);
    let ebn0: Vec<f64> = (0..=12).map(|j| 0.5 + 0.25 * j as f64).collect();
    let mut plot = Plot::new("Polar codes, N = 1024, k = 512, successive-cancellation decoding", "Eb/N0 (dB)", "bit error rate");
    plot.series.push(uncoded(7));
    for (c, design) in [0.5, 0.32].into_iter().enumerate() {
        let code = PolarCode::new(10, 512, design).unwrap();
        let pts: Vec<(f64, f64)> = ebn0
            .iter()
            .map(|&e| {
                let ber = simulate(
                    |rng| {
                        let info = random_bits(512, rng);
                        let cw = code.encode(&info).unwrap();
                        (count_errors(&code.decode(&channel(&cw, e, 0.5, rng)).unwrap(), &info), 512)
                    },
                    &mut rng,
                    60,
                    4000 * 512,
                );
                (e, ber)
            })
            .filter(|p| p.1 > 0.0)
            .collect();
        if c == 1 {
            summary.push(Series::new("polar (1024, 512), SC", pts.clone(), Style::LineMarkers, 4));
        }
        plot.series.push(Series::new(&format!("designed for a BEC({design})"), pts, Style::LineMarkers, c));
    }
    plot.log_y = true;
    plot.y_range = Some((1e-6, 0.5));
    plot.x_range = Some((0.0, 5.0));
    plot.legend = Legend::BottomLeft;
    write(CHAPTER, "polar_ber.svg", &plot.render());
}

// ------------------------------------------------------------------ helpers

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
