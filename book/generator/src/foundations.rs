//! Figures and tables for the chapter on shared types, utilities and file formats.

use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_file_io::deflate::{zlib_compress, zlib_decompress};
use lib_jsl_file_io::*;
use lib_jsl_utils::machine_epsilon::{measure_machine_epsilon, ulp};
use lib_jsl_utils::unit_conversion::unit_conversion;
use std::collections::BTreeMap;
use std::path::Path;

const CHAPTER: &str = "foundations";

struct Refs {
    entries: Vec<(String, String, String)>,
}

impl Refs {
    fn load() -> Self {
        let entries = include_str!("../data/foundations_reference.txt")
            .lines()
            .map(|line| {
                let (key, payload) = line.split_once('|').unwrap();
                let (kind, name) = key.split_once(' ').unwrap();
                (kind.to_string(), name.to_string(), payload.to_string())
            })
            .collect();
        Refs { entries }
    }
    fn of<'a>(&'a self, kind: &'a str) -> impl Iterator<Item = (&'a str, &'a str)> + 'a {
        self.entries.iter().filter(move |e| e.0 == kind).map(|e| (e.1.as_str(), e.2.as_str()))
    }
    fn get(&self, kind: &str, name: &str) -> Option<&str> {
        self.entries.iter().find(|e| e.0 == kind && e.1 == name).map(|e| e.2.as_str())
    }
}

fn base64_decode(text: &str) -> Vec<u8> {
    let value = |c: u8| match c {
        b'A'..=b'Z' => c - b'A',
        b'a'..=b'z' => c - b'a' + 26,
        b'0'..=b'9' => c - b'0' + 52,
        b'+' => 62,
        b'/' => 63,
        _ => panic!("not base64"),
    } as u32;
    let bytes: Vec<u8> = text.bytes().filter(|&c| c != b'=').collect();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        let n = chunk.iter().enumerate().fold(0u32, |acc, (i, &c)| acc | value(c) << (18 - 6 * i));
        out.extend_from_slice(&n.to_be_bytes()[1..chunk.len()]);
    }
    out
}

fn floats(text: &str) -> Vec<f64> {
    text.split(',').map(|v| v.parse().unwrap()).collect()
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn psnr(a: &[f64], b: &[f64]) -> f64 {
    let mse = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64;
    10.0 * (255.0 * 255.0 / mse).log10()
}

pub fn generate() {
    let refs = Refs::load();
    constants_table(&refs);
    reading_table(&refs);
    interop_table(&refs);
    jpeg_figure(&refs);
    deflate_table(&refs);
    audio_figure();
    utilities();
}

// ------------------------------------------------------------------ constants

fn constants_table(refs: &Refs) {
    let library: BTreeMap<&str, f64> = library_constants().into_iter().collect();
    let mut groups: BTreeMap<&str, (usize, f64, String)> = BTreeMap::new();
    for (name, value) in refs.of("const") {
        let want: f64 = value.parse().unwrap();
        let got = library[name];
        let rel = ((got - want) / want).abs();
        let module = name.split("::").next().unwrap();
        let entry = groups.entry(module).or_insert((0, 0.0, String::new()));
        entry.0 += 1;
        if rel >= entry.1 {
            entry.1 = rel;
            entry.2 = format!("`{}`", name.split("::").nth(1).unwrap());
        }
    }
    let source = |m: &str| if m == "math" { "mpmath, 40 digits" } else { "SciPy `constants` (CODATA 2022)" };
    let rows: Vec<Vec<String>> = groups.iter().map(|(m, (n, worst, at))| vec![format!("`{m}`"), source(m).into(), format!("{n}"), format!("{worst:.1e} ({at})")]).collect();
    write(CHAPTER, "constants.md", &md_table(&["module", "reference", "constants checked", "largest relative difference"], &rows));
}

// ------------------------------------------------------------------ reading files written by Python

fn decode(name: &str, bytes: &[u8]) -> Result<Vec<f64>, String> {
    let to_f64 = |v: &[u16]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
    if name.starts_with("png") {
        png_decode(bytes).map(|i| to_f64(&i.samples)).map_err(|e| format!("{e:?}"))
    } else if name.starts_with("ppm") || name.starts_with("pgm") || name.starts_with("pbm") {
        ppm_decode(bytes).map(|i| to_f64(&i.samples)).map_err(|e| format!("{e:?}"))
    } else if name.starts_with("jpeg") {
        jpeg_decode(bytes).map(|i| i.pixels.iter().map(|&x| x as f64).collect()).map_err(|e| format!("{e:?}"))
    } else if name.starts_with("wav") {
        parse_wav(bytes).map(|a| a.samples.iter().map(|&x| x as f64).collect()).map_err(|e| format!("{e:?}"))
    } else if name.starts_with("npy") {
        npy_decode(bytes).and_then(|a| a.to_f64()).map(|a| a.iter().copied().collect()).map_err(|e| format!("{e:?}"))
    } else if name.starts_with("zlib") {
        zlib_decompress(bytes).map(|d| d.iter().map(|&x| x as f64).collect()).map_err(|e| format!("{e:?}"))
    } else {
        Err("no decoder".into())
    }
}

fn reading_table(refs: &Refs) {
    let mut rows = Vec::new();
    for (name, payload) in refs.of("file") {
        let bytes = base64_decode(payload);
        let result = if name.starts_with("csv") {
            let delimiter = if name.contains("semicolon") { ';' } else { ',' };
            match parse_csv(&String::from_utf8(bytes.clone()).unwrap(), delimiter, true) {
                Ok(t) => {
                    let ok = t.header == Some(vec!["name".into(), "value".into(), "note".into()]) && t.rows == vec![vec!["a,b".to_string(), "1.5".into(), "say \"hi\"".into()], vec!["two\nlines".into(), "-2e3".into(), "".into()], vec!["plain".into(), "3".into(), "x".into()]];
                    format!("{} records, fields identical: {ok}", t.rows.len() + 1)
                }
                Err(e) => format!("error {e:?}"),
            }
        } else {
            let text: Option<String> = name.starts_with("zlib").then(|| base64_decode(refs.get("text", "zlib").unwrap()).iter().map(|b| b.to_string()).collect::<Vec<_>>().join(","));
            match (decode(name, &bytes), text.as_deref().or(refs.get("expect", name))) {
                (Ok(got), Some(want)) => {
                    let want = floats(want);
                    if got.len() != want.len() {
                        format!("{} values, expected {}", got.len(), want.len())
                    } else if name.starts_with("jpeg") {
                        format!("PSNR {:.1} dB against the original (Pillow: {:.1} dB)", psnr(&got, &want), floats(refs.get("pillowpsnr", name).unwrap())[0])
                    } else {
                        let d = max_diff(&got, &want);
                        if d == 0.0 { "identical".into() } else { format!("largest difference {d:.1e}") }
                    }
                }
                (Ok(_), None) => "decoded".into(),
                (Err(e), _) => format!("refused: {}", e.split('"').nth(1).unwrap_or(&e)),
            }
        };
        rows.push(vec![name.to_string(), format!("{}", bytes.len()), result]);
    }
    for (group, prefixes) in GROUPS {
        let chosen: Vec<Vec<String>> = rows.iter().filter(|r| prefixes.iter().any(|p| r[0].to_lowercase().starts_with(p))).cloned().collect();
        write(CHAPTER, &format!("reading_{group}.md"), &md_table(&["file written by Python", "bytes", "read by the library"], &chosen));
    }
}

/// The file kinds shown on each page, by the lower-case prefixes of their names.
const GROUPS: [(&str, &[&str]); 3] = [("images", &["png", "ppm", "pgm", "pbm", "jpeg", "zlib"]), ("wav", &["wav"]), ("data", &["csv", "npy"])];

fn interop_table(refs: &Refs) {
    let rows: Vec<Vec<String>> = refs.of("interop").map(|(name, result)| vec![name.to_string(), result.to_string()]).collect();
    for (group, prefixes) in GROUPS {
        let chosen: Vec<Vec<String>> = rows.iter().filter(|r| prefixes.iter().any(|p| r[0].to_lowercase().starts_with(p))).cloned().collect();
        write(CHAPTER, &format!("interop_{group}.md"), &md_table(&["file written by the library (read by)", "result"], &chosen));
    }
}

/// Writes files with the library's encoders for the reference script to read back.
pub fn write_interop_files(dir: &Path) {
    let save = |name: &str, shape: Vec<usize>, data: Vec<f64>| {
        let a = NpyArray::new(shape, NpyData::F64(data)).unwrap();
        std::fs::write(dir.join(format!("{name}.npy")), npy_encode(&a).unwrap()).unwrap();
    };
    let (w, h) = (40usize, 30usize);
    let px = |x: usize, y: usize, c: usize| ((x * 6 + y * 5 + c * 70 + (x * y) % 23) % 256) as u16;
    for ch in [1usize, 2, 3, 4] {
        let s: Vec<u16> = (0..h).flat_map(|y| (0..w).flat_map(move |x| (0..ch).map(move |c| px(x, y, c)))).collect();
        let image = RasterImage::new(w, h, ch, 8, s.clone()).unwrap();
        std::fs::write(dir.join(format!("image_{ch}.png")), png_encode(&image, 6).unwrap()).unwrap();
        save(&format!("image_{ch}"), vec![h, w, ch], s.iter().map(|&v| v as f64).collect());
        if ch == 1 || ch == 3 {
            std::fs::write(dir.join(format!("image_{ch}.ppm")), ppm_encode(&image).unwrap()).unwrap();
            let bytes: Vec<u8> = s.iter().map(|&v| v as u8).collect();
            for q in [50u8, 90] {
                std::fs::write(dir.join(format!("image_{ch}_q{q}.jpg")), jpeg_encode(&bytes, w, h, ch, q).unwrap()).unwrap();
            }
        }
    }
    let s16: Vec<u16> = (0..h * w).map(|i| (i * 2003 % 65536) as u16).collect();
    std::fs::write(dir.join("image_16.png"), png_encode(&RasterImage::new(w, h, 1, 16, s16.clone()).unwrap(), 9).unwrap()).unwrap();
    save("image_16", vec![h, w, 1], s16.iter().map(|&v| v as f64).collect());
    let samples: Vec<f32> = (0..1200).map(|i| 0.9 * ((i / 2) as f32 * 0.05).sin() * if i % 2 == 0 { 1.0 } else { -0.5 }).collect();
    save("audio", vec![600, 2], samples.iter().map(|&v| v as f64).collect());
    let audio = WavAudio { sample_rate: 22050, channels: 2, samples };
    for (name, format) in [("8-bit", WavSampleFormat::Pcm8), ("16-bit", WavSampleFormat::Pcm16), ("24-bit", WavSampleFormat::Pcm24), ("32-bit", WavSampleFormat::Pcm32), ("float", WavSampleFormat::Float32), ("double", WavSampleFormat::Float64)] {
        std::fs::write(dir.join(format!("audio_{name}.wav")), encode_wav(&audio, format).unwrap()).unwrap();
    }
    let table = CsvTable { header: Some(vec!["a".into(), "b,c".into()]), rows: vec![vec!["x \"q\"".into(), "1\n2".into()], vec!["".into(), "plain".into()]] };
    std::fs::write(dir.join("table.csv"), format_csv(&table, ',')).unwrap();
    let values: Vec<f64> = (0..12).map(|i| (i as f64 * 0.7).sin()).collect();
    save("npy_float64", vec![3, 4], values.clone());
    std::fs::write(dir.join("array_float64.npy"), npy_encode(&NpyArray::new(vec![3, 4], NpyData::F64(values)).unwrap()).unwrap()).unwrap();
    std::fs::write(dir.join("array_int32.npy"), npy_encode(&NpyArray::new(vec![2, 3], NpyData::I32(vec![-1, 2, 3, 4, 5, 6])).unwrap()).unwrap()).unwrap();
    std::fs::write(dir.join("array_complex128.npy"), npy_encode(&NpyArray::new(vec![2], NpyData::C128(vec![num_complex::Complex::new(1.0, -2.0), num_complex::Complex::new(3.5, 0.0)])).unwrap()).unwrap()).unwrap();
    std::fs::write(dir.join("array_bool.npy"), npy_encode(&NpyArray::new(vec![2], NpyData::Bool(vec![true, false])).unwrap()).unwrap()).unwrap();
    let text: Vec<u8> = (0..60).flat_map(|i| format!("line {i}: the quick brown fox jumps over the lazy dog\n").into_bytes()).collect();
    std::fs::write(dir.join("text.bin"), &text).unwrap();
    for level in [0u8, 6, 9] {
        std::fs::write(dir.join(format!("zlib_{level}.zlib")), zlib_compress(&text, level)).unwrap();
    }
}

// ------------------------------------------------------------------ JPEG rate and distortion

fn jpeg_figure(refs: &Refs) {
    let photo = base64_decode(refs.get("image", "astronaut").unwrap());
    let original: Vec<f64> = photo.iter().map(|&v| v as f64).collect();
    let pixels = (128 * 128) as f64;
    let mut ours = Vec::new();
    let mut rows = Vec::new();
    for q in [10u8, 20, 30, 40, 50, 60, 70, 80, 90, 95] {
        let bytes = jpeg_encode(&photo, 128, 128, 3, q).unwrap();
        let decoded: Vec<f64> = jpeg_decode(&bytes).unwrap().pixels.iter().map(|&v| v as f64).collect();
        let p = psnr(&decoded, &original);
        ours.push((8.0 * bytes.len() as f64 / pixels, p));
        if [20u8, 50, 90].contains(&q) {
            let pil = floats(refs.get("jpeg", &format!("4:4:4 {q}")).unwrap());
            rows.push(vec![format!("{q}"), format!("{} bytes, {p:.2} dB", bytes.len()), format!("{} bytes, {:.2} dB", pil[0], pil[1])]);
        }
    }
    let pillow = |label: &str| -> Vec<(f64, f64)> {
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 95].iter().map(|q| { let v = floats(refs.get("jpeg", &format!("{label} {q}")).unwrap()); (8.0 * v[0] / pixels, v[1]) }).collect()
    };
    let mut plot = Plot::new("JPEG rate and distortion, 128 × 128 colour photograph", "bits per pixel", "PSNR (dB)");
    plot.series.push(Series::new("library, 4:4:4", ours, Style::LineMarkers, 0));
    plot.series.push(Series::new("Pillow (libjpeg), 4:4:4", pillow("4:4:4"), Style::Dashed, 1));
    plot.series.push(Series::new("Pillow (libjpeg), 4:2:0", pillow("4:2:0"), Style::Dashed, 2));
    plot.legend = Legend::BottomRight;
    write(CHAPTER, "jpeg_rate.svg", &plot.render());
    write(CHAPTER, "jpeg_rate.md", &md_table(&["quality", "library (4:4:4)", "Pillow (4:4:4)"], &rows));
}

fn deflate_table(refs: &Refs) {
    let mut rows = Vec::new();
    let text = base64_decode(refs.get("text", "zlib").unwrap());
    let inputs = [("photograph pixels", base64_decode(refs.get("image", "astronaut").unwrap())), ("English text", text[..4500].to_vec()), ("random bytes", base64_decode(refs.get("zlibdata", "random bytes").unwrap()))];
    for (name, data) in inputs {
        let zl = floats(refs.get("zlib", name).unwrap());
        let (a, b) = (zlib_compress(&data, 6), zlib_compress(&data, 9));
        assert_eq!(zlib_decompress(&b).unwrap(), data);
        rows.push(vec![name.to_string(), format!("{}", data.len()), format!("{} / {}", a.len(), b.len()), format!("{} / {}", zl[0], zl[1])]);
    }
    write(CHAPTER, "deflate.md", &md_table(&["data", "bytes", "library, level 6 / 9", "zlib, level 6 / 9"], &rows));
}

// ------------------------------------------------------------------ audio

fn audio_figure() {
    let rate = 44100u32;
    let n = rate as usize;
    let mut s: u64 = 5;
    let mut noise = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let tau = 2.0 * std::f64::consts::PI;
    let signals: Vec<(&str, Vec<i16>)> = vec![
        ("440 Hz tone", (0..n).map(|i| (12000.0 * (tau * 440.0 * i as f64 / rate as f64).sin()) as i16).collect()),
        ("three-note chord", (0..n).map(|i| { let t = i as f64 / rate as f64; (6000.0 * ((tau * 261.6 * t).sin() + (tau * 329.6 * t).sin() + (tau * 392.0 * t).sin())) as i16 }).collect()),
        ("white noise", (0..n).map(|_| (16000.0 * noise()) as i16).collect()),
    ];
    let mut plot = Plot::new("The MP3-style codec: quality against bit rate (mono, 44.1 kHz)", "bit rate (kbit/s)", "SNR (dB)");
    for (k, (name, pcm)) in signals.iter().enumerate() {
        let mut pts = Vec::new();
        for kbps in [16u32, 32, 48, 64, 96, 128, 192, 256] {
            let encoded = mp3_encode(pcm, rate, 1, kbps).unwrap();
            let decoded = mp3_decode(&encoded).unwrap().samples;
            let (sig, err) = pcm.iter().zip(&decoded).fold((0.0, 0.0), |(a, b), (&x, &y)| (a + (x as f64).powi(2), b + (x as f64 - y as f64).powi(2)));
            pts.push((encoded.len() as f64 * 8.0 / 1000.0, 10.0 * (sig / err).log10()));
        }
        plot.series.push(Series::new(name, pts, Style::LineMarkers, k));
    }
    plot.legend = Legend::TopLeft;
    write(CHAPTER, "audio_rate.svg", &plot.render());
}

// ------------------------------------------------------------------ utilities

fn utilities() {
    let rows: Vec<Vec<String>> = [
        (1.0, "mi", "km", "1.609344 (exact)"),
        (100.0, "km/h", "mph", "62.137119223733…"),
        (1.0, "atm", "psi", "14.695948775513…"),
        (98.6, "F", "C", "37"),
        (1.0, "GiB", "MB", "1073.741824"),
        (1.0, "mW", "W", "0.001"),
        (1.0, "MW", "W", "1 000 000"),
        (760.0, "torr", "atm", "1 (exact)"),
    ]
    .iter()
    .map(|&(v, a, b, want)| vec![format!("{v} {a} in {b}"), format!("{}", unit_conversion(v, a, b).unwrap()), want.to_string()])
    .collect();
    let mut rows = rows;
    let refused = unit_conversion(1.0, "mw", "W").map(|v| v.to_string()).unwrap_or_else(|e| {
        let message = format!("{e:?}");
        format!("error: {}", message.split('"').nth(1).unwrap_or(""))
    });
    rows.push(vec!["1 mw in W".into(), refused, "ambiguous: milli- or mega-".into()]);
    write(CHAPTER, "units.md", &md_table(&["conversion", "library", "exact"], &rows));

    let mut plot = Plot::new("Spacing of f64 numbers", "x", "ulp(x) and ulp(x) / x");
    plot.log_x = true;
    plot.log_y = true;
    let xs: Vec<f64> = (0..=400).map(|k| 10f64.powf(-6.0 + 12.0 * k as f64 / 400.0)).collect();
    plot.series.push(Series::new("ulp(x)", xs.iter().map(|&x| (x, ulp(x))).collect(), Style::Line, 0));
    plot.series.push(Series::new("ulp(x) / x", xs.iter().map(|&x| (x, ulp(x) / x)).collect(), Style::Line, 1));
    plot.series.push(Series::new("machine epsilon 2⁻⁵²", vec![(1e-6, f64::EPSILON), (1e6, f64::EPSILON)], Style::Dashed, 6));
    plot.legend = Legend::TopLeft;
    write(CHAPTER, "ulp.svg", &plot.render());
    write(CHAPTER, "epsilon.md", &format!("Measured by halving until $1 + \\varepsilon/2$ rounds to 1, the machine epsilon is {:e}, equal to `f64::EPSILON` = $2^{{-52}}$: {}.", measure_machine_epsilon(), measure_machine_epsilon() == f64::EPSILON));
}
/// The library constants checked against the references, by name.
fn library_constants() -> Vec<(&'static str, f64)> {
    use lib_jsl_core::constants::*;
    vec![
        ("math::TWO_PI_OVER_3", math::TWO_PI_OVER_3),
        ("math::THREE_PI_OVER_2", math::THREE_PI_OVER_2),
        ("math::PI_SQUARED", math::PI_SQUARED),
        ("math::SQRT_PI", math::SQRT_PI),
        ("math::SQRT_TAU", math::SQRT_TAU),
        ("math::FRAC_1_SQRT_TAU", math::FRAC_1_SQRT_TAU),
        ("math::FRAC_1_SQRT_PI", math::FRAC_1_SQRT_PI),
        ("math::LN_PI", math::LN_PI),
        ("math::LN_SQRT_TAU", math::LN_SQRT_TAU),
        ("math::LN_TAU", math::LN_TAU),
        ("math::FOUR_PI", math::FOUR_PI),
        ("math::SQRT_3", math::SQRT_3),
        ("math::SQRT_5", math::SQRT_5),
        ("math::SQRT_6", math::SQRT_6),
        ("math::CBRT_2", math::CBRT_2),
        ("math::CBRT_3", math::CBRT_3),
        ("math::PHI", math::PHI),
        ("math::FRAC_1_PHI", math::FRAC_1_PHI),
        ("math::LN_PHI", math::LN_PHI),
        ("math::SILVER_RATIO", math::SILVER_RATIO),
        ("math::PLASTIC_NUMBER", math::PLASTIC_NUMBER),
        ("math::EULER_GAMMA", math::EULER_GAMMA),
        ("math::APERY", math::APERY),
        ("math::ZETA_2", math::ZETA_2),
        ("math::ZETA_4", math::ZETA_4),
        ("math::CATALAN", math::CATALAN),
        ("math::GLAISHER", math::GLAISHER),
        ("math::KHINCHIN", math::KHINCHIN),
        ("math::OMEGA", math::OMEGA),
        ("math::TWIN_PRIME", math::TWIN_PRIME),
        ("math::GIBBS", math::GIBBS),
        ("math::FEIGENBAUM_DELTA", math::FEIGENBAUM_DELTA),
        ("math::FEIGENBAUM_ALPHA", math::FEIGENBAUM_ALPHA),
        ("math::SIGMA_TO_FWHM", math::SIGMA_TO_FWHM),
        ("math::FWHM_TO_SIGMA", math::FWHM_TO_SIGMA),
        ("math::Z_90", math::Z_90),
        ("math::Z_95", math::Z_95),
        ("math::Z_99", math::Z_99),
        ("math::Z_999", math::Z_999),
        ("math::P_1_SIGMA", math::P_1_SIGMA),
        ("math::P_2_SIGMA", math::P_2_SIGMA),
        ("math::P_3_SIGMA", math::P_3_SIGMA),
        ("physics::HBAR", physics::HBAR),
        ("physics::R_GAS", physics::R_GAS),
        ("physics::FARADAY", physics::FARADAY),
        ("physics::SIGMA_SB", physics::SIGMA_SB),
        ("physics::C1_RADIATION", physics::C1_RADIATION),
        ("physics::C2_RADIATION", physics::C2_RADIATION),
        ("physics::G", physics::G),
        ("physics::ALPHA", physics::ALPHA),
        ("physics::INV_ALPHA", physics::INV_ALPHA),
        ("physics::M_E", physics::M_E),
        ("physics::M_P", physics::M_P),
        ("physics::M_N", physics::M_N),
        ("physics::AMU", physics::AMU),
        ("physics::MP_OVER_ME", physics::MP_OVER_ME),
        ("physics::WIEN_B", physics::WIEN_B),
        ("physics::WIEN_B_FREQ", physics::WIEN_B_FREQ),
        ("physics::BOHR_RADIUS", physics::BOHR_RADIUS),
        ("physics::COMPTON_WAVELENGTH", physics::COMPTON_WAVELENGTH),
        ("physics::CLASSICAL_ELECTRON_RADIUS", physics::CLASSICAL_ELECTRON_RADIUS),
        ("physics::THOMSON_CROSS_SECTION", physics::THOMSON_CROSS_SECTION),
        ("physics::RYDBERG", physics::RYDBERG),
        ("physics::HARTREE", physics::HARTREE),
        ("physics::PLANCK_LENGTH", physics::PLANCK_LENGTH),
        ("physics::PLANCK_MASS", physics::PLANCK_MASS),
        ("physics::PLANCK_TIME", physics::PLANCK_TIME),
        ("physics::PLANCK_TEMPERATURE", physics::PLANCK_TEMPERATURE),
        ("physics::MOLAR_VOLUME_STP", physics::MOLAR_VOLUME_STP),
        ("physics::PARSEC", physics::PARSEC),
        ("em::MU_0", em::MU_0),
        ("em::EPS_0", em::EPS_0),
        ("em::Z_0", em::Z_0),
        ("em::FLUX_QUANTUM", em::FLUX_QUANTUM),
        ("em::JOSEPHSON", em::JOSEPHSON),
        ("em::VON_KLITZING", em::VON_KLITZING),
        ("em::CONDUCTANCE_QUANTUM", em::CONDUCTANCE_QUANTUM),
        ("em::BOHR_MAGNETON", em::BOHR_MAGNETON),
        ("em::NUCLEAR_MAGNETON", em::NUCLEAR_MAGNETON),
        ("em::G_ELECTRON", em::G_ELECTRON),
        ("units::BTU", units::BTU),
        ("units::CALORIE", units::CALORIE),
        ("units::NAUTICAL_MILE", units::NAUTICAL_MILE),
        ("units::PSI", units::PSI),
        ("units::HORSEPOWER", units::HORSEPOWER),
        ("units::LBF", units::LBF),
        ("units::OUNCE", units::OUNCE),
        ("units::GALLON_US", units::GALLON_US),
    ]
}
