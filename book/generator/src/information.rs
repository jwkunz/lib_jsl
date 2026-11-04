//! Figures and tables for the chapter on information theory: entropy, source coding, LFSRs,
//! checksums and hashes.

use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_information::lfsr::{Lfsr, all_maximum_length_polynomials, gold_codes, gold_correlation_bound, maximum_length_polynomial};
use lib_jsl_information::operators::{crc32, crc32c, entropy, entropy_from_probabilities, kl_divergence, mutual_information, sha256, to_hex};
use lib_jsl_information::source_coders::gray_codes::gray_code;
use lib_jsl_information::{arithmetic, huffman, lz77, lz78, lzw, shannon_fano};
use lib_jsl_random::uniform_generator::UniformGenerator;
use lib_jsl_random::xoshiro256plusplus::Xoshiro256PlusPlus;
use std::collections::HashMap;
use std::time::Instant;

const CHAPTER: &str = "information";
const KJV: &[u8] = include_bytes!("../../../crates/information/test_assets/kjv.txt");

pub fn generate() {
    let refs: HashMap<&str, &str> = include_str!("../data/information_reference.txt").lines().map(|l| l.split_once('|').unwrap()).collect();
    entropy_figures(&refs);
    entropy_coding_figures(&refs);
    dictionary_figures();
    lfsr_figures();
    checksum_figures(&refs);
}

type Coder = (&'static str, fn(&[u8]) -> Vec<u8>, fn(&[u8]) -> Vec<u8>);

const CODERS: [Coder; 6] = [
    ("Huffman", huffman::huffman_compress, huffman::huffman_decompress),
    ("Shannon-Fano", shannon_fano::shannon_fano_compress, shannon_fano::shannon_fano_decompress),
    ("arithmetic", arithmetic::arithmetic_compress, arithmetic::arithmetic_decompress),
    ("LZ77", lz77::lz77_compress, lz77::lz77_decompress),
    ("LZ78", lz78::lz78_compress, lz78::lz78_decompress),
    ("LZW", lzw::lzw_compress, lzw::lzw_decompress),
];

/// Bytes of header before the coded bits of the static entropy coders: magic, length, symbol
/// count, and five bytes per symbol (plus a bit count for the arithmetic coder).
fn model_header(name: &str, data: &[u8]) -> usize {
    let symbols = (0..256).filter(|&b| data.contains(&(b as u8))).count();
    4 + 8 + 2 + 5 * symbols + if name == "arithmetic" { 8 } else { 0 }
}

// ------------------------------------------------------------------ entropy

fn entropy_figures(refs: &HashMap<&str, &str>) {
    let h = |p: f64| entropy_from_probabilities(&[p, 1.0 - p]);
    let mut plot = Plot::new("The binary entropy function", "probability of a 1", "entropy (bits)");
    plot.series.push(Series::new("H(p)", (0..=200).map(|k| k as f64 / 200.0).map(|p| (p, h(p))).collect(), Style::Line, 0));
    plot.legend = Legend::BottomRight;
    write(CHAPTER, "binary_entropy.svg", &plot.render());
    let p = [0.1, 0.2, 0.3, 0.4];
    let q = [0.25; 4];
    let parse = |k: &str| refs[k].parse::<f64>().unwrap();
    let mut rng = Rng::new(1);
    let random: Vec<u8> = (0..1_000_000).map(|_| (rng.uniform() * 256.0) as u8).collect();
    let rows = vec![
        vec!["bytes of the King James Bible (4.6 MB)".to_string(), format!("{:.12}", entropy(KJV)), format!("{:.12}", parse("entropy kjv"))],
        vec!["$D(p \\Vert q)$, $p = (0.1, 0.2, 0.3, 0.4)$, $q$ uniform".to_string(), format!("{:.12}", kl_divergence(&p, &q)), format!("{:.12}", parse("kl p q"))],
        vec!["$D(q \\Vert p)$".to_string(), format!("{:.12}", kl_divergence(&q, &p)), format!("{:.12}", parse("kl q p"))],
    ];
    write(CHAPTER, "entropy_reference.md", &md_table(&["quantity (bits)", "library", "SciPy `stats.entropy`"], &rows));
    // Entropy of bytes, pairs and the conditional entropy of the next letter.
    let pairs = block_entropy(KJV, 2);
    let triples = block_entropy(KJV, 3);
    let rows = vec![
        vec!["1 000 000 uniformly random bytes".to_string(), format!("{:.4}", entropy(&random))],
        vec!["the text, one byte at a time".to_string(), format!("{:.4}", entropy(KJV))],
        vec!["the text, pairs of bytes, per byte".to_string(), format!("{:.4}", pairs / 2.0)],
        vec!["the text, triples of bytes, per byte".to_string(), format!("{:.4}", triples / 3.0)],
        vec!["the text: next byte given the previous two, $H_3 - H_2$".to_string(), format!("{:.4}", triples - pairs)],
    ];
    write(CHAPTER, "entropy_table.md", &md_table(&["source", "entropy (bits per byte)"], &rows));
    // Mutual information of a binary symmetric channel with uniform input.
    let mut plot = Plot::new("Mutual information across a binary symmetric channel", "crossover probability", "bits per use");
    let pts: Vec<(f64, f64)> = (0..=100).map(|k| k as f64 / 100.0).map(|e| (e, mutual_information(&[0.5 * (1.0 - e), 0.5 * e, 0.5 * e, 0.5 * (1.0 - e)], 2, 2))).collect();
    plot.series.push(Series::new("I(X; Y) = 1 − H(ε)", pts, Style::Line, 1));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "bsc_information.svg", &plot.render());
}

/// The entropy in bits of the distribution of overlapping `k`-byte blocks.
fn block_entropy(data: &[u8], k: usize) -> f64 {
    let mut counts: HashMap<&[u8], u64> = HashMap::new();
    for w in data.windows(k) {
        *counts.entry(w).or_default() += 1;
    }
    let n = (data.len() - k + 1) as f64;
    counts.values().map(|&c| c as f64 / n).map(|p| -p * p.log2()).sum()
}

// ------------------------------------------------------------------ entropy coding

fn entropy_coding_figures(refs: &HashMap<&str, &str>) {
    let mut rng = Rng::new(7);
    let n = 200_000;
    // Sources with known entropy.
    let sources: Vec<(&str, Vec<u8>, f64)> = vec![
        ("uniform over 256 values", (0..n).map(|_| (rng.uniform() * 256.0) as u8).collect(), 8.0),
        ("dyadic: 1/2, 1/4, 1/8, …", (0..n).map(|_| (rng.uniform().log2().abs().floor() as u8).min(15)).collect(), f64::NAN),
        ("binary, P(1) = 0.1", (0..n).map(|_| (rng.uniform() < 0.1) as u8).collect(), entropy_from_probabilities(&[0.9, 0.1])),
        ("binary, P(1) = 0.01", (0..n).map(|_| (rng.uniform() < 0.01) as u8).collect(), entropy_from_probabilities(&[0.99, 0.01])),
        ("geometric over 64 values, ratio 0.8", (0..n).map(|_| ((1.0 - rng.uniform()).ln() / 0.8f64.ln()).floor().min(63.0) as u8).collect(), f64::NAN),
    ];
    let mut rows = Vec::new();
    for (name, data, _) in &sources {
        let h = entropy(data);
        let mut row = vec![name.to_string(), format!("{h:.4}")];
        for coder in &CODERS[..3] {
            let c = (coder.1)(data);
            assert_eq!((coder.2)(&c), *data, "{} round trip", coder.0);
            row.push(format!("{:.4}", 8.0 * (c.len() - model_header(coder.0, data)) as f64 / data.len() as f64));
        }
        rows.push(row);
    }
    write(CHAPTER, "entropy_coders_table.md", &md_table(&["source (200 000 symbols)", "empirical entropy", "Huffman", "Shannon-Fano", "arithmetic"], &rows));
    // The whole text with every coder, against standard compressors.
    let mut rows = Vec::new();
    for coder in &CODERS {
        let t0 = Instant::now();
        let c = (coder.1)(KJV);
        let t_c = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let d = (coder.2)(&c);
        let t_d = t0.elapsed().as_secs_f64();
        assert_eq!(d, KJV, "{} round trip", coder.0);
        rows.push(vec![coder.0.to_string(), format!("{}", c.len()), format!("{:.3}", 8.0 * c.len() as f64 / KJV.len() as f64), format!("{:.0}", KJV.len() as f64 / 1e6 / t_c), format!("{:.0}", KJV.len() as f64 / 1e6 / t_d)]);
    }
    for (name, key) in [("zlib, level 9 (Python)", "zlib kjv"), ("bzip2, level 9 (Python)", "bz2 kjv"), ("xz/LZMA, preset 9 (Python)", "lzma kjv")] {
        let len: f64 = refs[key].parse().unwrap();
        rows.push(vec![name.to_string(), format!("{len}"), format!("{:.3}", 8.0 * len / KJV.len() as f64), "".to_string(), "".to_string()]);
    }
    write(CHAPTER, "kjv_table.md", &md_table(&["coder", "compressed bytes", "bits per byte", "compress (MB/s)", "decompress (MB/s)"], &rows));
    write(CHAPTER, "kjv_facts.md", &format!("The text has {} bytes and an order-0 entropy of {:.3} bits per byte.", KJV.len(), entropy(KJV)));
}

// ------------------------------------------------------------------ dictionary coding

fn dictionary_figures() {
    let lengths: Vec<usize> = (0..=14).map(|k| (1000.0 * 2f64.powf(k as f64 * 12.0 / 14.0)) as usize).map(|l| l.min(KJV.len())).collect();
    let mut plot = Plot::new("Compression of the first n bytes of the King James Bible", "n (bytes)", "bits per byte");
    for (k, coder) in CODERS.iter().enumerate() {
        let pts: Vec<(f64, f64)> = lengths.iter().map(|&l| (l as f64, 8.0 * (coder.1)(&KJV[..l]).len() as f64 / l as f64)).collect();
        plot.series.push(Series::new(coder.0, pts, if k < 3 { Style::Dashed } else { Style::LineMarkers }, k));
    }
    plot.log_x = true;
    plot.y_range = Some((0.0, 9.0));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "dictionary_growth.svg", &plot.render());
    // Different kinds of data.
    let mut rng = Rng::new(3);
    let n = 100_000;
    let data: Vec<(&str, Vec<u8>)> = vec![
        ("text (first 100 kB of the Bible)", KJV[..n].to_vec()),
        ("random bytes", (0..n).map(|_| (rng.uniform() * 256.0) as u8).collect()),
        ("random letters A, C, G, T", (0..n).map(|_| b"ACGT"[(rng.uniform() * 4.0) as usize]).collect()),
        ("one byte repeated", vec![b'a'; n]),
        ("a 100-byte random block repeated", { let block: Vec<u8> = (0..100).map(|_| (rng.uniform() * 256.0) as u8).collect(); block.iter().cycle().take(n).copied().collect() }),
    ];
    let mut rows = Vec::new();
    for (name, d) in &data {
        let mut row = vec![name.to_string()];
        for coder in &CODERS {
            row.push(format!("{:.3}", 8.0 * (coder.1)(d).len() as f64 / d.len() as f64));
        }
        rows.push(row);
    }
    let mut header = vec!["data (100 000 bytes), bits per byte"];
    header.extend(CODERS.iter().map(|c| c.0));
    write(CHAPTER, "data_kinds_table.md", &md_table(&header, &rows));
}

// ------------------------------------------------------------------ LFSRs

fn euler_phi(mut n: u64) -> u64 {
    let mut result = n;
    let mut p = 2;
    while p * p <= n {
        if n.is_multiple_of(p) {
            while n.is_multiple_of(p) {
                n /= p;
            }
            result -= result / p;
        }
        p += 1;
    }
    if n > 1 {
        result -= result / n;
    }
    result
}

fn poly_string(p: u64) -> String {
    let n = 63 - p.leading_zeros();
    (0..=n)
        .rev()
        .filter(|&k| p >> k & 1 == 1)
        .map(|k| match k {
            0 => "1".to_string(),
            1 => "x".to_string(),
            _ => format!("x^{{{k}}}"),
        })
        .collect::<Vec<_>>()
        .join(" + ")
}

fn lfsr_figures() {
    // Counts of primitive polynomials against φ(2^n − 1)/n, and the library's chosen polynomial.
    let mut rows = Vec::new();
    for n in [2u32, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 31, 32] {
        let formula = euler_phi((1u64 << n) - 1) / n as u64;
        let count = if n <= 20 { format!("{}", all_maximum_length_polynomials(n).len()) } else { "".to_string() };
        rows.push(vec![format!("{n}"), format!("{}", (1u64 << n) - 1), count, format!("{formula}"), format!("${}$", poly_string(maximum_length_polynomial(n).unwrap()))]);
    }
    write(CHAPTER, "primitive_table.md", &md_table(&["degree n", "period 2ⁿ − 1", "primitive polynomials found", "φ(2ⁿ − 1)/n", "minimal-weight choice"], &rows));
    // Autocorrelation of an m-sequence and cross-correlations of Gold codes, n = 7.
    let n = 7;
    let period = 127;
    let chips = |bits: &[u8]| bits.iter().map(|&b| 1 - 2 * b as i64).collect::<Vec<i64>>();
    let corr = |a: &[i64], b: &[i64], s: usize| (0..a.len()).map(|t| a[t] * b[(t + s) % a.len()]).sum::<i64>();
    let m = chips(&Lfsr::from_polynomial(maximum_length_polynomial(n).unwrap(), 1).bits(period));
    let mut plot = Plot::new("Periodic autocorrelation of an m-sequence of period 127", "shift", "correlation");
    plot.series.push(Series::new("m-sequence", (0..period).map(|s| (s as f64, corr(&m, &m, s) as f64)).collect(), Style::Stems, 0));
    plot.y_range = Some((-10.0, 135.0));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "msequence_autocorrelation.svg", &plot.render());
    let family: Vec<Vec<i64>> = gold_codes(n).unwrap().iter().map(|s| chips(s)).collect();
    let mut values: HashMap<i64, u64> = HashMap::new();
    for a in 0..12 {
        for b in (a + 1)..family.len() {
            for s in 0..period {
                *values.entry(corr(&family[a], &family[b], s)).or_default() += 1;
            }
        }
    }
    // Cross-correlation of two m-sequences that are not a preferred pair.
    let all = all_maximum_length_polynomials(n);
    let (worst_other, other_poly) = all
        .iter()
        .map(|&p| {
            let other = chips(&Lfsr::from_polynomial(p, 1).bits(period));
            ((0..period).map(|s| corr(&m, &other, s).abs()).max().unwrap(), p)
        })
        .filter(|&(_, p)| p != maximum_length_polynomial(n).unwrap())
        .max()
        .unwrap();
    let mut keys: Vec<_> = values.keys().copied().collect();
    keys.sort();
    let t = gold_correlation_bound(n).unwrap();
    write(
        CHAPTER,
        "gold_facts.md",
        &format!(
            "Over all shifts between the first 12 codes and every other member of the family for $n = 7$, the cross-correlation takes only the values {} (the theory: $-1$, $-t(7) = -{t}$ and $t(7) - 2 = {}$), with counts {}. Pairing the first m-sequence with another of the same length, chosen badly (${}$ with ${}$), gives cross-correlations as large as {worst_other}.",
            keys.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(", "),
            t - 2,
            keys.iter().map(|k| values[k].to_string()).collect::<Vec<_>>().join(", "),
            poly_string(maximum_length_polynomial(n).unwrap()),
            poly_string(other_poly)
        ),
    );
    // Gray code rotary encoder picture: bits per position.
    let codes: Vec<String> = (0..8).map(|k| format!("{k} → {:03b}", gray_code(k))).collect();
    write(CHAPTER, "gray_facts.md", &format!("The 3-bit Gray code: {}.", codes.join(", ")));
}

// ------------------------------------------------------------------ checksums

fn checksum_figures(refs: &HashMap<&str, &str>) {
    let mut rows = Vec::new();
    for (label, data) in [("empty", &b""[..]), ("abc", &b"abc"[..]), ("check", &b"123456789"[..]), ("fox", &b"The quick brown fox jumps over the lazy dog"[..]), ("kjv", KJV)] {
        let shown = match label {
            "empty" => "\"\"".to_string(),
            "kjv" => "the Bible test text (4.6 MB)".to_string(),
            _ => format!("\"{}\"", std::str::from_utf8(data).unwrap()),
        };
        let want_crc: u32 = refs[format!("crc32 {label}").as_str()].parse().unwrap();
        let got_crc = crc32(data);
        let want_sha = refs[format!("sha256 {label}").as_str()];
        let got_sha = to_hex(&sha256(data));
        rows.push(vec![shown, format!("`{got_crc:08x}`"), if got_crc == want_crc { "equal" } else { "DIFFERENT" }.to_string(), format!("`{}…`", &got_sha[..16]), if got_sha == want_sha { "equal" } else { "DIFFERENT" }.to_string()]);
    }
    let million = vec![b'a'; 1_000_000];
    let got = to_hex(&sha256(&million));
    rows.push(vec!["a million \"a\"s".to_string(), "".to_string(), "".to_string(), format!("`{}…`", &got[..16]), if got == refs["sha256 million"] { "equal" } else { "DIFFERENT" }.to_string()]);
    write(CHAPTER, "checksum_reference.md", &md_table(&["input", "CRC-32", "vs `zlib.crc32`", "SHA-256 (first 16 hex digits)", "vs `hashlib.sha256`"], &rows));
    write(CHAPTER, "crc32c_facts.md", &format!("CRC-32C of \"123456789\" is `{:08x}`, the catalogued check value `e3069283`.", crc32c(b"123456789")));
    // Error detection by CRC-32.
    let mut rng = Rng::new(9);
    let msg: Vec<u8> = (0..1024).map(|_| (rng.uniform() * 256.0) as u8).collect();
    let good = crc32(&msg);
    let trials = 100_000;
    let mut rows = Vec::new();
    for (name, kind) in [("1 bit flipped", 0), ("2 bits flipped", 1), ("3 bits flipped", 2), ("a burst of ≤ 32 bits", 3), ("a burst of 33 to 64 bits", 4), ("8 random bytes replaced", 5)] {
        let mut missed = 0;
        for _ in 0..trials {
            let mut m = msg.clone();
            let bits = m.len() * 8;
            let flip = |m: &mut Vec<u8>, b: usize| m[b / 8] ^= 1 << (b % 8);
            match kind {
                0..=2 => {
                    let mut chosen = Vec::new();
                    while chosen.len() < kind + 1 {
                        let b = (rng.uniform() * bits as f64) as usize;
                        if !chosen.contains(&b) {
                            chosen.push(b);
                        }
                    }
                    chosen.into_iter().for_each(|b| flip(&mut m, b));
                }
                3 | 4 => {
                    let len = if kind == 3 { 1 + (rng.uniform() * 32.0) as usize } else { 33 + (rng.uniform() * 32.0) as usize };
                    let start = (rng.uniform() * (bits - len) as f64) as usize;
                    // A burst: first and last bits flipped, the ones between random.
                    flip(&mut m, start);
                    if len > 1 {
                        flip(&mut m, start + len - 1);
                    }
                    for b in start + 1..start + len.saturating_sub(1) {
                        if rng.uniform() < 0.5 {
                            flip(&mut m, b);
                        }
                    }
                }
                _ => {
                    for _ in 0..8 {
                        let i = (rng.uniform() * m.len() as f64) as usize;
                        let v = (rng.uniform() * 256.0) as u8;
                        m[i] = if v == m[i] { v ^ 1 } else { v };
                    }
                }
            }
            if m != msg && crc32(&m) == good {
                missed += 1;
            }
        }
        rows.push(vec![name.to_string(), format!("{trials}"), format!("{missed}")]);
    }
    write(CHAPTER, "crc_detection.md", &md_table(&["error in a 1024-byte message", "trials", "undetected"], &rows));
    // SHA-256 avalanche.
    let mut hist = vec![0u64; 257];
    let trials = 20_000;
    for _ in 0..trials {
        let msg: Vec<u8> = (0..64).map(|_| (rng.uniform() * 256.0) as u8).collect();
        let mut m2 = msg.clone();
        let b = (rng.uniform() * 512.0) as usize;
        m2[b / 8] ^= 1 << (b % 8);
        let (a, c) = (sha256(&msg), sha256(&m2));
        let d: u32 = a.iter().zip(&c).map(|(x, y)| (x ^ y).count_ones()).sum();
        hist[d as usize] += 1;
    }
    let mean = hist.iter().enumerate().map(|(k, &c)| k as f64 * c as f64).sum::<f64>() / trials as f64;
    let sd = (hist.iter().enumerate().map(|(k, &c)| (k as f64 - mean).powi(2) * c as f64).sum::<f64>() / trials as f64).sqrt();
    let mut plot = Plot::new(&format!("SHA-256 avalanche: output bits changed by flipping one input bit (mean {mean:.1}, sd {sd:.2})"), "output bits changed (of 256)", "fraction of trials");
    plot.series.push(Series::new("measured", (90..=166).map(|k| (k as f64, hist[k] as f64 / trials as f64)).collect(), Style::Markers, 0));
    let binom = |k: u64| (ln_choose(256, k) - 256.0 * 2f64.ln()).exp();
    plot.series.push(Series::new("binomial(256, 1/2)", (90..=166).map(|k| (k as f64, binom(k))).collect(), Style::Line, 1));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "avalanche.svg", &plot.render());
    // Throughput.
    let t0 = Instant::now();
    let c = crc32(KJV);
    let t_crc = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let s = sha256(KJV);
    let t_sha = t0.elapsed().as_secs_f64();
    write(CHAPTER, "throughput_facts.md", &format!("On the 4.6 MB test text, CRC-32 ran at {:.0} MB/s and SHA-256 at {:.0} MB/s (checksum `{c:08x}`, digest `{}…`).", KJV.len() as f64 / 1e6 / t_crc, KJV.len() as f64 / 1e6 / t_sha, &to_hex(&s)[..16]));
}

fn ln_choose(n: u64, k: u64) -> f64 {
    (1..=k).map(|i| ((n - k + i) as f64 / i as f64).ln()).sum()
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
}
