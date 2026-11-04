//! Figures and tables for the chapter on number theory.

use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_number_theory::base_exponent_in_modulus::base_exponent_in_modulus;
use lib_jsl_number_theory::chinese_remainder_theorem::chinese_remainder;
use lib_jsl_number_theory::greatest_common_divisor::{extended_gcd, gcd};
use lib_jsl_number_theory::integer_factorization::{euler_totient, factorize, pollard_rho};
use lib_jsl_number_theory::integer_square_root::{integer_square_root_binary_search, integer_square_root_newton};
use lib_jsl_number_theory::multiplicative_inverse_modulo::multiplicative_inverse_modulo;
use lib_jsl_number_theory::polynomial::{polynomial_from_roots, polynomial_roots};
use lib_jsl_number_theory::primality::{is_prime, is_strong_probable_prime, next_prime};
use lib_jsl_number_theory::prime_sieves::{prime_count, segmented_sieve, sieve_of_eratosthenes};
use lib_jsl_number_theory::text_coded_integer::TextCodedInteger;
use num_complex::Complex;
use std::collections::HashMap;
use std::time::Instant;

const CHAPTER: &str = "number_theory";

pub fn generate() {
    let refs: HashMap<&str, Vec<i128>> = include_str!("../data/number_theory_reference.txt")
        .lines()
        .map(|l| {
            let (k, v) = l.split_once('|').unwrap();
            (k, v.split(',').map(|x| x.parse::<i128>().unwrap()).collect())
        })
        .collect();
    reference_table(&refs);
    euclid_figures();
    prime_figures();
    polynomial_figures();
}

fn reference_table(refs: &HashMap<&str, Vec<i128>>) {
    let mut rows = Vec::new();
    let mut prime_ok = (0, 0);
    let mut factor_ok = (0, 0);
    for (key, v) in refs {
        if let Some(n) = key.strip_prefix("isprime ") {
            let n: u64 = n.parse().unwrap();
            prime_ok.1 += 1;
            prime_ok.0 += (is_prime(n) == (v[0] == 1)) as usize;
        } else if let Some(n) = key.strip_prefix("factor ") {
            let n: u64 = n.parse().unwrap();
            let got: Vec<i128> = factorize(n).iter().flat_map(|&(p, e)| [p as i128, e as i128]).collect();
            let tot_ok = euler_totient(n) as i128 == refs[format!("totient {n}").as_str()][0];
            factor_ok.1 += 1;
            factor_ok.0 += (got == *v && tot_ok) as usize;
        }
    }
    rows.push(vec!["primality (`isprime`), including strong pseudoprimes and Carmichael numbers".to_string(), format!("{} of {} agree", prime_ok.0, prime_ok.1)]);
    rows.push(vec!["factorisation and totient (`factorint`, `totient`), up to 2⁶⁴ − 1".to_string(), format!("{} of {} agree", factor_ok.0, factor_ok.1)]);
    let pi_ok = (1..=8).filter(|&k| prime_count(10u64.pow(k)) as i128 == refs[format!("primepi {}", 10u64.pow(k)).as_str()][0]).count();
    rows.push(vec!["prime counting π(10ᵏ), k = 1 … 8 (`primepi`)".to_string(), format!("{pi_ok} of 8 agree")]);
    let crt1 = chinese_remainder(&[2, 3, 2], &[3, 5, 7]).unwrap();
    let crt2 = chinese_remainder(&[4, 8, 13], &[6, 10, 15]).unwrap();
    rows.push(vec!["Chinese remainder theorem, coprime and non-coprime moduli (`crt`)".to_string(), format!("{}", [crt1.0, crt1.1, crt2.0, crt2.1] == [refs["crt"][0], refs["crt"][1], refs["crt noncoprime"][0], refs["crt noncoprime"][1]])]);
    let inv = [multiplicative_inverse_modulo(17, 3120).unwrap(), multiplicative_inverse_modulo(12345678901234567, 10i128.pow(30) + 57).unwrap()];
    rows.push(vec!["modular inverses, moduli up to 10³⁰ (`mod_inverse`)".to_string(), format!("{}", inv.to_vec() == refs["inverse"])]);
    let r1 = (10u128.pow(30) + 12345, (1u128 << 127) - 1);
    let sq = [integer_square_root_newton(r1.0), integer_square_root_newton(r1.1), integer_square_root_binary_search(r1.0), integer_square_root_binary_search(r1.1)];
    rows.push(vec!["integer square roots, Newton and bisection (`integer_nthroot`)".to_string(), format!("{}", sq.iter().map(|&v| v as i128).collect::<Vec<_>>() == [refs["isqrt"].clone(), refs["isqrt"].clone()].concat())]);
    let pw = [base_exponent_in_modulus(7, 10i128.pow(18) + 9, 10i128.pow(30) + 57).unwrap(), base_exponent_in_modulus(3, 1i128 << 100, (1i128 << 89) - 1).unwrap()];
    rows.push(vec!["modular powers, moduli up to 10³⁰ (Python `pow`)".to_string(), format!("{}", pw.to_vec() == refs["powmod"])]);
    rows.sort();
    write(CHAPTER, "reference.md", &md_table(&["computation (SymPy function)", "agreement"], &rows));
}

// ------------------------------------------------------------------ Euclid

fn euclid_steps(mut a: u64, mut b: u64) -> u32 {
    let mut steps = 0;
    while b != 0 {
        (a, b) = (b, a % b);
        steps += 1;
    }
    steps
}

fn euclid_figures() {
    // Average number of division steps against Heilbronn's (12 ln 2 / π²) ln n, and Lamé's worst
    // case at consecutive Fibonacci numbers.
    let mut seed = 12345u64;
    // 64 random bits from the high halves of two LCG steps.
    let mut rnd = |m: u64| {
        let mut half = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            seed >> 32
        };
        ((half() << 32) | half()) % m
    };
    let mut pts = Vec::new();
    let mut worst = Vec::new();
    let (mut f0, mut f1) = (1u64, 1u64);
    for k in 1..=18 {
        let n = 10f64.powf(k as f64);
        let m = n as u64;
        let trials = 20_000;
        let mean = (0..trials).map(|_| {
            let (a, b) = (rnd(m) + 1, rnd(m) + 1);
            assert_eq!(gcd(a as i128, b as i128) as u64, {
                let (mut x, mut y) = (a, b);
                while y != 0 {
                    (x, y) = (y, x % y);
                }
                x
            });
            euclid_steps(a, b) as f64
        }).sum::<f64>() / trials as f64;
        pts.push((n, mean));
        // The largest consecutive Fibonacci pair (f1, f0) with f1 ≤ m.
        while f0 + f1 <= m {
            (f0, f1) = (f1, f0 + f1);
        }
        worst.push((n, euclid_steps(f1, f0) as f64));
    }
    let mut plot = Plot::new("Division steps of the Euclidean algorithm for pairs below n", "n", "division steps");
    plot.series.push(Series::new("mean over 20 000 random pairs", pts.clone(), Style::Markers, 0));
    // Heilbronn's slope 12 ln 2 / π², with the offset fitted to the measurements.
    let slope = 12.0 * 2f64.ln() / std::f64::consts::PI.powi(2);
    let offset = pts.iter().map(|&(n, m)| m - slope * n.ln()).sum::<f64>() / pts.len() as f64;
    plot.series.push(Series::new(&format!("(12 ln 2 / π²) ln n + {offset:.2}"), pts.iter().map(|&(n, _)| (n, slope * n.ln() + offset)).collect(), Style::Line, 1));
    plot.series.push(Series::new("worst case: consecutive Fibonacci numbers", worst, Style::LineMarkers, 2));
    plot.log_x = true;
    plot.legend = Legend::TopLeft;
    write(CHAPTER, "euclid_steps.svg", &plot.render());
    // A worked extended Euclid / RSA example.
    let (p, q) = (1000000007i128, 998244353i128);
    let n = p * q;
    let phi = (p - 1) * (q - 1);
    let e = 65537i128;
    let d = multiplicative_inverse_modulo(e, phi).unwrap();
    let (g, x, y) = extended_gcd(e, phi);
    let message = 4242424242i128;
    let c = base_exponent_in_modulus(message, e, n).unwrap();
    let back = base_exponent_in_modulus(c, d, n).unwrap();
    write(
        CHAPTER,
        "rsa_facts.md",
        &format!(
            "With $p = {p}$, $q = {q}$, $n = pq = {n}$ and $e = {e}$, the extended Euclidean algorithm gives ${e} \\cdot ({x}) + {phi} \\cdot ({y}) = {g}$, so $d = {d}$. The message {message} encrypts to {c} and decrypts back to {back}."
        ),
    );
}

// ------------------------------------------------------------------ primes

fn li(x: f64) -> f64 {
    // The offset logarithmic integral Li(x) = ∫₂ˣ dt / ln t, by Simpson's rule in log space.
    let (a, b) = (2f64.ln(), x.ln());
    let n = 2000;
    let h = (b - a) / n as f64;
    let f = |u: f64| u.exp() / u;
    (0..=n).map(|i| {
        let w = if i == 0 || i == n { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
        w * f(a + i as f64 * h)
    }).sum::<f64>() * h / 3.0
}

fn prime_figures() {
    // π(x) against x/ln x and Li(x).
    let primes = sieve_of_eratosthenes(10_000_000);
    let xs: Vec<f64> = (0..=60).map(|i| 10f64.powf(2.0 + 5.0 * i as f64 / 60.0)).collect();
    let pi = |x: f64| primes.partition_point(|&p| (p as f64) <= x) as f64;
    let mut plot = Plot::new("The prime-counting function against its approximations", "x", "relative error of the approximation");
    plot.series.push(Series::new("x / ln x", xs.iter().map(|&x| (x, (x / x.ln() - pi(x)) / pi(x))).collect(), Style::Line, 0));
    plot.series.push(Series::new("Li(x)", xs.iter().map(|&x| (x, (li(x) - pi(x)) / pi(x))).collect(), Style::Line, 1));
    plot.log_x = true;
    plot.legend = Legend::TopRight;
    write(CHAPTER, "prime_counting.svg", &plot.render());
    // Sieve timings.
    let mut rows = Vec::new();
    for k in [6u32, 7, 8, 9] {
        let n = 10u64.pow(k);
        let t0 = Instant::now();
        let count = prime_count(n);
        let t_full = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let window = segmented_sieve(n - 1_000_000, n).len();
        let t_seg = t0.elapsed().as_secs_f64();
        rows.push(vec![format!("10^{k}"), format!("{count}"), format!("{:.0}", 1e3 * t_full), format!("{window}"), format!("{:.1}", 1e3 * t_seg)]);
    }
    write(CHAPTER, "sieve_table.md", &md_table(&["n", "π(n)", "sieve of Eratosthenes (ms)", "primes in [n − 10⁶, n]", "segmented sieve of that window (ms)"], &rows));
    // Strong pseudoprimes against the Miller-Rabin bases.
    let bases = [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    let mut rows = Vec::new();
    for n in [2047u64, 1373653, 25326001, 3215031751, 2152302898747, 3474749660383, 341550071728321, 3825123056546413051] {
        let fooled: Vec<String> = bases.iter().filter(|&&a| is_strong_probable_prime(n, a)).map(|a| a.to_string()).collect();
        let f: Vec<String> = factorize(n).iter().map(|(p, e)| if *e > 1 { format!("{p}^{e}") } else { p.to_string() }).collect();
        rows.push(vec![format!("{n}"), f.join(" · "), fooled.join(", "), format!("{}", is_prime(n))]);
    }
    write(CHAPTER, "pseudoprimes.md", &md_table(&["n", "factors", "bases it passes as a strong probable prime", "is_prime"], &rows));
    // Pollard's rho: time against the smallest factor.
    let mut pts = Vec::new();
    for k in [3u32, 4, 5, 6, 7, 8, 9] {
        let p = next_prime(10u64.pow(k)).unwrap();
        let q = next_prime(u64::MAX / p / 2).unwrap();
        let n = p * q;
        let reps = 20;
        let t0 = Instant::now();
        for _ in 0..reps {
            let d = pollard_rho(n);
            assert!(d == p || d == q);
        }
        pts.push((p as f64, t0.elapsed().as_secs_f64() / reps as f64 * 1e6));
    }
    let c = pts[3].1 / pts[3].0.sqrt();
    let mut plot = Plot::new("Pollard's rho on n = p·q ≈ 2⁶³: time against the smaller factor p", "p", "time (µs)");
    plot.series.push(Series::new("measured", pts.clone(), Style::Markers, 0));
    plot.series.push(Series::new("∝ √p", pts.iter().map(|&(p, _)| (p, c * p.sqrt())).collect(), Style::Line, 1));
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::TopLeft;
    write(CHAPTER, "pollard_rho.svg", &plot.render());
}

// ------------------------------------------------------------------ polynomials

fn polynomial_figures() {
    // Wilkinson's polynomial: the roots 1, 2, …, n recovered from the expanded coefficients.
    let mut pts = Vec::new();
    for n in 2..=20 {
        let roots: Vec<Complex<f64>> = (1..=n).map(|k| Complex::new(k as f64, 0.0)).collect();
        let coeffs = polynomial_from_roots(&roots);
        let found = polynomial_roots(coeffs.as_slice().unwrap()).unwrap();
        let mut f: Vec<Complex<f64>> = found.to_vec();
        f.sort_by(|a, b| a.re.total_cmp(&b.re));
        let err = f.iter().zip(&roots).map(|(a, b)| (a - b).norm()).fold(0.0, f64::max);
        pts.push((n as f64, err.max(1e-16)));
    }
    let mut plot = Plot::new("Wilkinson's polynomial ∏(x − k): roots from the expanded coefficients", "degree n", "largest root error");
    plot.series.push(Series::new("companion-matrix eigenvalues", pts, Style::LineMarkers, 0));
    plot.log_y = true;
    plot.legend = Legend::TopLeft;
    write(CHAPTER, "wilkinson.svg", &plot.render());
    // Text-coded integer arithmetic: factorials and timings.
    let mut rows = Vec::new();
    for n in [20u32, 50, 100, 200] {
        let t0 = Instant::now();
        let mut f = TextCodedInteger::new(1);
        for k in 2..=n {
            f = f * TextCodedInteger::new(k);
        }
        let t = t0.elapsed().as_secs_f64();
        let s = f.to_string();
        rows.push(vec![format!("{n}!"), format!("{}", s.len()), format!("{}…{}", &s[..12.min(s.len())], &s[s.len().saturating_sub(6)..]), format!("{:.2}", 1e3 * t)]);
    }
    write(CHAPTER, "factorials.md", &md_table(&["value", "decimal digits", "digits", "time to compute (ms)"], &rows));
}
