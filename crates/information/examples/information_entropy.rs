//! Entropy, divergence, mutual information and Gray codes.

use lib_jsl_information::operators::{cross_entropy, entropy, entropy_from_probabilities, kl_divergence, mutual_information};
use lib_jsl_information::source_coders::gray_codes::{gray_code, gray_decode};

fn main() {
    println!("entropy of \"aaaa\": {} bits per byte", entropy(b"aaaa"));
    println!("entropy of \"abab\": {} bits per byte", entropy(b"abab"));
    println!("entropy of \"abcd\": {} bits per byte", entropy(b"abcd"));
    println!("entropy of a fair die: {:.4} bits (log2 6 = {:.4})", entropy_from_probabilities(&[1.0 / 6.0; 6]), 6f64.log2());
    let p = [0.5, 0.25, 0.25];
    let q = [1.0 / 3.0; 3];
    println!("H(p) = {}, cross entropy H(p, q) = {:.4}, D(p||q) = {:.4}", entropy_from_probabilities(&p), cross_entropy(&p, &q), kl_divergence(&p, &q));
    // A binary symmetric channel with crossover 0.1 and a uniform input.
    let e = 0.1;
    let joint = [0.5 * (1.0 - e), 0.5 * e, 0.5 * e, 0.5 * (1.0 - e)];
    println!("I(X; Y) = {:.4} bits = 1 - H(0.1) = {:.4}", mutual_information(&joint, 2, 2), 1.0 - entropy_from_probabilities(&[e, 1.0 - e]));
    let codes: Vec<String> = (0..8).map(|n| format!("{:03b}", gray_code(n))).collect();
    println!("3-bit Gray code: {}", codes.join(" "));
    println!("gray_decode(0b110) = {}", gray_decode(0b110));
}
