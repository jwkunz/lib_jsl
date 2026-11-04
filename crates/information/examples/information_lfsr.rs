//! Linear feedback shift registers, primitive polynomials and Gold codes.

use lib_jsl_information::lfsr::{Lfsr, gold_codes, gold_correlation_bound, is_primitive_polynomial, maximum_length_polynomial};

fn main() {
    // x^4 + x + 1 is primitive: the register visits all 15 nonzero states.
    let mut r = Lfsr::from_polynomial(0b10011, 0b0001);
    println!("x^4 + x + 1: period {:?}, bits {:?}", r.period(100), r.bits(15));
    // x^4 + x^3 + x^2 + x + 1 is irreducible but not primitive: x^5 = 1, so the period is 5.
    println!("x^4 + x^3 + x^2 + x + 1: primitive {}, period {:?}", is_primitive_polynomial(0b11111), Lfsr::from_polynomial(0b11111, 1).period(100));
    println!("degree-16 maximum-length polynomial: {:#x}", maximum_length_polynomial(16).unwrap());
    let family = gold_codes(5).unwrap();
    let chips = |s: &[u8]| s.iter().map(|&b| 1 - 2 * b as i32).collect::<Vec<i32>>();
    let (a, b) = (chips(&family[2]), chips(&family[7]));
    let mut values: Vec<i32> = (0..31).map(|s| (0..31).map(|t| a[t] * b[(t + s) % 31]).sum()).collect();
    values.sort();
    values.dedup();
    println!("{} Gold codes of length {}; cross-correlation values {values:?} (bound t(5) = {})", family.len(), family[0].len(), gold_correlation_bound(5).unwrap());
}
