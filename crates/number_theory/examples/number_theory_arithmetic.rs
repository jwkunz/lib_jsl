//! Modular arithmetic: gcd, Bézout coefficients, inverses, powers, integer square roots and the
//! Chinese remainder theorem, ending with a toy RSA key.

use lib_jsl_number_theory::base_exponent_in_modulus::base_exponent_in_modulus;
use lib_jsl_number_theory::chinese_remainder_theorem::chinese_remainder;
use lib_jsl_number_theory::greatest_common_divisor::{extended_gcd, gcd};
use lib_jsl_number_theory::integer_square_root::integer_square_root_newton;
use lib_jsl_number_theory::multiplicative_inverse_modulo::multiplicative_inverse_modulo;

fn main() {
    println!("gcd(1071, 462) = {}", gcd(1071, 462));
    let (g, x, y) = extended_gcd(240, 46);
    println!("240·({x}) + 46·({y}) = {g}");
    println!("17⁻¹ mod 3120 = {}", multiplicative_inverse_modulo(17, 3120).unwrap());
    println!("2^1000 mod 1000000007 = {}", base_exponent_in_modulus(2, 1000, 1_000_000_007).unwrap());
    println!("⌊√(10^30 + 1)⌋ = {}", integer_square_root_newton(10u128.pow(30) + 1));
    // x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7), the problem of Sunzi.
    let (x, m) = chinese_remainder(&[2, 3, 2], &[3, 5, 7]).unwrap();
    println!("Sunzi: x ≡ {x} (mod {m})");
    // Toy RSA: n = p q, d = e⁻¹ mod (p − 1)(q − 1).
    let (p, q, e) = (61i128, 53i128, 17i128);
    let (n, phi) = (p * q, (p - 1) * (q - 1));
    let d = multiplicative_inverse_modulo(e, phi).unwrap();
    let c = base_exponent_in_modulus(65, e, n).unwrap();
    println!("RSA with n = {n}, e = {e}, d = {d}: 65 → {c} → {}", base_exponent_in_modulus(c, d, n).unwrap());
}
