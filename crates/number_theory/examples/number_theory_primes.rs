//! Primes: sieves, deterministic Miller-Rabin, factorisation, totients and divisors.

use lib_jsl_number_theory::integer_factorization::{divisors, euler_totient, factorize};
use lib_jsl_number_theory::primality::{is_prime, is_strong_probable_prime, next_prime};
use lib_jsl_number_theory::prime_sieves::{prime_count, segmented_sieve, sieve_of_eratosthenes};

fn main() {
    println!("primes below 50: {:?}", sieve_of_eratosthenes(50));
    println!("π(10⁶) = {}", prime_count(1_000_000));
    println!("primes in [10¹², 10¹² + 100]: {:?}", segmented_sieve(1_000_000_000_000, 1_000_000_000_100));
    // 2047 = 23 · 89 fools the base-2 Miller-Rabin test but not base 3.
    println!("2047: base 2 says prime {}, base 3 says prime {}, is_prime {}", is_strong_probable_prime(2047, 2), is_strong_probable_prime(2047, 3), is_prime(2047));
    println!("largest 64-bit prime is_prime: {}; next prime after 10^18: {:?}", is_prime(18446744073709551557), next_prime(1_000_000_000_000_000_000));
    // 2^64 − 1 and a product of two 10-digit primes.
    for n in [u64::MAX, 1_000_000_007 * 1_000_000_009] {
        println!("{n} = {:?}", factorize(n));
    }
    println!("φ(360) = {}, divisors of 360: {:?}", euler_totient(360), divisors(360));
}
