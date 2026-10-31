//! Reed-Solomon codes over GF(2^8).
//!
//! Arithmetic uses the field polynomial `x^8 + x^4 + x^3 + x^2 + 1` (0x11D) with primitive
//! element `α = 2`, and the generator polynomial has roots `α^0 ... α^(n-k-1)` (the convention
//! used by QR codes). Codes may be shortened to any `n <= 255`. Encoding is systematic: a codeword
//! is the message bytes followed by `n - k` parity bytes. Decoding corrects up to `(n - k) / 2`
//! byte errors using Berlekamp-Massey, a Chien search and a direct solve for error magnitudes.

use lib_jsl_core::ErrorsJSL;
use std::sync::OnceLock;

struct Field {
    exp: [u8; 512],
    log: [u8; 256],
}

#[allow(clippy::needless_range_loop)]
fn field() -> &'static Field {
    static F: OnceLock<Field> = OnceLock::new();
    F.get_or_init(|| {
        let mut exp = [0u8; 512];
        let mut log = [0u8; 256];
        let mut x: u16 = 1;
        for i in 0..255 {
            exp[i] = x as u8;
            log[x as usize] = i as u8;
            x <<= 1;
            if x & 0x100 != 0 {
                x ^= 0x11D;
            }
        }
        for i in 255..512 {
            exp[i] = exp[i - 255];
        }
        Field { exp, log }
    })
}

fn gmul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    let f = field();
    f.exp[f.log[a as usize] as usize + f.log[b as usize] as usize]
}

fn gdiv(a: u8, b: u8) -> u8 {
    assert!(b != 0, "division by zero in GF(256)");
    if a == 0 {
        return 0;
    }
    let f = field();
    f.exp[f.log[a as usize] as usize + 255 - f.log[b as usize] as usize]
}

fn alpha_pow(e: i64) -> u8 {
    field().exp[e.rem_euclid(255) as usize]
}

/// Evaluates a polynomial given highest-degree-first at `x`.
fn eval_high_first(poly: &[u8], x: u8) -> u8 {
    poly.iter().fold(0, |acc, &c| gmul(acc, x) ^ c)
}

/// Evaluates a polynomial given lowest-degree-first at `x`.
fn eval_low_first(poly: &[u8], x: u8) -> u8 {
    poly.iter().rev().fold(0, |acc, &c| gmul(acc, x) ^ c)
}

/// A Reed-Solomon code with codeword length `n` and `k` message bytes.
#[derive(Clone, Debug)]
pub struct ReedSolomon {
    n: usize,
    k: usize,
    generator: Vec<u8>, // highest degree first, monic, length nsym + 1
}

impl ReedSolomon {
    /// Creates a code with `n <= 255` and `0 < k < n`.
    pub fn new(n: usize, k: usize) -> Result<Self, ErrorsJSL> {
        if n > 255 || k == 0 || k >= n {
            return Err(ErrorsJSL::InvalidInputRange("Reed-Solomon requires 0 < k < n <= 255."));
        }
        let nsym = n - k;
        let mut generator = vec![1u8];
        for i in 0..nsym {
            let root = alpha_pow(i as i64);
            let mut next = vec![0u8; generator.len() + 1];
            for (j, &c) in generator.iter().enumerate() {
                next[j] ^= c;
                next[j + 1] ^= gmul(c, root);
            }
            generator = next;
        }
        Ok(ReedSolomon { n, k, generator })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// The number of byte errors that can be corrected, `(n - k) / 2`.
    pub fn error_capacity(&self) -> usize {
        (self.n - self.k) / 2
    }

    #[allow(clippy::needless_range_loop)]
    /// Computes the `n - k` parity bytes of a `k`-byte message.
    pub fn parity(&self, message: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
        if message.len() != self.k {
            return Err(ErrorsJSL::IncompatibleArraySizes((message.len(), self.k)));
        }
        let nsym = self.n - self.k;
        let mut rem = vec![0u8; nsym];
        for &byte in message {
            let feedback = byte ^ rem[0];
            rem.rotate_left(1);
            rem[nsym - 1] = 0;
            if feedback != 0 {
                for j in 0..nsym {
                    rem[j] ^= gmul(self.generator[j + 1], feedback);
                }
            }
        }
        Ok(rem)
    }

    /// Encodes a `k`-byte message into an `n`-byte systematic codeword (message then parity).
    pub fn encode(&self, message: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
        let mut cw = message.to_vec();
        cw.extend(self.parity(message)?);
        Ok(cw)
    }

    fn syndromes(&self, word: &[u8]) -> Vec<u8> {
        (0..self.n - self.k).map(|j| eval_high_first(word, alpha_pow(j as i64))).collect()
    }

    /// Decodes a received `n`-byte word. On success returns the corrected `k`-byte message and
    /// the number of corrected byte errors; fails if the word is uncorrectable.
    pub fn decode(&self, received: &[u8]) -> Result<(Vec<u8>, usize), ErrorsJSL> {
        let (cw, count) = self.correct(received)?;
        Ok((cw[..self.k].to_vec(), count))
    }

    /// Like [`decode`](Self::decode) but returns the full corrected `n`-byte codeword.
    pub fn correct(&self, received: &[u8]) -> Result<(Vec<u8>, usize), ErrorsJSL> {
        if received.len() != self.n {
            return Err(ErrorsJSL::IncompatibleArraySizes((received.len(), self.n)));
        }
        let nsym = self.n - self.k;
        let s = self.syndromes(received);
        if s.iter().all(|&v| v == 0) {
            return Ok((received.to_vec(), 0));
        }
        // Berlekamp-Massey for the error locator polynomial (lowest degree first).
        let mut c = vec![1u8];
        let mut b = vec![1u8];
        let (mut l, mut m, mut bb) = (0usize, 1usize, 1u8);
        for i in 0..nsym {
            let mut d = s[i];
            for j in 1..=l.min(c.len() - 1) {
                d ^= gmul(c[j], s[i - j]);
            }
            if d == 0 {
                m += 1;
            } else if 2 * l <= i {
                let t = c.clone();
                let coef = gdiv(d, bb);
                if c.len() < b.len() + m {
                    c.resize(b.len() + m, 0);
                }
                for (j, &bj) in b.iter().enumerate() {
                    c[j + m] ^= gmul(coef, bj);
                }
                l = i + 1 - l;
                b = t;
                bb = d;
                m = 1;
            } else {
                let coef = gdiv(d, bb);
                if c.len() < b.len() + m {
                    c.resize(b.len() + m, 0);
                }
                for (j, &bj) in b.iter().enumerate() {
                    c[j + m] ^= gmul(coef, bj);
                }
                m += 1;
            }
        }
        if l > nsym / 2 {
            return Err(ErrorsJSL::RuntimeError("Reed-Solomon word has too many errors to correct."));
        }
        // Chien search: error at codeword index i has locator X = α^(n-1-i), a root of σ at X^-1.
        let mut positions = Vec::new();
        for i in 0..self.n {
            let x_inv = alpha_pow(-((self.n - 1 - i) as i64));
            if eval_low_first(&c, x_inv) == 0 {
                positions.push(i);
            }
        }
        if positions.len() != l {
            return Err(ErrorsJSL::RuntimeError("Reed-Solomon word is uncorrectable."));
        }
        // Solve S_j = Σ_k e_k X_k^j, j = 0..l-1, for the error values by Gauss-Jordan elimination.
        let xs: Vec<u8> = positions.iter().map(|&i| alpha_pow((self.n - 1 - i) as i64)).collect();
        let mut a: Vec<Vec<u8>> = (0..l)
            .map(|j| {
                let mut row: Vec<u8> = xs.iter().map(|&x| (0..j).fold(1u8, |acc, _| gmul(acc, x))).collect();
                row.push(s[j]);
                row
            })
            .collect();
        for col in 0..l {
            let pivot = (col..l)
                .find(|&r| a[r][col] != 0)
                .ok_or(ErrorsJSL::RuntimeError("Reed-Solomon error magnitudes are singular."))?;
            a.swap(col, pivot);
            let inv = gdiv(1, a[col][col]);
            for v in a[col].iter_mut() {
                *v = gmul(*v, inv);
            }
            for r in 0..l {
                if r != col && a[r][col] != 0 {
                    let f = a[r][col];
                    let pivot_row = a[col].clone();
                    for (v, &p) in a[r].iter_mut().zip(&pivot_row) {
                        *v ^= gmul(f, p);
                    }
                }
            }
        }
        let mut word = received.to_vec();
        for (idx, &pos) in positions.iter().enumerate() {
            word[pos] ^= a[idx][l];
        }
        if self.syndromes(&word).iter().any(|&v| v != 0) {
            return Err(ErrorsJSL::RuntimeError("Reed-Solomon word is uncorrectable."));
        }
        Ok((word, l))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic generator for test data.
    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *seed >> 33
    }

    #[test]
    fn qr_code_example() {
        // QR version 1-M style block: 16 data bytes + 10 error-correction bytes.
        let msg = [0x40, 0xd2, 0x75, 0x47, 0x76, 0x17, 0x32, 0x06, 0x27, 0x26, 0x96, 0xc6, 0xc6, 0x96, 0x70, 0xec];
        let rs = ReedSolomon::new(26, 16).unwrap();
        assert_eq!(
            rs.parity(&msg).unwrap(),
            vec![0xbc, 0x2a, 0x90, 0x13, 0x6b, 0xaf, 0xef, 0xfd, 0x4b, 0xe0]
        );
    }

    #[test]
    fn corrects_up_to_capacity() {
        let rs = ReedSolomon::new(255, 223).unwrap();
        assert_eq!(rs.error_capacity(), 16);
        let mut seed = 12345;
        let msg: Vec<u8> = (0..223).map(|_| lcg(&mut seed) as u8).collect();
        let cw = rs.encode(&msg).unwrap();
        assert_eq!(rs.decode(&cw).unwrap(), (msg.clone(), 0));
        for nerr in [1usize, 5, 16] {
            let mut bad = cw.clone();
            let mut used = Vec::new();
            while used.len() < nerr {
                let p = lcg(&mut seed) as usize % 255;
                if !used.contains(&p) {
                    used.push(p);
                    bad[p] ^= (lcg(&mut seed) as u8) | 1;
                }
            }
            let (decoded, count) = rs.decode(&bad).unwrap();
            assert_eq!(decoded, msg, "{nerr} errors");
            assert_eq!(count, nerr);
        }
    }

    #[test]
    fn errors_in_parity_and_shortened_code() {
        let rs = ReedSolomon::new(30, 20).unwrap();
        let msg: Vec<u8> = (0..20).map(|i| (i * 13 + 5) as u8).collect();
        let mut cw = rs.encode(&msg).unwrap();
        cw[0] ^= 0xFF;
        cw[29] ^= 0x01;
        cw[22] ^= 0x80;
        cw[10] ^= 0x33;
        cw[15] ^= 0x44;
        let (decoded, count) = rs.correct(&cw).unwrap();
        assert_eq!(&decoded[..20], &msg[..]);
        assert_eq!(count, 5);
    }

    #[test]
    fn too_many_errors_is_not_silently_accepted() {
        let rs = ReedSolomon::new(255, 223).unwrap();
        let msg: Vec<u8> = (0..223).map(|i| i as u8).collect();
        let mut bad = rs.encode(&msg).unwrap();
        for i in 0..40 {
            bad[i * 6] ^= 0x5A;
        }
        match rs.decode(&bad) {
            Err(_) => {}
            Ok((decoded, _)) => assert_ne!(decoded, msg),
        }
    }

    #[test]
    fn invalid_parameters_and_lengths() {
        assert!(ReedSolomon::new(256, 200).is_err());
        assert!(ReedSolomon::new(10, 10).is_err());
        let rs = ReedSolomon::new(15, 11).unwrap();
        assert!(rs.encode(&[0; 10]).is_err());
        assert!(rs.decode(&[0; 14]).is_err());
    }
}
