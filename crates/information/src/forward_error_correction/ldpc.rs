//! Low-density parity-check (LDPC) codes with sum-product decoding.
//!
//! A code is defined by a sparse parity-check matrix `H` (each check lists the codeword positions
//! it covers). Systematic encoding is derived by Gauss-Jordan elimination of `H` over GF(2);
//! decoding is belief propagation (the sum-product algorithm) with a flooding schedule on
//! log-likelihood ratios `ln P(0)/P(1)`. Bits are `u8` values 0 or 1.

use lib_jsl_core::ErrorsJSL;

/// The outcome of belief-propagation decoding.
#[derive(Clone, Debug, PartialEq)]
pub struct LdpcDecoded {
    /// The hard-decision codeword estimate (`n` bits).
    pub codeword: Vec<u8>,
    /// Whether all parity checks were satisfied.
    pub converged: bool,
    /// Number of decoding iterations performed.
    pub iterations: usize,
}

/// An LDPC code.
#[derive(Clone, Debug)]
pub struct LdpcCode {
    n: usize,
    checks: Vec<Vec<usize>>,
    var_checks: Vec<Vec<usize>>,
    info_cols: Vec<usize>,
    /// For each pivot row of the reduced matrix: (pivot column, info columns it depends on).
    parity_rows: Vec<(usize, Vec<usize>)>,
}

impl LdpcCode {
    /// Builds a code of length `n` from the lists of codeword positions covered by each check.
    pub fn new(n: usize, checks: Vec<Vec<usize>>) -> Result<Self, ErrorsJSL> {
        if n == 0 || checks.is_empty() {
            return Err(ErrorsJSL::InvalidInputRange("An LDPC code needs a nonzero length and at least one check."));
        }
        let mut checks = checks;
        for c in checks.iter_mut() {
            if c.iter().any(|&v| v >= n) {
                return Err(ErrorsJSL::InvalidInputRange("A parity check references a position outside the codeword."));
            }
            c.sort_unstable();
            c.dedup();
        }
        let mut var_checks = vec![Vec::new(); n];
        for (ci, c) in checks.iter().enumerate() {
            for &v in c {
                var_checks[v].push(ci);
            }
        }
        // Gauss-Jordan elimination over GF(2) on dense rows.
        let m = checks.len();
        let mut rows: Vec<Vec<u8>> = checks
            .iter()
            .map(|c| {
                let mut r = vec![0u8; n];
                for &v in c {
                    r[v] = 1;
                }
                r
            })
            .collect();
        let mut pivots: Vec<usize> = Vec::new();
        let mut r = 0;
        for col in 0..n {
            if r == m {
                break;
            }
            if let Some(p) = (r..m).find(|&i| rows[i][col] == 1) {
                rows.swap(r, p);
                let pivot_row = rows[r].clone();
                for (i, row) in rows.iter_mut().enumerate() {
                    if i != r && row[col] == 1 {
                        for (x, &y) in row.iter_mut().zip(&pivot_row) {
                            *x ^= y;
                        }
                    }
                }
                pivots.push(col);
                r += 1;
            }
        }
        let is_pivot = {
            let mut v = vec![false; n];
            for &p in &pivots {
                v[p] = true;
            }
            v
        };
        let info_cols: Vec<usize> = (0..n).filter(|&c| !is_pivot[c]).collect();
        let parity_rows = pivots
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, info_cols.iter().copied().filter(|&c| rows[i][c] == 1).collect()))
            .collect();
        Ok(LdpcCode { n, checks, var_checks, info_cols, parity_rows })
    }

    /// Builds a regular Gallager `(wc, wr)` code of length `n` (`n` a multiple of `wr`, `wc >= 2`)
    /// with a deterministic pseudo-random column permutation controlled by `seed`.
    pub fn gallager(n: usize, wc: usize, wr: usize, seed: u64) -> Result<Self, ErrorsJSL> {
        if wc < 2 || wr < 3 || !n.is_multiple_of(wr) || n == 0 {
            return Err(ErrorsJSL::InvalidInputRange("Gallager codes need wc >= 2, wr >= 3 and n a multiple of wr."));
        }
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let rows_per_block = n / wr;
        let mut checks = Vec::with_capacity(wc * rows_per_block);
        for block in 0..wc {
            let mut perm: Vec<usize> = (0..n).collect();
            if block > 0 {
                for i in (1..n).rev() {
                    perm.swap(i, (next() % (i as u64 + 1)) as usize);
                }
            }
            for r in 0..rows_per_block {
                checks.push((r * wr..(r + 1) * wr).map(|c| perm[c]).collect());
            }
        }
        Self::new(n, checks)
    }

    /// Codeword length.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of information bits (`n` minus the rank of `H`).
    pub fn k(&self) -> usize {
        self.info_cols.len()
    }

    /// Number of parity-check rows (may exceed the rank).
    pub fn num_checks(&self) -> usize {
        self.checks.len()
    }

    /// Codeword positions that carry the information bits, in increasing order.
    pub fn info_positions(&self) -> &[usize] {
        &self.info_cols
    }

    /// Encodes `k` information bits into an `n`-bit systematic codeword.
    pub fn encode(&self, info: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
        if info.len() != self.k() {
            return Err(ErrorsJSL::IncompatibleArraySizes((info.len(), self.k())));
        }
        let mut cw = vec![0u8; self.n];
        for (&c, &b) in self.info_cols.iter().zip(info) {
            cw[c] = b & 1;
        }
        for (pivot, deps) in &self.parity_rows {
            cw[*pivot] = deps.iter().fold(0u8, |a, &c| a ^ cw[c]);
        }
        Ok(cw)
    }

    /// Extracts the information bits from a codeword.
    pub fn extract_info(&self, codeword: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
        if codeword.len() != self.n {
            return Err(ErrorsJSL::IncompatibleArraySizes((codeword.len(), self.n)));
        }
        Ok(self.info_cols.iter().map(|&c| codeword[c]).collect())
    }

    /// Returns true if every parity check is satisfied by `word`.
    pub fn is_codeword(&self, word: &[u8]) -> bool {
        word.len() == self.n && self.checks.iter().all(|c| c.iter().fold(0u8, |a, &v| a ^ (word[v] & 1)) == 0)
    }

    #[allow(clippy::needless_range_loop)]
    /// Sum-product decoding of channel LLRs for at most `max_iterations` iterations.
    pub fn decode(&self, llr: &[f64], max_iterations: usize) -> Result<LdpcDecoded, ErrorsJSL> {
        if llr.len() != self.n {
            return Err(ErrorsJSL::IncompatibleArraySizes((llr.len(), self.n)));
        }
        let hard = |l: &[f64]| -> Vec<u8> { l.iter().map(|&x| (x < 0.0) as u8).collect() };
        let mut word = hard(llr);
        if self.is_codeword(&word) {
            return Ok(LdpcDecoded { codeword: word, converged: true, iterations: 0 });
        }
        // Messages stored per check edge: v2c[c][j], c2v[c][j] for the j-th variable of check c.
        let mut v2c: Vec<Vec<f64>> = self.checks.iter().map(|c| c.iter().map(|&v| llr[v]).collect()).collect();
        let mut c2v: Vec<Vec<f64>> = self.checks.iter().map(|c| vec![0.0; c.len()]).collect();
        // Index of check c within var_checks[v], to route messages.
        let slot: Vec<Vec<usize>> = self
            .var_checks
            .iter()
            .enumerate()
            .map(|(v, cs)| cs.iter().map(|&c| self.checks[c].iter().position(|&x| x == v).unwrap()).collect())
            .collect();
        for iter in 1..=max_iterations {
            // Check-node update.
            for (c, vars) in self.checks.iter().enumerate() {
                let t: Vec<f64> = v2c[c].iter().map(|&m| (0.5 * m.clamp(-30.0, 30.0)).tanh()).collect();
                for j in 0..vars.len() {
                    let prod: f64 = t.iter().enumerate().filter(|&(i, _)| i != j).map(|(_, &x)| x).product();
                    c2v[c][j] = 2.0 * prod.clamp(-0.999_999_999_999, 0.999_999_999_999).atanh();
                }
            }
            // Variable-node update and tentative decision.
            let mut total = llr.to_vec();
            for v in 0..self.n {
                for (k, &c) in self.var_checks[v].iter().enumerate() {
                    total[v] += c2v[c][slot[v][k]];
                }
            }
            for v in 0..self.n {
                for (k, &c) in self.var_checks[v].iter().enumerate() {
                    let j = slot[v][k];
                    v2c[c][j] = total[v] - c2v[c][j];
                }
            }
            word = hard(&total);
            if self.is_codeword(&word) {
                return Ok(LdpcDecoded { codeword: word, converged: true, iterations: iter });
            }
        }
        Ok(LdpcDecoded { codeword: word, converged: false, iterations: max_iterations })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn gaussian(&mut self) -> f64 {
            let (u1, u2) = (self.uniform().max(1e-300), self.uniform());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    fn hamming_7_4() -> LdpcCode {
        LdpcCode::new(7, vec![vec![0, 1, 2, 4], vec![0, 1, 3, 5], vec![0, 2, 3, 6]]).unwrap()
    }

    #[test]
    fn small_code_parameters_and_encoding() {
        let c = hamming_7_4();
        assert_eq!((c.n(), c.k(), c.num_checks()), (7, 4, 3));
        for v in 0..16u32 {
            let info: Vec<u8> = (0..4).map(|i| (v >> i & 1) as u8).collect();
            let cw = c.encode(&info).unwrap();
            assert!(c.is_codeword(&cw));
            assert_eq!(c.extract_info(&cw).unwrap(), info);
        }
    }

    #[test]
    fn decodes_single_confident_errors_on_small_code() {
        let c = hamming_7_4();
        let cw = c.encode(&[1, 0, 1, 1]).unwrap();
        // One flipped bit with a weak LLR among confident correct ones.
        let mut llr: Vec<f64> = cw.iter().map(|&b| if b == 0 { 4.0 } else { -4.0 }).collect();
        llr[2] = -llr[2].signum() * 1.0;
        let out = c.decode(&llr, 50).unwrap();
        assert!(out.converged);
        assert_eq!(out.codeword, cw);
    }

    #[test]
    fn gallager_code_over_awgn() {
        let code = LdpcCode::gallager(192, 3, 6, 11).unwrap();
        assert_eq!(code.n(), 192);
        assert!(code.k() >= 96 && code.k() <= 100, "k = {}", code.k());
        let mut rng = Rng(2024);
        let sigma = 0.55;
        let mut ok = 0;
        let mut uncoded_errors = 0;
        let trials = 40;
        for _ in 0..trials {
            let info: Vec<u8> = (0..code.k()).map(|_| (rng.next_u64() & 1) as u8).collect();
            let cw = code.encode(&info).unwrap();
            let rx: Vec<f64> = cw.iter().map(|&b| (1.0 - 2.0 * b as f64) + sigma * rng.gaussian()).collect();
            uncoded_errors += rx.iter().zip(&cw).filter(|(r, b)| (**r < 0.0) as u8 != **b).count();
            let llr: Vec<f64> = rx.iter().map(|&r| 2.0 * r / (sigma * sigma)).collect();
            let out = code.decode(&llr, 60).unwrap();
            if out.converged && out.codeword == cw {
                ok += 1;
            }
        }
        assert!(uncoded_errors > 0, "channel should be noisy enough to cause raw bit errors");
        assert!(ok >= trials - 3, "only {ok}/{trials} frames decoded");
    }

    #[test]
    fn validation() {
        assert!(LdpcCode::new(4, vec![vec![0, 9]]).is_err());
        assert!(LdpcCode::new(0, vec![vec![0]]).is_err());
        assert!(LdpcCode::gallager(100, 3, 6, 1).is_err());
        let c = hamming_7_4();
        assert!(c.encode(&[0; 3]).is_err());
        assert!(c.decode(&[0.0; 6], 10).is_err());
        assert!(!c.is_codeword(&[1, 0, 0, 0, 0, 0, 0]));
        // An all-zero channel observation with no information is not a convergence failure.
        let clean = c.decode(&[5.0; 7], 5).unwrap();
        assert!(clean.converged && clean.iterations == 0);
    }
}
