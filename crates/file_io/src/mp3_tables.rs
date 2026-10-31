//! Standard MPEG-1 Layer III constant tables: the polyphase window and the Huffman codes.
//!
//! The data lives in `data/layer3_tables.bin` and is embedded at compile time; see
//! `data/README.md` for its provenance (public-domain minimp3), the extraction script and how it
//! was verified. The file is parsed once, on first use.

use std::sync::OnceLock;

static BLOB: &[u8] = include_bytes!("../data/layer3_tables.bin");

/// One code of a "big values" Huffman table: the value pair `(x, y)` and its bit pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HuffmanEntry {
    pub x: u8,
    pub y: u8,
    /// Code length in bits (0 only for table 0, which carries no bits).
    pub length: u8,
    /// The code, right-aligned in `length` bits.
    pub code: u32,
}

/// A Layer III "big values" Huffman table.
#[derive(Clone, Debug)]
pub struct HuffmanTable {
    /// Number of extra bits following an escaped value of 15 (0 if the table has no escapes).
    pub linbits: u8,
    /// Number of distinct x values (`0..xlen`); the table covers an `xlen` by `ylen` grid.
    pub xlen: u8,
    pub ylen: u8,
    /// All codes, sorted by `(x, y)`.
    pub entries: Vec<HuffmanEntry>,
}

impl HuffmanTable {
    /// The code for the pair `(x, y)`, if the table contains it (`x`, `y` must already be
    /// clipped to 15 for escaped values).
    pub fn code_for(&self, x: u8, y: u8) -> Option<&HuffmanEntry> {
        if x >= self.xlen || y >= self.ylen {
            return None;
        }
        self.entries.get(x as usize * self.ylen as usize + y as usize).filter(|e| e.x == x && e.y == y)
    }

    /// True for the unused table numbers 4 and 14.
    pub fn is_unused(&self) -> bool {
        self.entries.is_empty()
    }
}

/// One code of a "count1" Huffman table: which of the four values (v, w, x, y) are ±1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Count1Entry {
    /// Bit 3 is `v`, bit 2 is `w`, bit 1 is `x`, bit 0 is `y`; a set bit means magnitude 1.
    pub values: u8,
    pub length: u8,
    pub code: u8,
}

struct Tables {
    window: [i32; 512],
    huffman: Vec<HuffmanTable>,
    count1: Vec<Vec<Count1Entry>>,
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| parse(BLOB).expect("embedded layer3_tables.bin is valid"))
}

fn parse(blob: &[u8]) -> Result<Tables, &'static str> {
    struct Cursor<'a>(&'a [u8], usize);
    impl Cursor<'_> {
        fn take(&mut self, n: usize) -> Result<&[u8], &'static str> {
            let s = self.0.get(self.1..self.1 + n).ok_or("table file is truncated")?;
            self.1 += n;
            Ok(s)
        }
        fn u8(&mut self) -> Result<u8, &'static str> {
            Ok(self.take(1)?[0])
        }
    }
    let mut c = Cursor(blob, 0);
    if c.take(4)? != b"L3TB" || c.u8()? != 1 {
        return Err("bad table file header");
    }
    let mut window = [0i32; 512];
    for w in window.iter_mut() {
        let b = c.take(4)?;
        *w = i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
    let count = c.u8()? as usize;
    let mut huffman = Vec::with_capacity(count);
    for _ in 0..count {
        let h = c.take(5)?;
        let (linbits, xlen, ylen, n) = (h[0], h[1], h[2], u16::from_le_bytes([h[3], h[4]]) as usize);
        let mut entries = Vec::with_capacity(n);
        for _ in 0..n {
            let e = c.take(7)?;
            entries.push(HuffmanEntry { x: e[0], y: e[1], length: e[2], code: u32::from_le_bytes([e[3], e[4], e[5], e[6]]) });
        }
        huffman.push(HuffmanTable { linbits, xlen, ylen, entries });
    }
    let count = c.u8()? as usize;
    let mut count1 = Vec::with_capacity(count);
    for _ in 0..count {
        let n = c.u8()? as usize;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let e = c.take(3)?;
            v.push(Count1Entry { values: e[0], length: e[1], code: e[2] });
        }
        count1.push(v);
    }
    if c.1 != blob.len() {
        return Err("trailing bytes in table file");
    }
    Ok(Tables { window, huffman, count1 })
}

/// The ISO 11172-3 synthesis window `D[i]`, `i = 0..512`, as floating point.
pub fn synthesis_window() -> [f64; 512] {
    let t = tables();
    std::array::from_fn(|i| t.window[i] as f64 / 65536.0)
}

/// The ISO 11172-3 analysis window `C[i] = D[i] / 32`, `i = 0..512`.
pub fn analysis_window() -> [f64; 512] {
    let d = synthesis_window();
    std::array::from_fn(|i| d[i] / 32.0)
}

/// Big-values Huffman table number `n` (`0..32`).
pub fn huffman_table(n: usize) -> Option<&'static HuffmanTable> {
    tables().huffman.get(n)
}

/// Count1 table A (`false`) or B (`true`).
pub fn count1_table(b: bool) -> &'static [Count1Entry] {
    &tables().count1[b as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    use lib_jsl_information::operators::sha256::{sha256, to_hex};

    #[test]
    fn embedded_file_matches_the_recorded_digest() {
        assert_eq!(to_hex(&sha256(BLOB)), "7b54755ce4bfb1e6eedbc6e50f8910edc1562966669a5dc958e30ff9eebc85a2");
    }

    #[test]
    fn window_properties() {
        let d = synthesis_window();
        assert_eq!(d[0], 0.0);
        assert_eq!(d[256], 75038.0 / 65536.0);
        assert!((d[256] - 1.144989014).abs() < 1e-9);
        assert_eq!(d[1], -1.0 / 65536.0);
        // D[512 - i] = -D[i], except at multiples of 64 where the sign pattern restores symmetry.
        for i in 1..512 {
            let want = if i % 64 == 0 { d[i] } else { -d[i] };
            assert_eq!(d[512 - i], want, "i = {i}");
        }
        let c = analysis_window();
        assert!((c[256] - 0.035780907).abs() < 1e-9);
    }

    /// ISO polyphase analysis followed by synthesis must reconstruct the input (delay 481).
    #[test]
    fn polyphase_filterbank_reconstructs() {
        let d = synthesis_window();
        let c = analysis_window();
        let n_blocks = 90;
        let x: Vec<f64> = (0..n_blocks * 32).map(|i| ((i * 7919 + 13) % 2003) as f64 - 1000.0).collect();
        let mut buf = [0.0f64; 512];
        let mut sub = Vec::new();
        for b in 0..n_blocks {
            for i in (32..512).rev() {
                buf[i] = buf[i - 32];
            }
            for i in 0..32 {
                buf[i] = x[b * 32 + 31 - i];
            }
            let mut y = [0.0f64; 64];
            for (i, yi) in y.iter_mut().enumerate() {
                *yi = (0..8).map(|j| buf[i + 64 * j] * c[i + 64 * j]).sum();
            }
            sub.push(std::array::from_fn::<f64, 32, _>(|k| {
                (0..64).map(|i| ((2 * k + 1) as f64 * (i as f64 - 16.0) * std::f64::consts::PI / 64.0).cos() * y[i]).sum()
            }));
        }
        let mut v = vec![0.0f64; 1024];
        let mut out = Vec::new();
        for s in &sub {
            v.copy_within(0..960, 64);
            for (i, vi) in v.iter_mut().enumerate().take(64) {
                *vi = (0..32).map(|k| ((16 + i) as f64 * (2 * k + 1) as f64 * std::f64::consts::PI / 64.0).cos() * s[k]).sum();
            }
            let mut u = [0.0f64; 512];
            for a in 0..8 {
                for j in 0..32 {
                    u[64 * a + j] = v[128 * a + j];
                    u[64 * a + 32 + j] = v[128 * a + 96 + j];
                }
            }
            for j in 0..32 {
                out.push((0..16).map(|a| u[j + 32 * a] * d[j + 32 * a]).sum::<f64>());
            }
        }
        let n = x.len() - 481 - 600;
        let err: f64 = (0..n).map(|i| (out[481 + i] - x[i]).powi(2)).sum::<f64>();
        let sig: f64 = (0..n).map(|i| x[i].powi(2)).sum();
        let db = 10.0 * (err / sig).log10();
        assert!(db < -80.0, "reconstruction error {db} dB");
    }

    #[test]
    fn huffman_tables_are_well_formed() {
        let dims = [(1usize, 2u8), (2, 3), (3, 3), (5, 4), (6, 4), (7, 6), (8, 6), (9, 6), (10, 8), (11, 8), (12, 8), (13, 16), (15, 16)];
        for n in 0..32 {
            let t = huffman_table(n).unwrap();
            if n == 4 || n == 14 {
                assert!(t.is_unused());
                continue;
            }
            if n == 0 {
                assert_eq!((t.entries.len(), t.entries[0].length), (1, 0));
                continue;
            }
            assert_eq!(t.entries.len(), t.xlen as usize * t.ylen as usize, "table {n}");
            // Every grid cell present exactly once, found through the (x, y) index.
            for x in 0..t.xlen {
                for y in 0..t.ylen {
                    assert!(t.code_for(x, y).is_some(), "table {n} missing ({x},{y})");
                }
            }
            // Prefix-free with Kraft sum at most 1.
            let mut kraft = 0.0;
            for (i, a) in t.entries.iter().enumerate() {
                kraft += 0.5f64.powi(a.length as i32);
                for b in &t.entries[i + 1..] {
                    let (short, long) = if a.length <= b.length { (a, b) } else { (b, a) };
                    assert_ne!(long.code >> (long.length - short.length), short.code, "table {n}: prefix clash");
                }
            }
            assert!(kraft <= 1.0 + 1e-12, "table {n} Kraft {kraft}");
        }
        for (n, size) in dims {
            let t = huffman_table(n).unwrap();
            assert_eq!((t.xlen, t.ylen), (size, size), "table {n}");
        }
        assert!(huffman_table(32).is_none());
    }

    #[test]
    fn known_codes_and_linbits() {
        // Table 1 (2x2): (0,0) = "1", (0,1) = "001", (1,0) = "01", (1,1) = "000".
        let t1 = huffman_table(1).unwrap();
        let get = |x, y| t1.code_for(x, y).map(|e| (e.code, e.length));
        assert_eq!(get(0, 0), Some((0b1, 1)));
        assert_eq!(get(0, 1), Some((0b001, 3)));
        assert_eq!(get(1, 0), Some((0b01, 2)));
        assert_eq!(get(1, 1), Some((0b000, 3)));
        let linbits: Vec<u8> = (0..32).map(|n| huffman_table(n).unwrap().linbits).collect();
        assert_eq!(linbits, vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 6, 8, 10, 13, 4, 5, 6, 7, 8, 9, 11, 13]);
        // Tables 16-23 share one code set, as do 24-31.
        for group in [16..24, 24..32] {
            let first = &huffman_table(group.start).unwrap().entries;
            for n in group {
                assert_eq!(&huffman_table(n).unwrap().entries, first);
            }
        }
    }

    #[test]
    fn count1_tables() {
        let a = count1_table(false);
        let b = count1_table(true);
        assert_eq!((a.len(), b.len()), (16, 16));
        // Table A: the all-zero quadruple is the single bit "1".
        assert!(a.iter().any(|e| e.values == 0 && e.length == 1 && e.code == 1));
        // Table B: fixed 4-bit codes, the bitwise complement of the value pattern.
        assert!(b.iter().all(|e| e.length == 4 && e.code == (!e.values & 0xF)));
        for t in [a, b] {
            let mut seen: Vec<u8> = t.iter().map(|e| e.values).collect();
            seen.sort_unstable();
            assert_eq!(seen, (0..16).collect::<Vec<u8>>());
            let kraft: f64 = t.iter().map(|e| 0.5f64.powi(e.length as i32)).sum();
            assert!((kraft - 1.0).abs() < 1e-12, "count1 tables are complete prefix codes");
        }
    }

    #[test]
    fn corrupt_files_are_rejected() {
        assert!(parse(&BLOB[..BLOB.len() - 1]).is_err());
        assert!(parse(b"nope").is_err());
        let mut bad = BLOB.to_vec();
        bad.push(0);
        assert!(parse(&bad).is_err());
    }
}
