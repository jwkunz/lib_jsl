//! CRC-32 checksums.

use std::sync::OnceLock;

/// Reflected polynomial of the IEEE 802.3 CRC-32 (used by Ethernet, zip, PNG, gzip).
pub const POLY_IEEE: u32 = 0xEDB8_8320;
/// Reflected polynomial of CRC-32C (Castagnoli; used by iSCSI, SCTP, ext4).
pub const POLY_CASTAGNOLI: u32 = 0x82F6_3B78;

fn build_table(poly: u32) -> [u32; 256] {
    let mut table = [0u32; 256];
    for (i, entry) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { (c >> 1) ^ poly } else { c >> 1 };
        }
        *entry = c;
    }
    table
}

fn table(poly: u32) -> &'static [u32; 256] {
    static IEEE: OnceLock<[u32; 256]> = OnceLock::new();
    static CASTAGNOLI: OnceLock<[u32; 256]> = OnceLock::new();
    match poly {
        POLY_IEEE => IEEE.get_or_init(|| build_table(POLY_IEEE)),
        _ => CASTAGNOLI.get_or_init(|| build_table(POLY_CASTAGNOLI)),
    }
}

/// An incremental CRC-32 calculator.
#[derive(Clone, Debug)]
pub struct Crc32 {
    state: u32,
    poly: u32,
}

impl Crc32 {
    /// A calculator for the IEEE 802.3 CRC-32.
    pub fn new() -> Self {
        Crc32 { state: 0xFFFF_FFFF, poly: POLY_IEEE }
    }

    /// A calculator for CRC-32C (Castagnoli).
    pub fn castagnoli() -> Self {
        Crc32 { state: 0xFFFF_FFFF, poly: POLY_CASTAGNOLI }
    }

    /// Feeds more bytes into the checksum.
    pub fn update(&mut self, data: &[u8]) {
        let t = table(self.poly);
        for &b in data {
            self.state = t[((self.state ^ b as u32) & 0xFF) as usize] ^ (self.state >> 8);
        }
    }

    /// Returns the checksum of everything fed so far (the calculator may continue to be used).
    pub fn finalize(&self) -> u32 {
        !self.state
    }
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

/// The IEEE 802.3 CRC-32 of `data`.
pub fn crc32(data: &[u8]) -> u32 {
    let mut c = Crc32::new();
    c.update(data);
    c.finalize()
}

/// The CRC-32C (Castagnoli) of `data`.
pub fn crc32c(data: &[u8]) -> u32 {
    let mut c = Crc32::castagnoli();
    c.update(data);
    c.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_values() {
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b"The quick brown fox jumps over the lazy dog"), 0x414F_A339);
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c(&[0u8; 32]), 0x8A91_36AA);
    }

    #[test]
    fn incremental_matches_one_shot() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 31 % 251) as u8).collect();
        let mut c = Crc32::new();
        for chunk in data.chunks(77) {
            c.update(chunk);
        }
        assert_eq!(c.finalize(), crc32(&data));
    }

    #[test]
    fn appending_crc_gives_residue() {
        // Appending the little-endian CRC to the message yields the constant residue 0x2144DF1C.
        let mut msg = b"hello world".to_vec();
        msg.extend_from_slice(&crc32(&msg).to_le_bytes());
        assert_eq!(crc32(&msg), 0x2144_DF1C);
    }
}
