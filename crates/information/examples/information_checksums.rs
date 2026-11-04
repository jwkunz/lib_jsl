//! CRC-32 checksums and SHA-256 digests, in one call or streamed.

use lib_jsl_information::operators::{Crc32, Sha256, crc32, crc32c, sha256, to_hex};

fn main() {
    let data = b"The quick brown fox jumps over the lazy dog";
    println!("CRC-32  {:08x}", crc32(data));
    println!("CRC-32C {:08x}", crc32c(data));
    println!("SHA-256 {}", to_hex(&sha256(data)));
    // Streaming gives the same results, however the data is split.
    let (mut c, mut h) = (Crc32::new(), Sha256::new());
    for chunk in data.chunks(7) {
        c.update(chunk);
        h.update(chunk);
    }
    println!("streamed: {:08x} {}", c.finalize(), &to_hex(&h.finalize())[..16]);
    // One changed letter changes the checksum and about half of the digest's bits.
    let altered = b"The quick brown fox jumps over the lazy cog";
    let (d1, d2) = (sha256(data), sha256(altered));
    let changed: u32 = d1.iter().zip(&d2).map(|(a, b)| (a ^ b).count_ones()).sum();
    println!("\"dog\" -> \"cog\": CRC-32 {:08x}, {changed} of 256 digest bits changed", crc32(altered));
}
