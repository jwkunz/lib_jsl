//! DEFLATE (RFC 1951) and zlib (RFC 1950) compression, as used by PNG.
//!
//! The decoder handles all three block types (stored, fixed and dynamic Huffman). The encoder
//! finds matches with a hash-chain LZ77 search over the 32 KiB window and emits them with the
//! fixed Huffman code, or uses stored blocks at level 0.

use lib_jsl_core::ErrorsJSL;

const LENGTH_BASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LENGTH_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE: [u16; 30] = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const CODE_LENGTH_ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

fn corrupt(message: &'static str) -> ErrorsJSL {
    ErrorsJSL::RuntimeError(message)
}

/// The Adler-32 checksum used by zlib.
pub fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in data.chunks(5552) {
        for &byte in chunk {
            a += byte as u32;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

struct BitReader<'a> {
    data: &'a [u8],
    position: usize,
    bit_buffer: u32,
    bit_count: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader { data, position: 0, bit_buffer: 0, bit_count: 0 }
    }

    fn bits(&mut self, count: u32) -> Result<u32, ErrorsJSL> {
        while self.bit_count < count {
            let byte = *self.data.get(self.position).ok_or(corrupt("DEFLATE stream ended early."))?;
            self.position += 1;
            self.bit_buffer |= (byte as u32) << self.bit_count;
            self.bit_count += 8;
        }
        let value = self.bit_buffer & ((1u64 << count) - 1) as u32;
        self.bit_buffer >>= count;
        self.bit_count -= count;
        Ok(value)
    }

    fn align_to_byte(&mut self) {
        self.bit_buffer = 0;
        self.bit_count = 0;
    }
}

/// A canonical Huffman decoder in the counts/symbols form (as in zlib's `puff`).
struct Huffman {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Result<Self, ErrorsJSL> {
        let mut counts = [0u16; 16];
        lengths.iter().for_each(|&l| counts[l as usize] += 1);
        counts[0] = 0;
        // Reject over-subscribed codes (incomplete codes are allowed, as in zlib).
        let mut left: i32 = 1;
        for &c in &counts[1..] {
            left = 2 * left - c as i32;
            if left < 0 {
                return Err(corrupt("Over-subscribed Huffman code."));
            }
        }
        let mut offsets = [0u16; 16];
        for l in 1..15 {
            offsets[l + 1] = offsets[l] + counts[l];
        }
        let mut symbols = vec![0u16; lengths.len()];
        for (symbol, &l) in lengths.iter().enumerate() {
            if l != 0 {
                symbols[offsets[l as usize] as usize] = symbol as u16;
                offsets[l as usize] += 1;
            }
        }
        Ok(Huffman { counts, symbols })
    }

    fn decode(&self, reader: &mut BitReader) -> Result<u16, ErrorsJSL> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= reader.bits(1)? as i32;
            let count = self.counts[len] as i32;
            if code - count < first {
                return Ok(self.symbols[(index + code - first) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(corrupt("Invalid Huffman code."))
    }
}

fn fixed_tables() -> (Huffman, Huffman) {
    let mut lengths = [0u8; 288];
    lengths[..144].fill(8);
    lengths[144..256].fill(9);
    lengths[256..280].fill(7);
    lengths[280..].fill(8);
    (Huffman::new(&lengths).expect("valid"), Huffman::new(&[5u8; 30]).expect("valid"))
}

fn dynamic_tables(reader: &mut BitReader) -> Result<(Huffman, Huffman), ErrorsJSL> {
    let literal_count = reader.bits(5)? as usize + 257;
    let distance_count = reader.bits(5)? as usize + 1;
    let code_length_count = reader.bits(4)? as usize + 4;
    if literal_count > 286 || distance_count > 30 {
        return Err(corrupt("Too many Huffman codes."));
    }
    let mut code_lengths = [0u8; 19];
    for &i in CODE_LENGTH_ORDER.iter().take(code_length_count) {
        code_lengths[i] = reader.bits(3)? as u8;
    }
    let code_length_code = Huffman::new(&code_lengths)?;
    let mut lengths = vec![0u8; literal_count + distance_count];
    let mut i = 0;
    while i < lengths.len() {
        let symbol = code_length_code.decode(reader)?;
        let (value, repeat) = match symbol {
            0..=15 => (symbol as u8, 1),
            16 => (*lengths[..i].last().ok_or(corrupt("Repeat with no previous length."))?, 3 + reader.bits(2)? as usize),
            17 => (0, 3 + reader.bits(3)? as usize),
            _ => (0, 11 + reader.bits(7)? as usize),
        };
        if i + repeat > lengths.len() {
            return Err(corrupt("Code lengths overrun."));
        }
        lengths[i..i + repeat].fill(value);
        i += repeat;
    }
    if lengths[256] == 0 {
        return Err(corrupt("No end-of-block code."));
    }
    Ok((Huffman::new(&lengths[..literal_count])?, Huffman::new(&lengths[literal_count..])?))
}

/// Decompresses a raw DEFLATE stream.
pub fn inflate(data: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
    let mut reader = BitReader::new(data);
    let mut out = Vec::with_capacity(data.len() * 4);
    loop {
        let last = reader.bits(1)? == 1;
        match reader.bits(2)? {
            0 => {
                reader.align_to_byte();
                let p = reader.position;
                if p + 4 > data.len() {
                    return Err(corrupt("Truncated stored block."));
                }
                let len = u16::from_le_bytes([data[p], data[p + 1]]) as usize;
                let nlen = u16::from_le_bytes([data[p + 2], data[p + 3]]) as usize;
                if len != !nlen & 0xffff || p + 4 + len > data.len() {
                    return Err(corrupt("Corrupt stored block."));
                }
                out.extend_from_slice(&data[p + 4..p + 4 + len]);
                reader.position = p + 4 + len;
            }
            kind @ (1 | 2) => {
                let (literals, distances) = if kind == 1 { fixed_tables() } else { dynamic_tables(&mut reader)? };
                loop {
                    let symbol = literals.decode(&mut reader)? as usize;
                    match symbol {
                        0..=255 => out.push(symbol as u8),
                        256 => break,
                        257..=285 => {
                            let s = symbol - 257;
                            let length = LENGTH_BASE[s] as usize + reader.bits(LENGTH_EXTRA[s] as u32)? as usize;
                            let d = distances.decode(&mut reader)? as usize;
                            if d >= 30 {
                                return Err(corrupt("Invalid distance code."));
                            }
                            let distance = DIST_BASE[d] as usize + reader.bits(DIST_EXTRA[d] as u32)? as usize;
                            if distance > out.len() {
                                return Err(corrupt("Distance reaches before the start."));
                            }
                            let start = out.len() - distance;
                            for k in 0..length {
                                out.push(out[start + k]);
                            }
                        }
                        _ => return Err(corrupt("Invalid literal/length code.")),
                    }
                }
            }
            _ => return Err(corrupt("Invalid block type.")),
        }
        if last {
            return Ok(out);
        }
    }
}

struct BitWriter {
    out: Vec<u8>,
    buffer: u64,
    count: u32,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter { out: Vec::new(), buffer: 0, count: 0 }
    }

    /// Writes the low `count` bits of `value`, least significant first.
    fn bits(&mut self, value: u32, count: u32) {
        self.buffer |= (value as u64) << self.count;
        self.count += count;
        while self.count >= 8 {
            self.out.push(self.buffer as u8);
            self.buffer >>= 8;
            self.count -= 8;
        }
    }

    /// Writes a Huffman code, most significant bit first.
    fn code(&mut self, code: u32, length: u32) {
        let reversed = code.reverse_bits() >> (32 - length);
        self.bits(reversed, length);
    }

    fn finish(mut self) -> Vec<u8> {
        if self.count > 0 {
            self.out.push(self.buffer as u8);
        }
        self.out
    }
}

fn fixed_literal(writer: &mut BitWriter, symbol: usize) {
    match symbol {
        0..=143 => writer.code(0x30 + symbol as u32, 8),
        144..=255 => writer.code(0x190 + (symbol - 144) as u32, 9),
        256..=279 => writer.code((symbol - 256) as u32, 7),
        _ => writer.code(0xc0 + (symbol - 280) as u32, 8),
    }
}

fn write_match(writer: &mut BitWriter, length: usize, distance: usize) {
    let l = LENGTH_BASE.iter().rposition(|&b| b as usize <= length).expect("length >= 3");
    fixed_literal(writer, 257 + l);
    writer.bits((length - LENGTH_BASE[l] as usize) as u32, LENGTH_EXTRA[l] as u32);
    let d = DIST_BASE.iter().rposition(|&b| b as usize <= distance).expect("distance >= 1");
    writer.code(d as u32, 5);
    writer.bits((distance - DIST_BASE[d] as usize) as u32, DIST_EXTRA[d] as u32);
}

/// The data as stored (uncompressed) DEFLATE blocks.
fn stored(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 5 * (data.len() / 65535 + 1));
    let blocks: Vec<&[u8]> = if data.is_empty() { vec![&[][..]] } else { data.chunks(65535).collect() };
    for (i, block) in blocks.iter().enumerate() {
        out.push(if i + 1 == blocks.len() { 1 } else { 0 });
        out.extend_from_slice(&(block.len() as u16).to_le_bytes());
        out.extend_from_slice(&(!(block.len() as u16)).to_le_bytes());
        out.extend_from_slice(block);
    }
    out
}

/// Compresses to a raw DEFLATE stream. `level` 0 stores the data; any other level runs the LZ77
/// search, with longer hash chains (more compression, more time) for higher levels up to 9, and
/// falls back to stored blocks if the fixed Huffman code would make the data larger (as it does
/// for random bytes).
pub fn deflate(data: &[u8], level: u8) -> Vec<u8> {
    if level == 0 {
        return stored(data);
    }
    let compressed = deflate_fixed(data, level);
    if compressed.len() > data.len() + 5 * (data.len() / 65535 + 1) { stored(data) } else { compressed }
}

/// LZ77 with hash chains, coded with the fixed Huffman code in a single block.
fn deflate_fixed(data: &[u8], level: u8) -> Vec<u8> {
    const WINDOW: usize = 32768;
    const HASH_BITS: u32 = 15;
    let max_chain = 4usize << level.min(9);
    let hash = |i: usize| -> usize { (((data[i] as u32) << 16 | (data[i + 1] as u32) << 8 | data[i + 2] as u32).wrapping_mul(2_654_435_761) >> (32 - HASH_BITS)) as usize };
    let mut head = vec![usize::MAX; 1 << HASH_BITS];
    let mut previous = vec![usize::MAX; data.len()];
    let mut writer = BitWriter::new();
    writer.bits(1, 1); // final block
    writer.bits(1, 2); // fixed Huffman
    let mut i = 0;
    let insert = |head: &mut Vec<usize>, previous: &mut Vec<usize>, k: usize| {
        if k + 2 < data.len() {
            let h = hash(k);
            previous[k] = head[h];
            head[h] = k;
        }
    };
    while i < data.len() {
        let (mut best_length, mut best_distance) = (0, 0);
        if i + 2 < data.len() {
            let mut candidate = head[hash(i)];
            let mut chain = 0;
            while candidate != usize::MAX && i - candidate <= WINDOW && chain < max_chain {
                let limit = (data.len() - i).min(258);
                let mut length = 0;
                while length < limit && data[candidate + length] == data[i + length] {
                    length += 1;
                }
                if length > best_length {
                    best_length = length;
                    best_distance = i - candidate;
                    if length == limit {
                        break;
                    }
                }
                candidate = previous[candidate];
                chain += 1;
            }
        }
        if best_length >= 3 {
            write_match(&mut writer, best_length, best_distance);
            for k in i..i + best_length {
                insert(&mut head, &mut previous, k);
            }
            i += best_length;
        } else {
            fixed_literal(&mut writer, data[i] as usize);
            insert(&mut head, &mut previous, i);
            i += 1;
        }
    }
    fixed_literal(&mut writer, 256);
    writer.finish()
}

/// Decompresses a zlib stream (header, DEFLATE data, Adler-32 trailer), verifying the checksum.
pub fn zlib_decompress(data: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
    if data.len() < 6 {
        return Err(corrupt("zlib stream too short."));
    }
    let (cmf, flg) = (data[0], data[1]);
    if cmf & 0x0f != 8 || !((cmf as u16) << 8 | flg as u16).is_multiple_of(31) {
        return Err(corrupt("Invalid zlib header."));
    }
    if flg & 0x20 != 0 {
        return Err(corrupt("Preset zlib dictionaries are not supported."));
    }
    let out = inflate(&data[2..])?;
    let n = data.len();
    let expected = u32::from_be_bytes([data[n - 4], data[n - 3], data[n - 2], data[n - 1]]);
    if adler32(&out) != expected {
        return Err(corrupt("zlib checksum mismatch."));
    }
    Ok(out)
}

/// Compresses to a zlib stream (see [`deflate`] for `level`).
pub fn zlib_compress(data: &[u8], level: u8) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    out.extend(deflate(data, level));
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incompressible_data_is_stored_not_expanded() {
        let mut seed = 7u64;
        let random: Vec<u8> = (0..5000)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (seed >> 56) as u8
            })
            .collect();
        let packed = zlib_compress(&random, 9);
        assert!(packed.len() <= random.len() + 11, "{} bytes", packed.len());
        assert_eq!(zlib_decompress(&packed).unwrap(), random);
    }

    /// `zlib.compress(text, 9)` from CPython's zlib (dynamic Huffman blocks), where `text` is the
    /// pangram repeated 20 times followed by the bytes 0..=255.
    const ZLIB_DYNAMIC: &[u8] = &[
        120, 218, 11, 201, 72, 85, 40, 44, 205, 76, 206, 86, 72, 42, 202, 47, 207, 83, 72, 203, 175,
        80, 200, 42, 205, 45, 40, 86, 200, 47, 75, 45, 82, 40, 1, 74, 231, 36, 86, 85, 42, 164, 228,
        167, 235, 41, 132, 140, 42, 30, 85, 60, 170, 152, 218, 138, 25, 24, 153, 152, 89, 88, 217,
        216, 57, 56, 185, 184, 121, 120, 249, 248, 5, 4, 133, 132, 69, 68, 197, 196, 37, 36, 165,
        164, 101, 100, 229, 228, 21, 20, 149, 148, 85, 84, 213, 212, 53, 52, 181, 180, 117, 116,
        245, 244, 13, 12, 141, 140, 77, 76, 205, 204, 45, 44, 173, 172, 109, 108, 237, 236, 29, 28,
        157, 156, 93, 92, 221, 220, 61, 60, 189, 188, 125, 124, 253, 252, 3, 2, 131, 130, 67, 66,
        195, 194, 35, 34, 163, 162, 99, 98, 227, 226, 19, 18, 147, 146, 83, 82, 211, 210, 51, 50,
        179, 178, 115, 114, 243, 242, 11, 10, 139, 138, 75, 74, 203, 202, 43, 42, 171, 170, 107,
        106, 235, 234, 27, 26, 155, 154, 91, 90, 219, 218, 59, 58, 187, 186, 123, 122, 251, 250, 39,
        76, 156, 52, 121, 202, 212, 105, 211, 103, 204, 156, 53, 123, 206, 220, 121, 243, 23, 44,
        92, 180, 120, 201, 210, 101, 203, 87, 172, 92, 181, 122, 205, 218, 117, 235, 55, 108, 220,
        180, 121, 203, 214, 109, 219, 119, 236, 220, 181, 123, 207, 222, 125, 251, 15, 28, 60, 116,
        248, 200, 209, 99, 199, 79, 156, 60, 117, 250, 204, 217, 115, 231, 47, 92, 188, 116, 249,
        202, 213, 107, 215, 111, 220, 188, 117, 251, 206, 221, 123, 247, 31, 60, 124, 244, 248, 201,
        211, 103, 207, 95, 188, 124, 245, 250, 205, 219, 119, 239, 63, 124, 252, 244, 249, 203, 215,
        111, 223, 127, 252, 252, 245, 251, 207, 223, 127, 255, 1, 20, 151, 194, 156,
    ];

    fn text() -> Vec<u8> {
        let mut t = b"The quick brown fox jumps over the lazy dog. ".repeat(20);
        t.extend(0..=255u8);
        t
    }

    #[test]
    fn decodes_a_dynamic_huffman_stream_from_zlib() {
        assert_eq!(zlib_decompress(ZLIB_DYNAMIC).unwrap(), text());
        // A corrupted checksum is detected.
        let mut bad = ZLIB_DYNAMIC.to_vec();
        let n = bad.len();
        bad[n - 1] ^= 1;
        assert!(zlib_decompress(&bad).is_err());
        assert!(zlib_decompress(&ZLIB_DYNAMIC[..20]).is_err());
    }

    #[test]
    fn round_trips_at_every_level_and_compresses_redundant_data() {
        let mut inputs = vec![Vec::new(), vec![7u8], text(), vec![0u8; 100_000]];
        let mut s: u32 = 1;
        inputs.push((0..70_000).map(|_| { s = s.wrapping_mul(1_103_515_245).wrapping_add(12345); (s >> 24) as u8 }).collect());
        for input in &inputs {
            for level in [0, 1, 6, 9] {
                let z = zlib_compress(input, level);
                assert_eq!(&zlib_decompress(&z).unwrap(), input, "level {level}, {} bytes", input.len());
            }
        }
        assert!(zlib_compress(&text(), 6).len() < text().len() / 2);
        assert!(zlib_compress(&vec![0u8; 100_000], 6).len() < 1000);
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }
}
