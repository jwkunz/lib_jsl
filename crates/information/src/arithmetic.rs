//! Manual arithmetic compression and decompression helpers.
//!
//! Arithmetic coding is another entropy coder, but unlike Huffman coding it
//! does not assign one discrete bit-pattern per symbol. Instead, it keeps a
//! numeric interval and progressively narrows that interval as each symbol is
//! processed. Frequent symbols claim large sub-intervals and rare symbols claim
//! small ones, which lets the coder represent symbol probabilities more
//! precisely than a prefix tree.
//!
//! # Format Overview
//!
//! The compressed stream begins with the ASCII header `ARI0`. It is followed by
//! everything the decoder needs to rebuild the exact same static probability
//! model:
//!
//! - 4-byte magic header: `ARI0`
//! - 8-byte little-endian original length
//! - 2-byte little-endian symbol count
//! - repeated `symbol_count` times:
//!   - 1 byte: symbol value
//!   - 4 bytes little-endian: symbol frequency
//! - 8-byte little-endian payload bit length
//! - packed arithmetic-code bits in most-significant-bit-first order
//!
//! The implementation below uses a classic fixed-precision arithmetic coder
//! with E1/E2/E3 renormalization. The comments in the hot paths explain how the
//! interval is updated and why bits are emitted or consumed when the interval
//! moves wholly into the low half, high half, or middle band.

use lib_jsl_core::ErrorsJSL;

const MAGIC: &[u8; 4] = b"ARI0";
const CODE_VALUE_BITS: u32 = 32;
const TOP_VALUE: u64 = (1u64 << CODE_VALUE_BITS) - 1;
const FIRST_QTR: u64 = TOP_VALUE / 4 + 1;
const HALF: u64 = 2 * FIRST_QTR;
const THIRD_QTR: u64 = 3 * FIRST_QTR;

#[derive(Clone, Debug)]
struct Model {
    frequencies: [u32; 256],
    cumulative: [u64; 257],
    total: u64,
}

/// Compress a byte slice with a manual arithmetic coder.
pub fn arithmetic_compress(input: &[u8]) -> Vec<u8> {
    try_arithmetic_compress(input).unwrap_or_default()
}

/// Fallible variant of [`arithmetic_compress`].
pub fn try_arithmetic_compress(input: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
    if input.len() > u64::MAX as usize {
        return Err(ErrorsJSL::InvalidInputRange("Arithmetic coder input is too large to store its original length."));
    }

    let model = build_model(input)?;
    let symbol_count = model.frequencies.iter().filter(|&&count| count > 0).count();

    let mut output = Vec::new();
    output.extend_from_slice(MAGIC);
    output.extend_from_slice(&(input.len() as u64).to_le_bytes());
    output.extend_from_slice(&(symbol_count as u16).to_le_bytes());

    // Store the static model explicitly so decompression can reconstruct the
    // same cumulative frequency table before it starts decoding bits.
    for (symbol, &frequency) in model.frequencies.iter().enumerate() {
        if frequency > 0 {
            output.push(symbol as u8);
            output.extend_from_slice(&frequency.to_le_bytes());
        }
    }

    if input.is_empty() {
        output.extend_from_slice(&0u64.to_le_bytes());
        return Ok(output);
    }

    let mut writer = BitWriter::default();
    let mut low = 0u64;
    let mut high = TOP_VALUE;
    let mut pending_bits = 0u64;

    for &symbol in input {
        let symbol_index = symbol as usize;
        let range = (high - low + 1) as u128;
        let cum_low = model.cumulative[symbol_index] as u128;
        let cum_high = model.cumulative[symbol_index + 1] as u128;
        let total = model.total as u128;

        // Narrow the current interval down to the slice owned by this symbol.
        high = low + ((range * cum_high) / total - 1) as u64;
        low = low + ((range * cum_low) / total) as u64;

        // Renormalize whenever the interval has moved entirely into the low
        // half, high half, or the middle "underflow" band.
        loop {
            if high < HALF {
                output_bit_plus_pending(&mut writer, false, &mut pending_bits);
            } else if low >= HALF {
                output_bit_plus_pending(&mut writer, true, &mut pending_bits);
                low -= HALF;
                high -= HALF;
            } else if low >= FIRST_QTR && high < THIRD_QTR {
                pending_bits += 1;
                low -= FIRST_QTR;
                high -= FIRST_QTR;
            } else {
                break;
            }

            low = low.saturating_mul(2);
            high = high.saturating_mul(2).saturating_add(1);
        }
    }

    // Emit one final disambiguating bit plus any pending opposite bits so the
    // decoder can safely land inside the final interval.
    pending_bits += 1;
    if low < FIRST_QTR {
        output_bit_plus_pending(&mut writer, false, &mut pending_bits);
    } else {
        output_bit_plus_pending(&mut writer, true, &mut pending_bits);
    }

    let (payload_bytes, payload_bits) = writer.finish();
    output.extend_from_slice(&payload_bits.to_le_bytes());
    output.extend_from_slice(&payload_bytes);
    Ok(output)
}

/// Decompress a byte slice previously produced by [`arithmetic_compress`].
pub fn arithmetic_decompress(input: &[u8]) -> Vec<u8> {
    try_arithmetic_decompress(input).unwrap_or_default()
}

/// Fallible variant of [`arithmetic_decompress`].
pub fn try_arithmetic_decompress(input: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
    if input.len() < MAGIC.len() || &input[..MAGIC.len()] != MAGIC {
        return Err(ErrorsJSL::InvalidInputRange("Compressed payload is missing the arithmetic coder header."));
    }

    let mut cursor = MAGIC.len();
    if cursor + 8 + 2 > input.len() {
        return Err(ErrorsJSL::InvalidInputRange("Arithmetic coder header is truncated before length or symbol count."));
    }

    let original_len = u64::from_le_bytes(
        input[cursor..cursor + 8]
            .try_into()
            .map_err(|_| ErrorsJSL::RuntimeError("Failed to parse arithmetic coder original length field."))?,
    ) as usize;
    cursor += 8;

    let symbol_count = u16::from_le_bytes(
        input[cursor..cursor + 2]
            .try_into()
            .map_err(|_| ErrorsJSL::RuntimeError("Failed to parse arithmetic coder symbol count field."))?,
    ) as usize;
    cursor += 2;

    let table_bytes = symbol_count
        .checked_mul(5)
        .ok_or(ErrorsJSL::InvalidInputRange("Arithmetic coder frequency table size overflowed."))?;
    if cursor + table_bytes + 8 > input.len() {
        return Err(ErrorsJSL::InvalidInputRange("Arithmetic coder header extends beyond the compressed payload."));
    }

    let mut frequencies = [0u32; 256];
    for _ in 0..symbol_count {
        let symbol = input[cursor];
        let frequency = u32::from_le_bytes(
            input[cursor + 1..cursor + 5]
                .try_into()
                .map_err(|_| ErrorsJSL::RuntimeError("Failed to parse arithmetic coder symbol frequency field."))?,
        );
        cursor += 5;

        if frequency == 0 {
            return Err(ErrorsJSL::InvalidInputRange("Arithmetic coder frequency table cannot contain zero-frequency entries."));
        }
        if frequencies[symbol as usize] != 0 {
            return Err(ErrorsJSL::InvalidInputRange("Arithmetic coder frequency table contains duplicate symbol entries."));
        }

        frequencies[symbol as usize] = frequency;
    }

    let payload_bits = u64::from_le_bytes(
        input[cursor..cursor + 8]
            .try_into()
            .map_err(|_| ErrorsJSL::RuntimeError("Failed to parse arithmetic coder payload bit length field."))?,
    );
    cursor += 8;

    let payload = &input[cursor..];
    let available_bits = (payload.len() as u64)
        .checked_mul(8)
        .ok_or(ErrorsJSL::InvalidInputRange("Arithmetic coder payload size overflowed while checking bit capacity."))?;
    if payload_bits > available_bits {
        return Err(ErrorsJSL::InvalidInputRange("Arithmetic coder payload bit length exceeds the available payload bytes."));
    }

    if original_len == 0 {
        return Ok(Vec::new());
    }

    let model = build_model_from_frequencies(frequencies)?;
    if model.total as usize != original_len {
        return Err(ErrorsJSL::InvalidInputRange("Arithmetic coder frequency table does not match the stored original length."));
    }

    let mut reader = BitReader::new(payload, payload_bits);
    let mut low = 0u64;
    let mut high = TOP_VALUE;
    let mut code = 0u64;

    // Prime the decoder with an initial fixed-width code value assembled from
    // the front of the packed payload. Missing tail bits are interpreted as
    // zero, which matches how the encoder pads its final byte.
    for _ in 0..CODE_VALUE_BITS {
        code = (code << 1) | reader.read_bit_or_zero();
    }

    let mut output = Vec::with_capacity(original_len);

    while output.len() < original_len {
        let range = (high - low + 1) as u128;
        let total = model.total as u128;

        // Map the current code value back into the model's cumulative frequency
        // space. This tells us which symbol sub-interval contains the code.
        let scaled_value =
            (((code - low + 1) as u128 * total - 1) / range) as u64;

        let symbol = find_symbol(&model.cumulative, scaled_value)?;
        output.push(symbol as u8);

        let cum_low = model.cumulative[symbol] as u128;
        let cum_high = model.cumulative[symbol + 1] as u128;

        high = low + ((range * cum_high) / total - 1) as u64;
        low = low + ((range * cum_low) / total) as u64;

        loop {
            if high < HALF {
                // Nothing to subtract; the interval already sits in the low half.
            } else if low >= HALF {
                code -= HALF;
                low -= HALF;
                high -= HALF;
            } else if low >= FIRST_QTR && high < THIRD_QTR {
                code -= FIRST_QTR;
                low -= FIRST_QTR;
                high -= FIRST_QTR;
            } else {
                break;
            }

            low = low.saturating_mul(2);
            high = high.saturating_mul(2).saturating_add(1);
            code = code.saturating_mul(2).saturating_add(reader.read_bit_or_zero());
        }
    }

    Ok(output)
}

fn build_model(input: &[u8]) -> Result<Model, ErrorsJSL> {
    let mut frequencies = [0u32; 256];
    for &byte in input {
        frequencies[byte as usize] = frequencies[byte as usize]
            .checked_add(1)
            .ok_or(ErrorsJSL::InvalidInputRange("Arithmetic coder frequency overflowed while counting symbols."))?;
    }
    build_model_from_frequencies(frequencies)
}

fn build_model_from_frequencies(frequencies: [u32; 256]) -> Result<Model, ErrorsJSL> {
    let mut cumulative = [0u64; 257];
    for (index, &frequency) in frequencies.iter().enumerate() {
        cumulative[index + 1] = cumulative[index]
            .checked_add(frequency as u64)
            .ok_or(ErrorsJSL::InvalidInputRange("Arithmetic coder cumulative frequency overflowed."))?;
    }

    Ok(Model {
        frequencies,
        cumulative,
        total: cumulative[256],
    })
}

fn find_symbol(cumulative: &[u64; 257], scaled_value: u64) -> Result<usize, ErrorsJSL> {
    let mut low = 0usize;
    let mut high = 256usize;

    while low < high {
        let mid = (low + high) / 2;
        if cumulative[mid + 1] <= scaled_value {
            low = mid + 1;
        } else if cumulative[mid] > scaled_value {
            high = mid;
        } else {
            return Ok(mid);
        }
    }

    Err(ErrorsJSL::InvalidInputRange("Arithmetic coder could not map the current code value to a symbol interval."))
}

fn output_bit_plus_pending(writer: &mut BitWriter, bit: bool, pending_bits: &mut u64) {
    writer.write_bit(bit);
    while *pending_bits > 0 {
        writer.write_bit(!bit);
        *pending_bits -= 1;
    }
}

#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    current_byte: u8,
    bits_in_current_byte: u8,
    bits_written: u64,
}

impl BitWriter {
    fn write_bit(&mut self, bit: bool) {
        self.current_byte <<= 1;
        if bit {
            self.current_byte |= 1;
        }
        self.bits_in_current_byte += 1;
        self.bits_written += 1;

        if self.bits_in_current_byte == 8 {
            self.bytes.push(self.current_byte);
            self.current_byte = 0;
            self.bits_in_current_byte = 0;
        }
    }

    fn finish(mut self) -> (Vec<u8>, u64) {
        if self.bits_in_current_byte > 0 {
            self.current_byte <<= 8 - self.bits_in_current_byte;
            self.bytes.push(self.current_byte);
        }
        (self.bytes, self.bits_written)
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    total_bits: u64,
    bit_position: u64,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8], total_bits: u64) -> Self {
        Self {
            bytes,
            total_bits,
            bit_position: 0,
        }
    }

    fn read_bit_or_zero(&mut self) -> u64 {
        if self.bit_position >= self.total_bits {
            return 0;
        }

        let byte_index = (self.bit_position / 8) as usize;
        let bit_index = 7 - (self.bit_position % 8) as u8;
        self.bit_position += 1;
        ((self.bytes[byte_index] >> bit_index) & 1) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_empty_input() {
        let payload = b"";
        let compressed = try_arithmetic_compress(payload).unwrap();
        let decompressed = try_arithmetic_decompress(&compressed).unwrap();

        assert_eq!(decompressed, payload);
        assert_eq!(&compressed[..MAGIC.len()], MAGIC);
    }

    #[test]
    fn round_trip_repeated_text() {
        let payload = b"BANANA_BANDANA_BANANA_BANDANA";
        let compressed = try_arithmetic_compress(payload).unwrap();
        let decompressed = try_arithmetic_decompress(&compressed).unwrap();

        assert_eq!(decompressed, payload);
    }

    #[test]
    fn round_trip_binary_data() {
        let payload = [0u8, 1, 1, 2, 3, 5, 8, 13, 13, 13, 21, 34, 55];
        let compressed = try_arithmetic_compress(&payload).unwrap();
        let decompressed = try_arithmetic_decompress(&compressed).unwrap();

        assert_eq!(decompressed, payload);
    }

    #[test]
    fn round_trip_single_symbol_input() {
        let payload = [9u8; 2048];
        let compressed = try_arithmetic_compress(&payload).unwrap();
        let decompressed = try_arithmetic_decompress(&compressed).unwrap();

        assert_eq!(decompressed, payload);
    }

    #[test]
    fn round_trip_kjv_text_asset() {
        let payload = include_bytes!("../test_assets/kjv.txt");
        let compressed = try_arithmetic_compress(payload).unwrap();
        let decompressed = try_arithmetic_decompress(&compressed).unwrap();

        assert_eq!(decompressed, payload);
    }

    #[test]
    fn malformed_stream_returns_error() {
        let malformed = vec![b'A', b'R', b'I', b'0', 0x01];
        let result = try_arithmetic_decompress(&malformed);

        assert!(result.is_err());
    }

    #[test]
    fn wrapper_returns_empty_on_malformed_stream() {
        let malformed = vec![0x01, 0x02, 0x03];
        let decompressed = arithmetic_decompress(&malformed);

        assert!(decompressed.is_empty());
    }
}
