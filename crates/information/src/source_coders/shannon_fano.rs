//! Manual Shannon-Fano compression and decompression helpers.
//!
//! Shannon-Fano coding is a classic entropy coder that predates Huffman coding.
//! It starts from symbol frequencies, sorts symbols by descending probability,
//! and then recursively splits the list into two groups whose total weights are
//! as balanced as possible. The left group receives a `0` bit and the right
//! group receives a `1` bit. Repeating that process produces a prefix-free code
//! table that can be used to pack the input into bits.
//!
//! # Format Overview
//!
//! The compressed stream begins with the ASCII header `SFAN`. It is followed by
//! a stored static frequency table and then the packed payload bits:
//!
//! - 4-byte magic header: `SFAN`
//! - 8-byte little-endian original length
//! - 2-byte little-endian symbol count
//! - repeated `symbol_count` times:
//!   - 1 byte: symbol value
//!   - 4 bytes little-endian: symbol frequency
//! - packed compressed bits in most-significant-bit-first order
//!
//! As with the Huffman and arithmetic coders in this crate, the decoder is
//! fully self-contained because it rebuilds the exact same model from the
//! stored frequency table.

use lib_jsl_core::ErrorsJSL;

const MAGIC: &[u8; 4] = b"SFAN";

#[derive(Clone, Copy, Debug)]
struct SymbolFrequency {
    symbol: u8,
    frequency: u32,
}

#[derive(Clone, Debug)]
enum DecodeNode {
    Leaf { symbol: u8 },
    Internal {
        left: Option<Box<DecodeNode>>,
        right: Option<Box<DecodeNode>>,
    },
}

impl DecodeNode {
    fn new_internal() -> Self {
        Self::Internal {
            left: None,
            right: None,
        }
    }
}

/// Compress a byte slice with a manual Shannon-Fano encoder.
pub fn shannon_fano_compress(input: &[u8]) -> Vec<u8> {
    try_shannon_fano_compress(input).unwrap_or_default()
}

/// Fallible variant of [`shannon_fano_compress`].
pub fn try_shannon_fano_compress(input: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
    if input.len() > u64::MAX as usize {
        return Err(ErrorsJSL::InvalidInputRange("Shannon-Fano input is too large to encode its original length."));
    }

    let frequencies = build_frequency_table(input);
    let symbol_count = frequencies.iter().filter(|&&count| count > 0).count();
    if symbol_count > u16::MAX as usize {
        return Err(ErrorsJSL::InvalidInputRange("Shannon-Fano frequency table has too many distinct symbols."));
    }

    let mut output = Vec::new();
    output.extend_from_slice(MAGIC);
    output.extend_from_slice(&(input.len() as u64).to_le_bytes());
    output.extend_from_slice(&(symbol_count as u16).to_le_bytes());

    for (symbol, &frequency) in frequencies.iter().enumerate() {
        if frequency > 0 {
            output.push(symbol as u8);
            output.extend_from_slice(&frequency.to_le_bytes());
        }
    }

    if input.is_empty() {
        return Ok(output);
    }

    let mut symbols = sorted_symbols(&frequencies);
    let mut codes = vec![Vec::new(); 256];
    assign_codes(&mut symbols, &mut codes);

    let mut current_byte = 0u8;
    let mut bit_count = 0u8;

    for &byte in input {
        for &bit in &codes[byte as usize] {
            current_byte <<= 1;
            if bit {
                current_byte |= 1;
            }
            bit_count += 1;

            if bit_count == 8 {
                output.push(current_byte);
                current_byte = 0;
                bit_count = 0;
            }
        }
    }

    if bit_count > 0 {
        current_byte <<= 8 - bit_count;
        output.push(current_byte);
    }

    Ok(output)
}

/// Decompress a byte slice previously produced by [`shannon_fano_compress`].
pub fn shannon_fano_decompress(input: &[u8]) -> Vec<u8> {
    try_shannon_fano_decompress(input).unwrap_or_default()
}

/// Fallible variant of [`shannon_fano_decompress`].
pub fn try_shannon_fano_decompress(input: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
    if input.len() < MAGIC.len() || &input[..MAGIC.len()] != MAGIC {
        return Err(ErrorsJSL::InvalidInputRange("Compressed payload is missing the Shannon-Fano header."));
    }

    let mut cursor = MAGIC.len();
    if cursor + 8 + 2 > input.len() {
        return Err(ErrorsJSL::InvalidInputRange("Shannon-Fano header is truncated before length or symbol count."));
    }

    let original_len = u64::from_le_bytes(
        input[cursor..cursor + 8]
            .try_into()
            .map_err(|_| ErrorsJSL::RuntimeError("Failed to parse Shannon-Fano original length field."))?,
    ) as usize;
    cursor += 8;

    let symbol_count = u16::from_le_bytes(
        input[cursor..cursor + 2]
            .try_into()
            .map_err(|_| ErrorsJSL::RuntimeError("Failed to parse Shannon-Fano symbol count field."))?,
    ) as usize;
    cursor += 2;

    let table_bytes = symbol_count
        .checked_mul(5)
        .ok_or(ErrorsJSL::InvalidInputRange("Shannon-Fano frequency table size overflowed."))?;
    if cursor + table_bytes > input.len() {
        return Err(ErrorsJSL::InvalidInputRange("Shannon-Fano frequency table extends beyond the compressed payload."));
    }

    let mut frequencies = [0u32; 256];
    for _ in 0..symbol_count {
        let symbol = input[cursor];
        let frequency = u32::from_le_bytes(
            input[cursor + 1..cursor + 5]
                .try_into()
                .map_err(|_| ErrorsJSL::RuntimeError("Failed to parse Shannon-Fano symbol frequency field."))?,
        );
        cursor += 5;

        if frequency == 0 {
            return Err(ErrorsJSL::InvalidInputRange("Shannon-Fano frequency table cannot contain zero-frequency entries."));
        }
        if frequencies[symbol as usize] != 0 {
            return Err(ErrorsJSL::InvalidInputRange("Shannon-Fano frequency table contains duplicate symbol entries."));
        }

        frequencies[symbol as usize] = frequency;
    }

    if original_len == 0 {
        return Ok(Vec::new());
    }

    let total_frequency: usize = frequencies.iter().map(|&count| count as usize).sum();
    if total_frequency != original_len {
        return Err(ErrorsJSL::InvalidInputRange("Shannon-Fano frequency table does not match the stored original length."));
    }

    let mut symbols = sorted_symbols(&frequencies);
    let mut codes = vec![Vec::new(); 256];
    assign_codes(&mut symbols, &mut codes);
    let decode_root = build_decode_tree(&codes, &frequencies)?;

    if let DecodeNode::Leaf { symbol } = decode_root {
        return Ok(vec![symbol; original_len]);
    }

    let bitstream = &input[cursor..];
    let mut output = Vec::with_capacity(original_len);
    let mut current = &decode_root;

    for &packed_byte in bitstream {
        for shift in (0..8).rev() {
            let bit_is_set = ((packed_byte >> shift) & 1) == 1;

            current = match current {
                DecodeNode::Leaf { symbol } => {
                    output.push(*symbol);
                    if output.len() == original_len {
                        return Ok(output);
                    }
                    descend_from_root(&decode_root, bit_is_set)?
                }
                DecodeNode::Internal { left, right } => {
                    if bit_is_set {
                        right.as_deref().ok_or(ErrorsJSL::InvalidInputRange(
                            "Shannon-Fano bitstream referenced a missing right branch.",
                        ))?
                    } else {
                        left.as_deref().ok_or(ErrorsJSL::InvalidInputRange(
                            "Shannon-Fano bitstream referenced a missing left branch.",
                        ))?
                    }
                }
            };

            if let DecodeNode::Leaf { symbol } = current {
                output.push(*symbol);
                if output.len() == original_len {
                    return Ok(output);
                }
                current = &decode_root;
            }
        }
    }

    Err(ErrorsJSL::InvalidInputRange(
        "Shannon-Fano bitstream ended before the declared number of output bytes was reconstructed.",
    ))
}

fn descend_from_root<'a>(root: &'a DecodeNode, bit_is_set: bool) -> Result<&'a DecodeNode, ErrorsJSL> {
    match root {
        DecodeNode::Internal { left, right } => {
            if bit_is_set {
                right
                    .as_deref()
                    .ok_or(ErrorsJSL::InvalidInputRange("Shannon-Fano root is missing its right branch."))
            } else {
                left
                    .as_deref()
                    .ok_or(ErrorsJSL::InvalidInputRange("Shannon-Fano root is missing its left branch."))
            }
        }
        DecodeNode::Leaf { .. } => Err(ErrorsJSL::RuntimeError(
            "Tried to descend from a leaf-only Shannon-Fano tree.",
        )),
    }
}

fn build_frequency_table(input: &[u8]) -> [u32; 256] {
    let mut frequencies = [0u32; 256];
    for &byte in input {
        frequencies[byte as usize] = frequencies[byte as usize].saturating_add(1);
    }
    frequencies
}

fn sorted_symbols(frequencies: &[u32; 256]) -> Vec<SymbolFrequency> {
    let mut symbols = Vec::new();
    for (symbol, &frequency) in frequencies.iter().enumerate() {
        if frequency > 0 {
            symbols.push(SymbolFrequency {
                symbol: symbol as u8,
                frequency,
            });
        }
    }

    // Shannon-Fano needs a deterministic ordering before the recursive splits
    // begin. Higher frequencies come first, and symbol value breaks ties.
    symbols.sort_by(|left, right| {
        right
            .frequency
            .cmp(&left.frequency)
            .then_with(|| left.symbol.cmp(&right.symbol))
    });
    symbols
}

fn assign_codes(symbols: &mut [SymbolFrequency], codes: &mut [Vec<bool>]) {
    if symbols.is_empty() {
        return;
    }
    assign_codes_recursive(symbols, codes);
}

fn assign_codes_recursive(symbols: &mut [SymbolFrequency], codes: &mut [Vec<bool>]) {
    if symbols.len() == 1 {
        // A one-symbol input still needs a code path so the bit-packing logic
        // has something concrete to use.
        if codes[symbols[0].symbol as usize].is_empty() {
            codes[symbols[0].symbol as usize].push(false);
        }
        return;
    }

    let split_index = choose_split_index(symbols);

    for entry in &symbols[..split_index] {
        codes[entry.symbol as usize].push(false);
    }
    for entry in &symbols[split_index..] {
        codes[entry.symbol as usize].push(true);
    }

    let (left, right) = symbols.split_at_mut(split_index);
    assign_codes_recursive(left, codes);
    assign_codes_recursive(right, codes);
}

fn choose_split_index(symbols: &[SymbolFrequency]) -> usize {
    let total: u64 = symbols.iter().map(|entry| entry.frequency as u64).sum();
    let mut left_sum = 0u64;
    let mut best_index = 1usize;
    let mut best_difference = total;

    // Choose the split that makes the left and right frequency totals as
    // balanced as possible while keeping both halves non-empty.
    for index in 1..symbols.len() {
        left_sum += symbols[index - 1].frequency as u64;
        let right_sum = total - left_sum;
        let difference = left_sum.abs_diff(right_sum);

        if difference < best_difference {
            best_difference = difference;
            best_index = index;
        }
    }

    best_index
}

fn build_decode_tree(codes: &[Vec<bool>], frequencies: &[u32; 256]) -> Result<DecodeNode, ErrorsJSL> {
    let mut root = DecodeNode::new_internal();

    for (symbol, code) in codes.iter().enumerate() {
        if frequencies[symbol] == 0 {
            continue;
        }

        if code.is_empty() {
            return Err(ErrorsJSL::InvalidInputRange(
                "Shannon-Fano code table unexpectedly contained an empty code for a present symbol.",
            ));
        }

        insert_code(&mut root, code, symbol as u8)?;
    }

    Ok(root)
}

fn insert_code(root: &mut DecodeNode, code: &[bool], symbol: u8) -> Result<(), ErrorsJSL> {
    insert_code_recursive(root, code, 0, symbol)
}

fn insert_code_recursive(
    node: &mut DecodeNode,
    code: &[bool],
    depth: usize,
    symbol: u8,
) -> Result<(), ErrorsJSL> {
    match node {
        DecodeNode::Leaf { .. } => Err(ErrorsJSL::InvalidInputRange(
            "Shannon-Fano code table attempted to descend through an existing leaf.",
        )),
        DecodeNode::Internal { left, right } => {
            let bit = code
                .get(depth)
                .ok_or(ErrorsJSL::RuntimeError("Shannon-Fano insertion depth exceeded the code length."))?;
            let slot = if *bit { right } else { left };

            if depth + 1 == code.len() {
                if slot.is_some() {
                    return Err(ErrorsJSL::InvalidInputRange(
                        "Shannon-Fano code table attempted to assign the same code path twice.",
                    ));
                }
                *slot = Some(Box::new(DecodeNode::Leaf { symbol }));
                Ok(())
            } else {
                if slot.is_none() {
                    *slot = Some(Box::new(DecodeNode::new_internal()));
                }
                insert_code_recursive(
                    slot.as_deref_mut().ok_or(ErrorsJSL::RuntimeError(
                        "Shannon-Fano decode tree branch disappeared during insertion.",
                    ))?,
                    code,
                    depth + 1,
                    symbol,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_empty_input() {
        let payload = b"";
        let compressed = try_shannon_fano_compress(payload).unwrap();
        let decompressed = try_shannon_fano_decompress(&compressed).unwrap();

        assert_eq!(decompressed, payload);
        assert_eq!(&compressed[..MAGIC.len()], MAGIC);
    }

    #[test]
    fn round_trip_repeated_text() {
        let payload = b"this is a shannon fano coding example";
        let compressed = try_shannon_fano_compress(payload).unwrap();
        let decompressed = try_shannon_fano_decompress(&compressed).unwrap();

        assert_eq!(decompressed, payload);
    }

    #[test]
    fn round_trip_binary_data() {
        let payload = [0u8, 0, 1, 1, 1, 2, 3, 5, 8, 8, 13, 21, 34, 55];
        let compressed = try_shannon_fano_compress(&payload).unwrap();
        let decompressed = try_shannon_fano_decompress(&compressed).unwrap();

        assert_eq!(decompressed, payload);
    }

    #[test]
    fn round_trip_single_symbol_input() {
        let payload = [7u8; 2048];
        let compressed = try_shannon_fano_compress(&payload).unwrap();
        let decompressed = try_shannon_fano_decompress(&compressed).unwrap();

        assert_eq!(decompressed, payload);
    }

    #[test]
    fn round_trip_kjv_text_asset() {
        let payload = include_bytes!("../../test_assets/kjv.txt");
        let compressed = try_shannon_fano_compress(payload).unwrap();
        let decompressed = try_shannon_fano_decompress(&compressed).unwrap();

        assert_eq!(decompressed, payload);
    }

    #[test]
    fn malformed_stream_returns_error() {
        let malformed = vec![b'S', b'F', b'A', b'N', 0x01];
        let result = try_shannon_fano_decompress(&malformed);

        assert!(result.is_err());
    }

    #[test]
    fn wrapper_returns_empty_on_malformed_stream() {
        let malformed = vec![0x01, 0x02, 0x03];
        let decompressed = shannon_fano_decompress(&malformed);

        assert!(decompressed.is_empty());
    }
}
