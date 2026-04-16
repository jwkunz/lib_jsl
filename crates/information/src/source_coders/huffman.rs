//! Manual Huffman compression and decompression helpers.
//!
//! Huffman coding is an entropy coder rather than a dictionary coder. Instead
//! of referring to repeated phrases, it assigns short bit patterns to frequent
//! bytes and longer bit patterns to rare bytes. The assigned codes are
//! prefix-free, meaning no complete code is the prefix of another code, which
//! lets the decoder recover the original byte stream bit by bit.
//!
//! # Format Overview
//!
//! The compressed stream begins with the ASCII header `HUF0`. It is followed
//! by a compact description of the coding model and then the packed bitstream:
//!
//! - 4-byte magic header: `HUF0`
//! - 8-byte little-endian original length
//! - 2-byte little-endian symbol count
//! - repeated `symbol_count` times:
//!   - 1 byte: symbol value
//!   - 4 bytes little-endian: symbol frequency
//! - packed compressed bits in most-significant-bit-first order
//!
//! The decoder rebuilds the exact same Huffman tree from the stored frequency
//! table, so compression and decompression remain fully self-contained.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use lib_jsl_core::ErrorsJSL;

const MAGIC: &[u8; 4] = b"HUF0";

#[derive(Clone, Debug)]
enum HuffmanNode {
    Leaf { symbol: u8 },
    Internal { left: Box<HuffmanNode>, right: Box<HuffmanNode> },
}

#[derive(Clone, Debug)]
struct HeapEntry {
    frequency: u32,
    min_symbol: u8,
    node: HuffmanNode,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.frequency == other.frequency && self.min_symbol == other.min_symbol
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // `BinaryHeap` is a max-heap by default, so we reverse the comparison
        // to make lower frequencies come out first. The `min_symbol` tie-break
        // keeps tree construction deterministic across platforms.
        other
            .frequency
            .cmp(&self.frequency)
            .then_with(|| other.min_symbol.cmp(&self.min_symbol))
    }
}

/// Compress a byte slice with a manual Huffman encoder.
pub fn huffman_compress(input: &[u8]) -> Vec<u8> {
    try_huffman_compress(input).unwrap_or_default()
}

/// Fallible variant of [`huffman_compress`].
pub fn try_huffman_compress(input: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
    if input.len() > u64::MAX as usize {
        return Err(ErrorsJSL::InvalidInputRange("Huffman input is too large to encode its original length."));
    }

    let frequencies = build_frequency_table(input);
    let symbol_count = frequencies.iter().filter(|&&count| count > 0).count();

    if symbol_count > u16::MAX as usize {
        return Err(ErrorsJSL::InvalidInputRange("Huffman frequency table has too many distinct symbols."));
    }

    let mut output = Vec::new();
    output.extend_from_slice(MAGIC);
    output.extend_from_slice(&(input.len() as u64).to_le_bytes());
    output.extend_from_slice(&(symbol_count as u16).to_le_bytes());

    // Store the model explicitly so the decoder can rebuild the same tree.
    for (symbol, &frequency) in frequencies.iter().enumerate() {
        if frequency > 0 {
            output.push(symbol as u8);
            output.extend_from_slice(&frequency.to_le_bytes());
        }
    }

    if input.is_empty() {
        return Ok(output);
    }

    let root = build_tree_from_frequencies(&frequencies)?;
    let mut codes = vec![Vec::new(); 256];
    let mut current_path = Vec::new();
    build_code_table(&root, &mut current_path, &mut codes);

    // Pack the variable-length codes into bytes. Bits are written
    // most-significant-bit first because that is easy to inspect in hex dumps.
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

    // If the final byte is only partially filled, shift the payload bits up
    // into the high positions and leave the unused low bits as zero padding.
    if bit_count > 0 {
        current_byte <<= 8 - bit_count;
        output.push(current_byte);
    }

    Ok(output)
}

/// Decompress a byte slice previously produced by [`huffman_compress`].
pub fn huffman_decompress(input: &[u8]) -> Vec<u8> {
    try_huffman_decompress(input).unwrap_or_default()
}

/// Fallible variant of [`huffman_decompress`].
pub fn try_huffman_decompress(input: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
    if input.len() < MAGIC.len() || &input[..MAGIC.len()] != MAGIC {
        return Err(ErrorsJSL::InvalidInputRange("Compressed payload is missing the Huffman header."));
    }

    let mut cursor = MAGIC.len();

    if cursor + 8 + 2 > input.len() {
        return Err(ErrorsJSL::InvalidInputRange("Huffman header is truncated before length or symbol count."));
    }

    let original_len = u64::from_le_bytes(
        input[cursor..cursor + 8]
            .try_into()
            .map_err(|_| ErrorsJSL::RuntimeError("Failed to parse Huffman original length field."))?,
    ) as usize;
    cursor += 8;

    let symbol_count = u16::from_le_bytes(
        input[cursor..cursor + 2]
            .try_into()
            .map_err(|_| ErrorsJSL::RuntimeError("Failed to parse Huffman symbol count field."))?,
    ) as usize;
    cursor += 2;

    let table_bytes = symbol_count
        .checked_mul(5)
        .ok_or(ErrorsJSL::InvalidInputRange("Huffman frequency table size overflowed."))?;
    if cursor + table_bytes > input.len() {
        return Err(ErrorsJSL::InvalidInputRange("Huffman frequency table extends beyond the compressed payload."));
    }

    let mut frequencies = [0u32; 256];
    for _ in 0..symbol_count {
        let symbol = input[cursor];
        let frequency = u32::from_le_bytes(
            input[cursor + 1..cursor + 5]
                .try_into()
                .map_err(|_| ErrorsJSL::RuntimeError("Failed to parse Huffman symbol frequency field."))?,
        );
        cursor += 5;

        if frequency == 0 {
            return Err(ErrorsJSL::InvalidInputRange("Huffman frequency table cannot contain zero-frequency entries."));
        }
        if frequencies[symbol as usize] != 0 {
            return Err(ErrorsJSL::InvalidInputRange("Huffman frequency table contains duplicate symbol entries."));
        }

        frequencies[symbol as usize] = frequency;
    }

    if original_len == 0 {
        return Ok(Vec::new());
    }

    let total_frequency: usize = frequencies.iter().map(|&count| count as usize).sum();
    if total_frequency != original_len {
        return Err(ErrorsJSL::InvalidInputRange("Huffman frequency table does not match the stored original length."));
    }

    let root = build_tree_from_frequencies(&frequencies)?;

    // A single-symbol input produces a one-node tree and requires no payload
    // bits at all. In that case the original message is simply that symbol
    // repeated `original_len` times.
    if let HuffmanNode::Leaf { symbol } = root {
        return Ok(vec![symbol; original_len]);
    }

    let bitstream = &input[cursor..];
    let mut output = Vec::with_capacity(original_len);
    let mut current = &root;

    for &packed_byte in bitstream {
        for shift in (0..8).rev() {
            // Read one compressed bit and follow one edge through the decode
            // tree.
            let bit_is_set = ((packed_byte >> shift) & 1) == 1;

            current = match current {
                HuffmanNode::Leaf { symbol } => {
                    // If the previous step already landed on a leaf, emit that
                    // symbol and use the current bit to start a fresh descent
                    // from the root.
                    output.push(*symbol);
                    if output.len() == original_len {
                        return Ok(output);
                    }

                    descend_from_root(&root, bit_is_set)?
                }
                HuffmanNode::Internal { left, right } => {
                    if bit_is_set { right.as_ref() } else { left.as_ref() }
                }
            };

            if let HuffmanNode::Leaf { symbol } = current {
                // Reaching a leaf after consuming the current bit means one
                // full codeword has been decoded.
                output.push(*symbol);
                if output.len() == original_len {
                    return Ok(output);
                }
                current = &root;
            }
        }
    }

    Err(ErrorsJSL::InvalidInputRange("Huffman bitstream ended before the declared number of output bytes was reconstructed."))
}

fn descend_from_root<'a>(root: &'a HuffmanNode, bit_is_set: bool) -> Result<&'a HuffmanNode, ErrorsJSL> {
    match root {
        HuffmanNode::Internal { left, right } => {
            if bit_is_set { Ok(right.as_ref()) } else { Ok(left.as_ref()) }
        }
        HuffmanNode::Leaf { .. } => Err(ErrorsJSL::RuntimeError("Tried to descend from a leaf-only Huffman tree.")),
    }
}

fn build_frequency_table(input: &[u8]) -> [u32; 256] {
    let mut frequencies = [0u32; 256];
    for &byte in input {
        frequencies[byte as usize] = frequencies[byte as usize].saturating_add(1);
    }
    frequencies
}

fn build_tree_from_frequencies(frequencies: &[u32; 256]) -> Result<HuffmanNode, ErrorsJSL> {
    let mut heap = BinaryHeap::new();

    for (symbol, &frequency) in frequencies.iter().enumerate() {
        if frequency > 0 {
            // Start with one heap node per present symbol.
            heap.push(HeapEntry {
                frequency,
                min_symbol: symbol as u8,
                node: HuffmanNode::Leaf { symbol: symbol as u8 },
            });
        }
    }

    if heap.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("Cannot build a Huffman tree from an empty frequency table."));
    }

    while heap.len() > 1 {
        // Repeatedly merge the two lightest subtrees. That greedy step is the
        // core of the Huffman construction.
        let left = heap.pop().ok_or(ErrorsJSL::RuntimeError("Huffman heap unexpectedly emptied while building the tree."))?;
        let right = heap.pop().ok_or(ErrorsJSL::RuntimeError("Huffman heap unexpectedly emptied while building the tree."))?;

        heap.push(HeapEntry {
            frequency: left.frequency.saturating_add(right.frequency),
            min_symbol: left.min_symbol.min(right.min_symbol),
            node: HuffmanNode::Internal {
                left: Box::new(left.node),
                right: Box::new(right.node),
            },
        });
    }

    heap.pop()
        .map(|entry| entry.node)
        .ok_or(ErrorsJSL::RuntimeError("Failed to extract the final Huffman tree from the heap."))
}

fn build_code_table(node: &HuffmanNode, current_path: &mut Vec<bool>, codes: &mut [Vec<bool>]) {
    match node {
        HuffmanNode::Leaf { symbol } => {
            // A one-symbol file still needs a code. We use a single zero bit so
            // the encoder has something concrete to emit if a caller ever wants
            // to inspect the generated table directly.
            if current_path.is_empty() {
                current_path.push(false);
                codes[*symbol as usize] = current_path.clone();
                current_path.pop();
            } else {
                codes[*symbol as usize] = current_path.clone();
            }
        }
        HuffmanNode::Internal { left, right } => {
            // Appending `false` means "go left" at this depth.
            current_path.push(false);
            build_code_table(left, current_path, codes);
            current_path.pop();

            // Appending `true` means "go right" at this depth.
            current_path.push(true);
            build_code_table(right, current_path, codes);
            current_path.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_empty_input() {
        let payload = b"";
        let compressed = try_huffman_compress(payload).unwrap();
        let decompressed = try_huffman_decompress(&compressed).unwrap();

        assert_eq!(decompressed, payload);
        assert_eq!(&compressed[..MAGIC.len()], MAGIC);
    }

    #[test]
    fn round_trip_repeated_text() {
        let payload = b"this is an example of a huffman tree";
        let compressed = try_huffman_compress(payload).unwrap();
        let decompressed = try_huffman_decompress(&compressed).unwrap();

        assert_eq!(decompressed, payload);
    }

    #[test]
    fn round_trip_binary_data() {
        let payload = [0u8, 0, 0, 1, 1, 2, 3, 5, 8, 13, 21, 21, 21, 34];
        let compressed = try_huffman_compress(&payload).unwrap();
        let decompressed = try_huffman_decompress(&compressed).unwrap();

        assert_eq!(decompressed, payload);
    }

    #[test]
    fn round_trip_single_symbol_input() {
        let payload = [42u8; 1024];
        let compressed = try_huffman_compress(&payload).unwrap();
        let decompressed = try_huffman_decompress(&compressed).unwrap();

        assert_eq!(decompressed, payload);
    }

    #[test]
    fn round_trip_kjv_text_asset() {
        let payload = include_bytes!("../../test_assets/kjv.txt");
        let compressed = try_huffman_compress(payload).unwrap();
        let decompressed = try_huffman_decompress(&compressed).unwrap();

        assert_eq!(decompressed, payload);
    }

    #[test]
    fn malformed_stream_returns_error() {
        let malformed = vec![b'H', b'U', b'F', b'0', 0x01];
        let result = try_huffman_decompress(&malformed);

        assert!(result.is_err());
    }

    #[test]
    fn wrapper_returns_empty_on_malformed_stream() {
        let malformed = vec![0x01, 0x02, 0x03];
        let decompressed = huffman_decompress(&malformed);

        assert!(decompressed.is_empty());
    }
}
