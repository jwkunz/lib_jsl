//! Manual LZ78 compression and decompression helpers.
//!
//! LZ78 differs from LZ77 in one important way: instead of pointing into a
//! sliding history window, the encoder explicitly builds a dictionary of phrases
//! and emits dictionary indices. Each new token is "an existing phrase" plus
//! "one extra byte".
//!
//! # Format Overview
//!
//! The compressed stream begins with the ASCII header `LZ78`. The remainder is
//! a sequence of dictionary tokens:
//!
//! - 16-bit little-endian prefix index
//! - 1-byte suffix-present flag
//! - optional 1-byte suffix literal when the flag is `1`
//!
//! Dictionary index `0` means "the empty phrase". A token therefore says:
//!
//! - take dictionary entry `prefix_index`
//! - append the optional suffix byte
//! - emit that phrase to the output
//! - insert the new phrase into the dictionary
//!
//! The optional-suffix form is used so the final phrase can be emitted even if
//! the input ends exactly on a phrase already present in the dictionary.

use std::collections::HashMap;

use lib_jsl_core::ErrorsJSL;

const MAGIC: &[u8; 4] = b"LZ78";
const HAS_SUFFIX: u8 = 1;
const NO_SUFFIX: u8 = 0;
const MAX_INDEX: usize = u16::MAX as usize;

/// Compress a byte slice with a manual LZ78 encoder.
pub fn lz78_compress(input: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len().saturating_add(MAGIC.len() + 4));
    output.extend_from_slice(MAGIC);

    // The dictionary maps complete phrases to their assigned indices. Index
    // zero is reserved for the empty phrase and therefore does not appear here.
    let mut dictionary: HashMap<Vec<u8>, u16> = HashMap::new();
    let mut next_index: usize = 1;

    // `current_phrase` is the longest phrase we have matched so far while
    // walking the input. We extend it greedily until the next extension would
    // no longer exist in the dictionary.
    let mut current_phrase = Vec::new();

    for &byte in input {
        let mut extended_phrase = current_phrase.clone();
        extended_phrase.push(byte);

        if dictionary.contains_key(&extended_phrase) {
            // If the longer phrase already exists, keep growing it. This is the
            // LZ78 "longest dictionary phrase" step.
            current_phrase = extended_phrase;
        } else {
            // We found the first byte that takes us outside the existing
            // dictionary, so emit:
            // - the index of the longest known prefix
            // - the new suffix byte that extends it
            let prefix_index = phrase_index(&dictionary, &current_phrase);
            emit_token(&mut output, prefix_index, Some(byte));

            // Then teach the dictionary about this newly discovered phrase,
            // unless we have reached the maximum index size representable by the
            // on-wire format.
            if next_index <= MAX_INDEX {
                dictionary.insert(extended_phrase, next_index as u16);
                next_index += 1;
            }

            // Start searching for the next phrase from scratch.
            current_phrase.clear();
        }
    }

    // If we end the input while holding an already-known phrase, emit it using
    // the "no suffix" form.
    if !current_phrase.is_empty() {
        let prefix_index = phrase_index(&dictionary, &current_phrase);
        emit_token(&mut output, prefix_index, None);
    }

    output
}

/// Decompress a byte slice previously produced by [`lz78_compress`].
pub fn lz78_decompress(input: &[u8]) -> Vec<u8> {
    try_lz78_decompress(input).unwrap_or_default()
}

/// Fallible variant of [`lz78_decompress`].
pub fn try_lz78_decompress(input: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
    if input.len() < MAGIC.len() || &input[..MAGIC.len()] != MAGIC {
        return Err(ErrorsJSL::InvalidInputRange("Compressed payload is missing the LZ78 header."));
    }

    let mut output = Vec::new();

    // Dictionary index zero is the empty phrase; every other entry is appended
    // as tokens are decoded.
    let mut dictionary: Vec<Vec<u8>> = vec![Vec::new()];
    let mut cursor = MAGIC.len();

    while cursor < input.len() {
        if cursor + 3 > input.len() {
            return Err(ErrorsJSL::InvalidInputRange("LZ78 token is missing prefix index or suffix flag bytes."));
        }

        let prefix_index = u16::from_le_bytes([input[cursor], input[cursor + 1]]) as usize;
        cursor += 2;

        let suffix_flag = input[cursor];
        cursor += 1;

        if prefix_index >= dictionary.len() {
            return Err(ErrorsJSL::InvalidInputRange("LZ78 prefix index points outside the reconstructed dictionary."));
        }

        // Rebuild the phrase by copying the referenced dictionary phrase first.
        let mut phrase = dictionary[prefix_index].clone();

        match suffix_flag {
            HAS_SUFFIX => {
                if cursor >= input.len() {
                    return Err(ErrorsJSL::InvalidInputRange("LZ78 token declares a suffix byte but none is present."));
                }

                // Then append the one new literal byte that made this phrase a
                // fresh dictionary entry during compression.
                phrase.push(input[cursor]);
                cursor += 1;
            }
            NO_SUFFIX => {
                // This is the end-of-stream form used when the encoder ended on
                // an already-known phrase. No additional byte is appended.
            }
            _ => {
                return Err(ErrorsJSL::InvalidInputRange("LZ78 suffix flag must be either 0 or 1."));
            }
        }

        output.extend_from_slice(&phrase);

        // Every non-empty decoded phrase becomes the next dictionary entry, just
        // as it did during compression.
        if !phrase.is_empty() && dictionary.len() <= MAX_INDEX {
            dictionary.push(phrase);
        }
    }

    Ok(output)
}

/// Look up the dictionary index for a phrase, returning zero for the empty
/// phrase.
fn phrase_index(dictionary: &HashMap<Vec<u8>, u16>, phrase: &[u8]) -> u16 {
    if phrase.is_empty() {
        0
    } else {
        dictionary.get(phrase).copied().unwrap_or(0)
    }
}

/// Emit one LZ78 token into the custom byte stream.
fn emit_token(output: &mut Vec<u8>, prefix_index: u16, suffix: Option<u8>) {
    output.extend_from_slice(&prefix_index.to_le_bytes());

    match suffix {
        Some(byte) => {
            output.push(HAS_SUFFIX);
            output.push(byte);
        }
        None => {
            output.push(NO_SUFFIX);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_empty_input() {
        let payload = b"";
        let compressed = lz78_compress(payload);
        let decompressed = lz78_decompress(&compressed);

        assert_eq!(decompressed, payload);
        assert_eq!(&compressed[..MAGIC.len()], MAGIC);
    }

    #[test]
    fn round_trip_repeated_text() {
        let payload = b"TOBEORNOTTOBEORTOBEORNOTO";
        let compressed = lz78_compress(payload);
        let decompressed = lz78_decompress(&compressed);

        assert_eq!(decompressed, payload);
    }

    #[test]
    fn round_trip_binary_data() {
        let payload = [0u8, 1, 0, 1, 0, 1, 2, 2, 2, 3, 4, 3, 4];
        let compressed = lz78_compress(&payload);
        let decompressed = lz78_decompress(&compressed);

        assert_eq!(decompressed, payload);
    }

    #[test]
    fn round_trip_kjv_text_asset() {
        // Embed the asset directly into the test binary so this large-file
        // regression test keeps working no matter where `cargo test` is run.
        let payload = include_bytes!("../../test_assets/kjv.txt");
        let compressed = lz78_compress(payload);
        let decompressed = lz78_decompress(&compressed);

        assert_eq!(decompressed, payload);
    }

    #[test]
    fn malformed_stream_returns_error_in_try_api() {
        let malformed = vec![b'L', b'Z', b'7', b'8', 0x01, 0x00, 0x00];
        let result = try_lz78_decompress(&malformed);

        assert!(result.is_err());
    }

    #[test]
    fn malformed_stream_returns_empty_in_wrapper_api() {
        let malformed = vec![0x00, 0x00, 0x01];
        let decompressed = lz78_decompress(&malformed);

        assert!(decompressed.is_empty());
    }
}
