//! Manual LZW compression and decompression helpers.
//!
//! LZW starts from the LZ78 idea of building a dictionary of phrases, but it
//! removes the explicit "suffix byte" from the token stream. Instead, both the
//! encoder and decoder grow the same dictionary deterministically as they read
//! codes.
//!
//! # Format Overview
//!
//! The compressed stream begins with the ASCII header `LZW0`. The remainder is
//! a sequence of 16-bit little-endian dictionary codes.
//!
//! The initial dictionary contains all 256 one-byte values. Whenever the
//! encoder emits a code for phrase `W` and then sees the next phrase `WK`, the
//! new combined phrase is inserted into the dictionary. The decoder mirrors
//! that same update rule.

use std::collections::HashMap;

use lib_jsl_core::ErrorsJSL;

const MAGIC: &[u8; 4] = b"LZW0";
const MAX_CODE: usize = u16::MAX as usize;

/// Compress a byte slice with a manual LZW encoder.
pub fn lzw_compress(input: &[u8]) -> Vec<u8> {
    // We reserve enough room for the header plus a rough stream of 16-bit
    // codes. LZW can compress or expand depending on the input, so this is
    // only a starting guess.
    let mut output = Vec::with_capacity(input.len().saturating_add(MAGIC.len() + 2));
    output.extend_from_slice(MAGIC);

    // Seed the dictionary with every one-byte phrase so compression can start
    // immediately from raw input bytes.
    let mut dictionary: HashMap<Vec<u8>, u16> =
        (0u16..=255).map(|code| (vec![code as u8], code)).collect();
    let mut next_code: usize = 256;

    let mut current_phrase = Vec::new();

    for &byte in input {
        // Hypothesize that the next input byte extends the phrase we are
        // currently building. If that longer phrase already exists, we keep
        // growing; otherwise we emit the known prefix and learn the longer one.
        let mut extended_phrase = current_phrase.clone();
        extended_phrase.push(byte);

        if dictionary.contains_key(&extended_phrase) {
            // Keep extending while the dictionary already knows the longer
            // phrase.
            current_phrase = extended_phrase;
        } else {
            // The longer phrase is new, so emit the code for the longest phrase
            // we *did* know.
            //
            // Example:
            // if `TOBE` is in the dictionary but `TOBEN` is not, we emit the
            // code for `TOBE`, then insert `TOBEN`.
            if !current_phrase.is_empty() {
                let code = dictionary[&current_phrase];
                output.extend_from_slice(&code.to_le_bytes());
            }

            // Then add the newly discovered phrase to the dictionary if the
            // 16-bit code space still has room.
            if next_code <= MAX_CODE {
                dictionary.insert(extended_phrase, next_code as u16);
                next_code += 1;
            }

            // The current unmatched suffix is now just the single byte that
            // caused the miss.
            current_phrase.clear();
            current_phrase.push(byte);
        }
    }

    // Emit the final phrase after the scan finishes.
    if !current_phrase.is_empty() {
        let code = dictionary[&current_phrase];
        output.extend_from_slice(&code.to_le_bytes());
    }

    output
}

/// Decompress a byte slice previously produced by [`lzw_compress`].
pub fn lzw_decompress(input: &[u8]) -> Vec<u8> {
    try_lzw_decompress(input).unwrap_or_default()
}

/// Fallible variant of [`lzw_decompress`].
pub fn try_lzw_decompress(input: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
    // The magic header distinguishes this payload from the other source coders
    // in the crate, which all have their own byte layouts.
    if input.len() < MAGIC.len() || &input[..MAGIC.len()] != MAGIC {
        return Err(ErrorsJSL::InvalidInputRange("Compressed payload is missing the LZW header."));
    }

    if (input.len() - MAGIC.len()) % 2 != 0 {
        return Err(ErrorsJSL::InvalidInputRange("LZW code stream must contain an even number of bytes after the header."));
    }

    // Seed the decoder dictionary with the same 256 one-byte phrases used by
    // the encoder.
    let mut dictionary: Vec<Vec<u8>> = (0u16..=255).map(|code| vec![code as u8]).collect();
    let mut next_code: usize = 256;
    let mut output = Vec::new();
    let mut cursor = MAGIC.len();
    let mut previous_phrase: Option<Vec<u8>> = None;

    while cursor < input.len() {
        // Read the next 16-bit phrase code from the compressed stream.
        let code = u16::from_le_bytes([input[cursor], input[cursor + 1]]);
        cursor += 2;

        let phrase = if let Some(existing) = dictionary.get(code as usize) {
            // The common case: the code already names a phrase we know.
            existing.clone()
        } else if (code as usize) == next_code {
            // This is the classic LZW "KwKwK" edge case. The encoder has just
            // created the next dictionary entry, and the decoder can reconstruct
            // it as `previous_phrase + first(previous_phrase)`.
            let previous = previous_phrase.as_ref().ok_or(ErrorsJSL::InvalidInputRange(
                "LZW encountered a forward reference before any previous phrase existed.",
            ))?;
            let first = *previous.first().ok_or(ErrorsJSL::InvalidInputRange(
                "LZW previous phrase unexpectedly had no leading byte.",
            ))?;

            let mut inferred = previous.clone();
            inferred.push(first);
            inferred
        } else {
            return Err(ErrorsJSL::InvalidInputRange("LZW code points outside the reconstructed dictionary."));
        };

        // Once the phrase is known, emit it directly to reconstruct the source.
        output.extend_from_slice(&phrase);

        if let Some(previous) = &previous_phrase {
            // Mirror the encoder's update rule by creating
            // `previous_phrase + first_byte_of_current_phrase`.
            let mut new_phrase = previous.clone();
            let first = *phrase.first().ok_or(ErrorsJSL::InvalidInputRange(
                "LZW current phrase unexpectedly had no leading byte.",
            ))?;
            new_phrase.push(first);

            if next_code <= MAX_CODE {
                dictionary.push(new_phrase);
                next_code += 1;
            }
        }

        previous_phrase = Some(phrase);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_empty_input() {
        let payload = b"";
        let compressed = lzw_compress(payload);
        let decompressed = lzw_decompress(&compressed);

        assert_eq!(decompressed, payload);
        assert_eq!(&compressed[..MAGIC.len()], MAGIC);
    }

    #[test]
    fn round_trip_repeated_text() {
        let payload = b"TOBEORNOTTOBEORTOBEORNOT";
        let compressed = lzw_compress(payload);
        let decompressed = lzw_decompress(&compressed);

        assert_eq!(decompressed, payload);
    }

    #[test]
    fn round_trip_binary_data() {
        let payload = [1u8, 2, 1, 2, 1, 2, 1, 2, 3, 3, 3, 4, 5, 4, 5];
        let compressed = lzw_compress(&payload);
        let decompressed = lzw_decompress(&compressed);

        assert_eq!(decompressed, payload);
    }

    #[test]
    fn round_trip_kjv_text_asset() {
        // Embed the asset directly into the test binary so this large-file
        // regression test keeps working no matter where `cargo test` is run.
        let payload = include_bytes!("../../test_assets/kjv.txt");
        let compressed = lzw_compress(payload);
        let decompressed = lzw_decompress(&compressed);

        assert_eq!(decompressed, payload);
    }

    #[test]
    fn malformed_stream_returns_error_in_try_api() {
        let malformed = vec![b'L', b'Z', b'W', b'0', 0x34];
        let result = try_lzw_decompress(&malformed);

        assert!(result.is_err());
    }

    #[test]
    fn malformed_stream_returns_empty_in_wrapper_api() {
        let malformed = vec![0x34];
        let decompressed = lzw_decompress(&malformed);

        assert!(decompressed.is_empty());
    }
}
