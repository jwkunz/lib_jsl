//! Manual Lempel-Ziv compression and decompression helpers.
//!
//! This module implements a compact LZ77-style codec over raw bytes. The goal
//! is not to compete with production formats such as DEFLATE or zstd; instead,
//! it provides a transparent reference implementation that is easy to study,
//! debug, and extend.
//!
//! # Format Overview
//!
//! The compressed stream begins with the ASCII header `LZ77`. The remainder of
//! the payload is a sequence of variable-length tokens:
//!
//! - literal run token:
//!   - high bit clear in the control byte
//!   - lower 7 bits encode `literal_count - 1`
//!   - the control byte is followed by `literal_count` raw bytes
//! - back-reference token:
//!   - high bit set in the control byte
//!   - lower 7 bits encode `match_len - MIN_MATCH`
//!   - followed by a 16-bit little-endian offset
//!
//! A back-reference means "copy `match_len` bytes from `offset` bytes before
//! the current output position". This is the classic LZ77 idea: when we see
//! repeated substrings, we emit a pointer into previously decoded output rather
//! than repeating the bytes literally.
//!
//! # API Design
//!
//! The user-facing functions requested for this crate consume `&[u8]` and return
//! `Vec<u8>`. To keep the implementation honest, the fallible decoder logic is
//! exposed as [`try_lempel_ziv_decompress`]. The infallible
//! [`lempel_ziv_decompress`] wrapper returns an empty vector when the payload is
//! malformed.
//!
//! That behavior keeps the byte-in/byte-out signature simple while still giving
//! callers a way to distinguish format errors when they want it.

use lib_jsl_core::ErrorsJSL;

const MAGIC: &[u8; 4] = b"LZ77";
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = MIN_MATCH + 0x7F;
const MAX_LITERAL_RUN: usize = 0x80;
const MAX_WINDOW: usize = u16::MAX as usize;

/// Compress a byte slice with a manual Lempel-Ziv (LZ77-style) encoder.
///
/// The encoder scans the already-seen prefix of `input` and looks for the
/// longest repeated substring that starts at the current position. When it
/// finds a match of at least three bytes, it emits a back-reference token;
/// otherwise it emits literal bytes.
///
/// The returned vector contains a small `LZ77` header followed by the custom
/// token stream described in the module-level documentation.
pub fn lempel_ziv_compress(input: &[u8]) -> Vec<u8> {
    // Start with enough room for the original data plus a tiny amount of
    // framing overhead. This is only a hint to reduce reallocations; the final
    // compressed stream may still grow or shrink relative to the input.
    let mut output = Vec::with_capacity(input.len().saturating_add(MAGIC.len() + 1));

    // Write the magic header first so the decoder can quickly recognize that
    // this byte stream belongs to our custom format.
    output.extend_from_slice(MAGIC);

    // `literals` temporarily collects bytes that we choose not to encode as a
    // back-reference. We buffer them so several adjacent literal bytes can be
    // emitted as one token instead of many one-byte tokens.
    let mut literals = Vec::new();

    // `position` is the encoder's read head into the original input.
    let mut position = 0usize;

    // Keep walking forward until every input byte has been represented by
    // either a literal run or a back-reference.
    while position < input.len() {
        // Ask the history window for the best repeat we can encode starting at
        // this exact byte position.
        let (best_offset, best_len) = find_longest_match(input, position);

        if best_len >= MIN_MATCH {
            // Before writing a match token, flush any pending literal bytes.
            // This keeps the compressed stream in the same logical order as the
            // original input: first the raw bytes, then the repeated region.
            flush_literals(&mut output, &mut literals);

            // Emit a token that says "go backwards by `best_offset` bytes in
            // the already-decoded output and copy `best_len` bytes from there".
            emit_match(&mut output, best_offset, best_len);

            // Because the match covers `best_len` bytes from the input, we can
            // skip over that whole region in one step.
            position += best_len;
        } else {
            // No useful repetition was found, so this byte must be carried
            // through literally.
            literals.push(input[position]);
            position += 1;

            // Literal runs are capped at 128 bytes because the run length is
            // stored in the lower seven bits of the control byte.
            if literals.len() == MAX_LITERAL_RUN {
                flush_literals(&mut output, &mut literals);
            }
        }
    }

    // If the input ended while we were still collecting literal bytes, emit
    // that final run now.
    flush_literals(&mut output, &mut literals);
    output
}

/// Decompress a byte slice previously produced by [`lempel_ziv_compress`].
///
/// This wrapper keeps the requested simple signature. If the compressed stream
/// is malformed, it returns an empty vector. Code that needs explicit error
/// reporting should call [`try_lempel_ziv_decompress`] instead.
pub fn lempel_ziv_decompress(input: &[u8]) -> Vec<u8> {
    try_lempel_ziv_decompress(input).unwrap_or_default()
}

/// Fallible variant of [`lempel_ziv_decompress`].
///
/// This performs structural validation of the custom format:
///
/// - the `LZ77` header must be present
/// - literal tokens must contain the declared number of bytes
/// - back-references must point into already-decoded output
/// - offsets must be non-zero
pub fn try_lempel_ziv_decompress(input: &[u8]) -> Result<Vec<u8>, ErrorsJSL> {
    // The decoder expects the stream to begin with the exact four-byte magic
    // header. If it is absent, the rest of the bytes are ambiguous, so we stop
    // immediately.
    if input.len() < MAGIC.len() || &input[..MAGIC.len()] != MAGIC {
        return Err(ErrorsJSL::InvalidInputRange("Compressed payload is missing the LZ77 header."));
    }

    // `output` is the progressively reconstructed original message. In LZ77,
    // back-references always point into this already-produced prefix.
    let mut output = Vec::new();

    // `cursor` is the decoder's read head into the compressed stream. We start
    // just after the magic header because those bytes have already been
    // consumed by validation.
    let mut cursor = MAGIC.len();

    // Decode one token at a time until we reach the end of the compressed
    // payload.
    while cursor < input.len() {
        // Every token begins with a one-byte control field that determines the
        // token kind and how much additional data we should read.
        let control = input[cursor];
        cursor += 1;

        if control & 0x80 == 0 {
            // Literal token: the lower seven bits store `count - 1`, so adding
            // one recovers the real number of raw bytes that follow.
            let literal_count = (control as usize) + 1;

            if cursor + literal_count > input.len() {
                return Err(ErrorsJSL::InvalidInputRange("Literal token extends beyond the compressed payload."));
            }

            // Copy the literal bytes directly from the compressed stream to the
            // reconstructed output, then move the cursor past them.
            output.extend_from_slice(&input[cursor..cursor + literal_count]);
            cursor += literal_count;
        } else {
            // Match token: after the control byte we require a two-byte offset.
            if cursor + 2 > input.len() {
                return Err(ErrorsJSL::InvalidInputRange("Back-reference token is missing its offset bytes."));
            }

            // The lower seven bits store `length - MIN_MATCH`, so adding
            // `MIN_MATCH` reconstructs the actual number of bytes to copy.
            let match_len = ((control & 0x7F) as usize) + MIN_MATCH;

            // The offset tells us how far back from the current end of the
            // output we should begin copying.
            let offset = u16::from_le_bytes([input[cursor], input[cursor + 1]]) as usize;
            cursor += 2;

            if offset == 0 || offset > output.len() {
                return Err(ErrorsJSL::InvalidInputRange("Back-reference offset points outside the decoded prefix."));
            }

            // Convert the backward-looking offset into a concrete starting index
            // inside the bytes we have already reconstructed.
            let start = output.len() - offset;

            // We copy one byte at a time so overlapping matches work exactly
            // like classic LZ77 decoding. This is important for patterns such
            // as `AAAAAA`, where newly written bytes become the source for
            // later bytes in the same token.
            for index in 0..match_len {
                // Read from the historical window, then append to the end of the
                // output. On later iterations, `output` may already contain the
                // bytes written by earlier iterations of this same loop, which
                // is precisely how overlap expansion works.
                let byte = output[start + index];
                output.push(byte);
            }
        }
    }

    Ok(output)
}

/// Search for the best back-reference that starts at `position`.
///
/// The implementation is intentionally straightforward rather than heavily
/// optimized: we scan the legal history window and measure how many bytes match
/// for each candidate. That keeps the algorithm easy to audit for correctness.
fn find_longest_match(input: &[u8], position: usize) -> (usize, usize) {
    // At the very first byte there is no history to point backward into, so a
    // match is impossible.
    if position == 0 {
        return (0, 0);
    }

    // Restrict the search to the largest offset representable by our 16-bit
    // format field.
    let window_start = position.saturating_sub(MAX_WINDOW);
    let mut best_offset = 0usize;
    let mut best_len = 0usize;

    // Consider every possible starting point in the look-behind window.
    for candidate in window_start..position {
        let mut match_len = 0usize;

        // Keep extending the candidate match while:
        // 1. we are still inside the original input,
        // 2. the candidate byte agrees with the byte at `position + match_len`,
        // 3. the match is still encodable in one token.
        //
        // The modulo term is the key overlap trick on the encoder side. If the
        // candidate region is shorter than the full repeated pattern, we allow
        // the comparison to wrap across the candidate span the same way the
        // decoder would observe bytes appearing during an overlapping copy.
        while position + match_len < input.len()
            && input[candidate + (match_len % (position - candidate))] == input[position + match_len]
            && match_len < MAX_MATCH
        {
            match_len += 1;
        }

        // Keep whichever candidate gives the longest repeat. LZ-family codecs
        // usually prefer longer matches because they replace more input bytes
        // with a single compact token.
        if match_len > best_len {
            best_len = match_len;
            best_offset = position - candidate;

            // Once we hit the maximum encodable match length we cannot improve
            // further, so ending the search here saves work.
            if best_len == MAX_MATCH {
                break;
            }
        }
    }

    (best_offset, best_len)
}

/// Emit a back-reference token into the compressed stream.
fn emit_match(output: &mut Vec<u8>, offset: usize, length: usize) {
    // Set the high bit to mark this as a match token, and pack the match
    // length into the remaining seven bits after subtracting `MIN_MATCH`.
    let control = 0x80 | ((length - MIN_MATCH) as u8);
    output.push(control);

    // Store the backward distance as a little-endian `u16`, which matches the
    // maximum window size enforced by the encoder.
    output.extend_from_slice(&(offset as u16).to_le_bytes());
}

/// Emit any queued literal bytes as one or more literal-run tokens.
fn flush_literals(output: &mut Vec<u8>, literals: &mut Vec<u8>) {
    // `cursor` walks through the temporary literal buffer as we split it into
    // one or more encoded runs.
    let mut cursor = 0usize;

    while cursor < literals.len() {
        // Each literal token can encode at most 128 bytes because only seven
        // control bits are available for the run length.
        let run_len = (literals.len() - cursor).min(MAX_LITERAL_RUN);

        // Literal tokens keep the high bit clear, so the control byte is just
        // `run_len - 1`.
        output.push((run_len - 1) as u8);
        output.extend_from_slice(&literals[cursor..cursor + run_len]);
        cursor += run_len;
    }

    // Reset the staging buffer so the caller can begin collecting a fresh run.
    literals.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_empty_input() {
        let payload = b"";
        let compressed = lempel_ziv_compress(payload);
        let decompressed = lempel_ziv_decompress(&compressed);

        assert_eq!(decompressed, payload);
        assert_eq!(&compressed[..MAGIC.len()], MAGIC);
    }

    #[test]
    fn round_trip_repeated_text() {
        let payload = b"TOBEORNOTTOBEORTOBEORNOT";
        let compressed = lempel_ziv_compress(payload);
        let decompressed = lempel_ziv_decompress(&compressed);

        assert_eq!(decompressed, payload);
        assert!(compressed.len() < payload.len() + MAGIC.len());
    }

    #[test]
    fn round_trip_binary_data_with_overlap() {
        let payload = [7u8, 7, 7, 7, 7, 7, 9, 9, 9, 1, 2, 1, 2, 1, 2, 1, 2];
        let compressed = lempel_ziv_compress(&payload);
        let decompressed = lempel_ziv_decompress(&compressed);

        assert_eq!(decompressed, payload);
    }

    #[test]
    fn malformed_stream_returns_error_in_try_api() {
        let malformed = vec![b'L', b'Z', b'7', b'7', 0x80, 0x00, 0x00];
        let result = try_lempel_ziv_decompress(&malformed);

        assert!(result.is_err());
    }

    #[test]
    fn malformed_stream_returns_empty_in_wrapper_api() {
        let malformed = vec![0x80, 0x00, 0x00];
        let decompressed = lempel_ziv_decompress(&malformed);

        assert!(decompressed.is_empty());
    }
}
