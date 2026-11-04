//! PNG images (ISO/IEC 15948): a lossless codec and file helpers.
//!
//! Decoding supports every standard colour type (greyscale, RGB, palette, grey + alpha, RGBA)
//! at every bit depth, all five scanline filters, Adam7 interlacing and palette transparency
//! (`tRNS`, which expands a palette image to RGBA). Samples come out as 8 bits, or 16 bits for
//! 16-bit images; low bit-depth greyscale is scaled to 8 bits. Encoding writes non-interlaced
//! 8- or 16-bit greyscale, grey + alpha, RGB or RGBA, choosing a filter per row by the usual
//! minimum-sum-of-absolute-differences heuristic, compressed with [`crate::deflate`].

use crate::deflate::{zlib_compress, zlib_decompress};
use lib_jsl_core::ErrorsJSL;
use std::path::Path;

/// A raster image with interleaved samples, shared by the lossless formats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RasterImage {
    pub width: usize,
    pub height: usize,
    /// 1 (grey), 2 (grey + alpha), 3 (RGB) or 4 (RGBA).
    pub channels: usize,
    /// 8 or 16.
    pub bit_depth: u8,
    /// Row-major interleaved samples, `width · height · channels` of them, each below
    /// `2^bit_depth`.
    pub samples: Vec<u16>,
}

impl RasterImage {
    pub fn new(width: usize, height: usize, channels: usize, bit_depth: u8, samples: Vec<u16>) -> Result<Self, ErrorsJSL> {
        let image = RasterImage { width, height, channels, bit_depth, samples };
        image.validate()?;
        Ok(image)
    }

    fn validate(&self) -> Result<(), ErrorsJSL> {
        if !(1..=4).contains(&self.channels) || !(self.bit_depth == 8 || self.bit_depth == 16) {
            return Err(ErrorsJSL::InvalidInputRange("Images need 1 to 4 channels and 8- or 16-bit samples."));
        }
        if self.samples.len() != self.width * self.height * self.channels {
            return Err(ErrorsJSL::IncompatibleArraySizes((self.samples.len(), self.width * self.height * self.channels)));
        }
        if self.bit_depth == 8 && self.samples.iter().any(|&s| s > 255) {
            return Err(ErrorsJSL::InvalidInputRange("8-bit samples must be below 256."));
        }
        Ok(())
    }

    /// The sample at `(x, y)` in channel `c`.
    pub fn sample(&self, x: usize, y: usize, c: usize) -> u16 {
        self.samples[(y * self.width + x) * self.channels + c]
    }

    /// The samples as bytes, scaling 16-bit values to 8 bits.
    pub fn to_u8(&self) -> Vec<u8> {
        match self.bit_depth {
            16 => self.samples.iter().map(|&s| (s >> 8) as u8).collect(),
            _ => self.samples.iter().map(|&s| s as u8).collect(),
        }
    }
}

const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    for (n, entry) in table.iter_mut().enumerate() {
        let mut c = n as u32;
        for _ in 0..8 {
            c = if c & 1 == 1 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *entry = c;
    }
    table
}

fn crc32(parts: &[&[u8]]) -> u32 {
    let table = crc32_table();
    let mut c = 0xffff_ffffu32;
    for part in parts {
        for &b in *part {
            c = table[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
        }
    }
    c ^ 0xffff_ffff
}

fn corrupt(message: &'static str) -> ErrorsJSL {
    ErrorsJSL::RuntimeError(message)
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let p = a as i16 + b as i16 - c as i16;
    let (pa, pb, pc) = ((p - a as i16).abs(), (p - b as i16).abs(), (p - c as i16).abs());
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Reverses the scanline filters of one (sub-)image of `rows` rows of `stride` bytes.
fn unfilter(data: &[u8], rows: usize, stride: usize, bpp: usize) -> Result<Vec<u8>, ErrorsJSL> {
    if data.len() < rows * (stride + 1) {
        return Err(corrupt("PNG image data is truncated."));
    }
    let mut out = vec![0u8; rows * stride];
    for y in 0..rows {
        let filter = data[y * (stride + 1)];
        let line = &data[y * (stride + 1) + 1..(y + 1) * (stride + 1)];
        let (done, current) = out.split_at_mut(y * stride);
        let previous = if y > 0 { &done[(y - 1) * stride..] } else { &[][..] };
        let current = &mut current[..stride];
        for i in 0..stride {
            let a = if i >= bpp { current[i - bpp] } else { 0 };
            let b = if y > 0 { previous[i] } else { 0 };
            let c = if y > 0 && i >= bpp { previous[i - bpp] } else { 0 };
            let predictor = match filter {
                0 => 0,
                1 => a,
                2 => b,
                3 => ((a as u16 + b as u16) / 2) as u8,
                4 => paeth(a, b, c),
                _ => return Err(corrupt("Unknown PNG filter type.")),
            };
            current[i] = line[i].wrapping_add(predictor);
        }
    }
    Ok(out)
}

/// Unpacks a row of `count` samples of `depth` bits (big-endian within bytes).
fn unpack(row: &[u8], count: usize, depth: u8) -> Vec<u16> {
    match depth {
        16 => (0..count).map(|i| u16::from_be_bytes([row[2 * i], row[2 * i + 1]])).collect(),
        8 => row[..count].iter().map(|&b| b as u16).collect(),
        _ => {
            let per_byte = 8 / depth as usize;
            let mask = (1u16 << depth) - 1;
            (0..count).map(|i| (row[i / per_byte] as u16 >> (8 - depth as usize * (i % per_byte + 1))) & mask).collect()
        }
    }
}

/// Decodes a PNG byte stream.
pub fn png_decode(data: &[u8]) -> Result<RasterImage, ErrorsJSL> {
    if data.len() < 8 || data[..8] != SIGNATURE {
        return Err(corrupt("Not a PNG file."));
    }
    let mut position = 8;
    let (mut header, mut palette, mut transparency) = (None, None, None);
    let mut compressed = Vec::new();
    while position + 12 <= data.len() {
        let length = u32::from_be_bytes(data[position..position + 4].try_into().expect("4 bytes")) as usize;
        let kind = &data[position + 4..position + 8];
        let end = position + 8 + length;
        if end + 4 > data.len() {
            return Err(corrupt("PNG chunk is truncated."));
        }
        let body = &data[position + 8..end];
        let stored = u32::from_be_bytes(data[end..end + 4].try_into().expect("4 bytes"));
        if crc32(&[kind, body]) != stored {
            return Err(corrupt("PNG chunk CRC mismatch."));
        }
        match kind {
            b"IHDR" => {
                if body.len() != 13 {
                    return Err(corrupt("Bad IHDR chunk."));
                }
                header = Some(body.to_vec());
            }
            b"PLTE" => palette = Some(body.to_vec()),
            b"tRNS" => transparency = Some(body.to_vec()),
            b"IDAT" => compressed.extend_from_slice(body),
            b"IEND" => break,
            _ if kind[0] & 0x20 == 0 => return Err(corrupt("Unsupported critical PNG chunk.")),
            _ => {}
        }
        position = end + 4;
    }
    let h = header.ok_or(corrupt("Missing IHDR chunk."))?;
    let width = u32::from_be_bytes(h[0..4].try_into().expect("4 bytes")) as usize;
    let height = u32::from_be_bytes(h[4..8].try_into().expect("4 bytes")) as usize;
    let (depth, colour, interlace) = (h[8], h[9], h[12]);
    let channels = match colour {
        0 | 3 => 1,
        2 => 3,
        4 => 2,
        6 => 4,
        _ => return Err(corrupt("Unknown PNG colour type.")),
    };
    let depth_ok = match colour {
        0 => [1, 2, 4, 8, 16].contains(&depth),
        3 => [1, 2, 4, 8].contains(&depth),
        _ => depth == 8 || depth == 16,
    };
    if !depth_ok || h[10] != 0 || h[11] != 0 || interlace > 1 || width == 0 || height == 0 {
        return Err(corrupt("Unsupported PNG header."));
    }
    if width > i32::MAX as usize || height > i32::MAX as usize {
        return Err(corrupt("PNG dimensions exceed 2^31 - 1."));
    }
    let raw = zlib_decompress(&compressed)?;
    let bits_per_pixel = channels * depth as usize;
    let bpp = bits_per_pixel.div_ceil(8);
    let passes: Vec<(usize, usize, usize, usize)> =
        if interlace == 1 { vec![(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)] } else { vec![(0, 0, 1, 1)] };
    // Check that the image data are all there before allocating for the size the header claims.
    let needed = passes.iter().try_fold(0usize, |total, &(x0, y0, dx, dy)| {
        let (pw, ph) = ((width.saturating_sub(x0)).div_ceil(dx), (height.saturating_sub(y0)).div_ceil(dy));
        if pw == 0 || ph == 0 {
            return Some(total);
        }
        let stride = pw.checked_mul(bits_per_pixel)?.div_ceil(8);
        total.checked_add(ph.checked_mul(stride.checked_add(1)?)?)
    });
    if needed.is_none_or(|n| raw.len() < n) {
        return Err(corrupt("PNG image data are shorter than the header implies."));
    }
    // Decode each (sub-)image into full-resolution sample positions.
    let mut samples = vec![0u16; width * height * channels];
    let mut offset = 0;
    for (x0, y0, dx, dy) in passes {
        let (pw, ph) = ((width.saturating_sub(x0)).div_ceil(dx), (height.saturating_sub(y0)).div_ceil(dy));
        if pw == 0 || ph == 0 {
            continue;
        }
        let stride = (pw * bits_per_pixel).div_ceil(8);
        let rows = unfilter(&raw[offset.min(raw.len())..], ph, stride, bpp)?;
        offset += ph * (stride + 1);
        for py in 0..ph {
            let values = unpack(&rows[py * stride..(py + 1) * stride], pw * channels, depth);
            for px in 0..pw {
                let (x, y) = (x0 + px * dx, y0 + py * dy);
                for c in 0..channels {
                    samples[(y * width + x) * channels + c] = values[px * channels + c];
                }
            }
        }
    }
    if colour == 3 {
        let palette = palette.ok_or(corrupt("Palette image without PLTE."))?;
        let alpha = transparency.unwrap_or_default();
        let out_channels = if alpha.is_empty() { 3 } else { 4 };
        let mut expanded = Vec::with_capacity(width * height * out_channels);
        for &index in &samples {
            let i = index as usize;
            if 3 * i + 2 >= palette.len() {
                return Err(corrupt("Palette index out of range."));
            }
            expanded.extend(palette[3 * i..3 * i + 3].iter().map(|&v| v as u16));
            if out_channels == 4 {
                expanded.push(*alpha.get(i).unwrap_or(&255) as u16);
            }
        }
        return RasterImage::new(width, height, out_channels, 8, expanded);
    }
    let out_depth = if depth == 16 { 16 } else { 8 };
    if depth < 8 {
        let scale = 255 / ((1u16 << depth) - 1);
        samples.iter_mut().for_each(|s| *s *= scale);
    }
    RasterImage::new(width, height, channels, out_depth, samples)
}

/// Encodes an image as PNG (zlib `level` 0 to 9).
pub fn png_encode(image: &RasterImage, level: u8) -> Result<Vec<u8>, ErrorsJSL> {
    image.validate()?;
    if image.width == 0 || image.height == 0 {
        return Err(ErrorsJSL::InvalidInputRange("PNG images cannot be empty."));
    }
    let colour = match image.channels {
        1 => 0u8,
        2 => 4,
        3 => 2,
        _ => 6,
    };
    let bytes_per_sample = image.bit_depth as usize / 8;
    let bpp = image.channels * bytes_per_sample;
    let stride = image.width * bpp;
    let packed: Vec<u8> = if image.bit_depth == 16 { image.samples.iter().flat_map(|s| s.to_be_bytes()).collect() } else { image.samples.iter().map(|&s| s as u8).collect() };
    let mut filtered = Vec::with_capacity(image.height * (stride + 1));
    let mut candidate = vec![0u8; stride];
    for y in 0..image.height {
        let row = &packed[y * stride..(y + 1) * stride];
        let previous = if y > 0 { &packed[(y - 1) * stride..y * stride] } else { &[][..] };
        let mut best: (u64, u8, Vec<u8>) = (u64::MAX, 0, Vec::new());
        for filter in 0..5u8 {
            for i in 0..stride {
                let a = if i >= bpp { row[i - bpp] } else { 0 };
                let b = if y > 0 { previous[i] } else { 0 };
                let c = if y > 0 && i >= bpp { previous[i - bpp] } else { 0 };
                let predictor = match filter {
                    0 => 0,
                    1 => a,
                    2 => b,
                    3 => ((a as u16 + b as u16) / 2) as u8,
                    _ => paeth(a, b, c),
                };
                candidate[i] = row[i].wrapping_sub(predictor);
            }
            let cost: u64 = candidate.iter().map(|&v| (v as i8).unsigned_abs() as u64).sum();
            if cost < best.0 {
                best = (cost, filter, candidate.clone());
            }
        }
        filtered.push(best.1);
        filtered.extend_from_slice(&best.2);
    }
    let mut out = SIGNATURE.to_vec();
    let mut chunk = |kind: &[u8; 4], body: &[u8]| {
        out.extend_from_slice(&(body.len() as u32).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        out.extend_from_slice(&crc32(&[kind, body]).to_be_bytes());
    };
    let mut header = Vec::with_capacity(13);
    header.extend_from_slice(&(image.width as u32).to_be_bytes());
    header.extend_from_slice(&(image.height as u32).to_be_bytes());
    header.extend_from_slice(&[image.bit_depth, colour, 0, 0, 0]);
    chunk(b"IHDR", &header);
    chunk(b"IDAT", &zlib_compress(&filtered, level));
    chunk(b"IEND", &[]);
    Ok(out)
}

/// Reads a PNG file.
pub fn read_png_file(path: impl AsRef<Path>) -> Result<RasterImage, ErrorsJSL> {
    let data = std::fs::read(path).map_err(|_| ErrorsJSL::RuntimeError("Could not read the PNG file."))?;
    png_decode(&data)
}

/// Writes a PNG file (zlib level 6).
pub fn write_png_file(path: impl AsRef<Path>, image: &RasterImage) -> Result<(), ErrorsJSL> {
    std::fs::write(path, png_encode(image, 6)?).map_err(|_| ErrorsJSL::RuntimeError("Could not write the PNG file."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_header_promising_more_data_than_the_file_holds_is_an_error() {
        // Rewrite the IHDR of a valid file to claim 2^31 - 1 square pixels (with a valid CRC):
        // this used to overflow the sample buffer's size and panic.
        let image = RasterImage::new(4, 4, 1, 8, vec![9; 16]).unwrap();
        let mut bytes = png_encode(&image, 6).unwrap();
        bytes[16..20].copy_from_slice(&0x7fff_ffffu32.to_be_bytes());
        bytes[20..24].copy_from_slice(&0x7fff_ffffu32.to_be_bytes());
        let crc = crc32(&[&bytes[12..16], &bytes[16..29]]);
        bytes[29..33].copy_from_slice(&crc.to_be_bytes());
        assert!(png_decode(&bytes).is_err());
    }

    // Test images built independently with CPython's zlib and struct modules.
    /// 4×3 RGB, rows filtered with Sub, Up and Paeth.
    const FILTERED_RGB: &[u8] = &[
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 4, 0, 0, 0, 3, 8, 2,
        0, 0, 0, 59, 150, 57, 145, 0, 0, 0, 27, 73, 68, 65, 84, 120, 218, 99, 100, 96, 248, 111, 35,
        119, 14, 130, 152, 68, 220, 190, 193, 17, 11, 136, 146, 131, 34, 0, 13, 139, 14, 135, 144,
        81, 59, 57, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
    ];
    /// 10×9 8-bit greyscale, Adam7 interlaced, pixel (x, y) = (7x + 13y) mod 256.
    const ADAM7_GRAY: &[u8] = &[
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 10, 0, 0, 0, 9, 8, 0,
        0, 0, 1, 89, 202, 210, 89, 0, 0, 0, 117, 73, 68, 65, 84, 120, 218, 99, 96, 176, 96, 200, 88,
        192, 32, 195, 208, 194, 96, 18, 144, 195, 192, 167, 197, 224, 20, 199, 80, 54, 137, 65, 74,
        195, 204, 37, 136, 193, 47, 38, 171, 162, 141, 129, 93, 84, 217, 208, 158, 65, 81, 223, 214,
        59, 146, 193, 218, 51, 60, 181, 152, 33, 52, 185, 176, 190, 151, 33, 191, 182, 123, 230,
        114, 6, 94, 17, 105, 37, 77, 3, 115, 59, 87, 31, 6, 117, 61, 83, 27, 103, 175, 192, 136,
        248, 52, 6, 71, 15, 255, 176, 216, 148, 236, 162, 202, 6, 134, 232, 164, 204, 130, 242, 186,
        214, 158, 201, 179, 0, 108, 97, 29, 92, 81, 53, 226, 176, 0, 0, 0, 0, 73, 69, 78, 68, 174,
        66, 96, 130,
    ];
    /// 5×2 2-bit palette with tRNS (alpha 255, 128, 0 for the first three entries).
    const PALETTE_2BIT: &[u8] = &[
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 5, 0, 0, 0, 2, 2, 3,
        0, 0, 0, 237, 4, 254, 206, 0, 0, 0, 12, 80, 76, 84, 69, 255, 0, 0, 0, 255, 0, 0, 0, 255, 9,
        9, 9, 92, 113, 126, 134, 0, 0, 0, 3, 116, 82, 78, 83, 255, 128, 0, 127, 109, 104, 120, 0, 0,
        0, 14, 73, 68, 65, 84, 120, 218, 99, 144, 118, 96, 120, 210, 0, 0, 3, 213, 1, 192, 188, 47,
        210, 168, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
    ];

    #[test]
    fn decodes_independently_encoded_images() {
        let rgb = png_decode(FILTERED_RGB).unwrap();
        assert_eq!((rgb.width, rgb.height, rgb.channels, rgb.bit_depth), (4, 3, 3, 8));
        for y in 0..3 {
            for x in 0..4 {
                let want = [(x * 60 + y * 20) % 256, (x * 30 + y * 70) % 256, (255 - x * 50 - y * 10) % 256];
                assert_eq!([rgb.sample(x, y, 0), rgb.sample(x, y, 1), rgb.sample(x, y, 2)], want.map(|v| v as u16), "({x}, {y})");
            }
        }
        let gray = png_decode(ADAM7_GRAY).unwrap();
        assert_eq!((gray.width, gray.height, gray.channels), (10, 9, 1));
        assert!((0..9).all(|y| (0..10).all(|x| gray.sample(x, y, 0) == ((7 * x + 13 * y) % 256) as u16)));
        let palette = png_decode(PALETTE_2BIT).unwrap();
        assert_eq!(palette.channels, 4);
        assert_eq!(&palette.samples[..8], &[255, 0, 0, 255, 0, 255, 0, 128]);
        assert_eq!(&palette.samples[12..16], &[9, 9, 9, 255]); // index 3 has no tRNS entry
    }

    #[test]
    fn round_trips_every_encodable_format_and_detects_corruption() {
        for (channels, depth) in [(1, 8), (2, 8), (3, 8), (4, 8), (1, 16), (3, 16), (4, 16)] {
            let (w, h) = (37, 23);
            let max = if depth == 8 { 255 } else { 65535 };
            let samples = (0..w * h * channels).map(|i| ((((i * 2654435761usize) >> 7) % (max + 1)) / if i % 3 == 0 { 1 } else { 7 }) as u16).collect();
            let image = RasterImage::new(w, h, channels, depth, samples).unwrap();
            let encoded = png_encode(&image, 6).unwrap();
            assert_eq!(png_decode(&encoded).unwrap(), image, "{channels} channels, {depth} bits");
        }
        // A smooth gradient compresses well thanks to the row filters.
        let gradient = RasterImage::new(256, 256, 1, 8, (0..65536).map(|i| ((i % 256 + i / 256) / 2) as u16).collect()).unwrap();
        assert!(png_encode(&gradient, 9).unwrap().len() < 4000);
        let mut bad = png_encode(&gradient, 6).unwrap();
        bad[40] ^= 0xff;
        assert!(png_decode(&bad).is_err());
        assert!(png_decode(b"not a png").is_err());
        assert!(RasterImage::new(2, 2, 5, 8, vec![0; 20]).is_err() && RasterImage::new(2, 2, 1, 8, vec![300; 4]).is_err());
    }
}
