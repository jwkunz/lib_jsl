//! Netpbm images: PBM (P1, P4), PGM (P2, P5) and PPM (P3, P6).
//!
//! The simplest lossless formats: a short text header (magic number, width, height and maximum
//! value, with `#` comments) followed by the samples, as ASCII numbers (P1–P3) or raw bytes
//! (P4–P6; 16-bit big-endian when the maximum exceeds 255). Bitmaps (PBM, where 1 is black) are
//! read as 8-bit greyscale with black = 0 and white = 255. Writing produces binary PGM (one
//! channel) or PPM (three channels).
//!
//! See the book entry [Image files](https://jwkunz.github.io/lib_jsl/book/foundations/images.html#netpbm).

use crate::png::RasterImage;
use lib_jsl_core::ErrorsJSL;
use std::path::Path;

fn corrupt(message: &'static str) -> ErrorsJSL {
    ErrorsJSL::RuntimeError(message)
}

struct Cursor<'a> {
    data: &'a [u8],
    position: usize,
}

impl Cursor<'_> {
    /// Skips whitespace and comments, then reads one whitespace-delimited token.
    fn token(&mut self) -> Result<&[u8], ErrorsJSL> {
        loop {
            match self.data.get(self.position) {
                Some(b'#') => {
                    while self.data.get(self.position).is_some_and(|&c| c != b'\n') {
                        self.position += 1;
                    }
                }
                Some(c) if c.is_ascii_whitespace() => self.position += 1,
                Some(_) => break,
                None => return Err(corrupt("Netpbm data ended early.")),
            }
        }
        let start = self.position;
        while self.data.get(self.position).is_some_and(|c| !c.is_ascii_whitespace() && *c != b'#') {
            self.position += 1;
        }
        Ok(&self.data[start..self.position])
    }

    fn number(&mut self) -> Result<usize, ErrorsJSL> {
        std::str::from_utf8(self.token()?).ok().and_then(|t| t.parse().ok()).ok_or(corrupt("Bad number in Netpbm data."))
    }
}

/// Decodes any Netpbm image (P1 to P6).
pub fn ppm_decode(data: &[u8]) -> Result<RasterImage, ErrorsJSL> {
    let mut cursor = Cursor { data, position: 0 };
    let magic = cursor.token()?.to_vec();
    let kind = match magic.as_slice() {
        [b'P', k @ b'1'..=b'6'] => k - b'0',
        _ => return Err(corrupt("Not a Netpbm image.")),
    };
    let (width, height) = (cursor.number()?, cursor.number()?);
    let bitmap = kind == 1 || kind == 4;
    let max = if bitmap { 1 } else { cursor.number()? };
    if max == 0 || max > 65535 {
        return Err(corrupt("The Netpbm maximum value must be 1 to 65535."));
    }
    let channels = if kind == 3 || kind == 6 { 3 } else { 1 };
    // Every sample takes at least one bit of what remains (one character in the text formats), so
    // a header promising more is rejected before anything is allocated.
    let count = width.checked_mul(height).and_then(|n| n.checked_mul(channels)).ok_or(corrupt("The Netpbm image is too large."))?;
    if count / 8 > data.len().saturating_sub(cursor.position) {
        return Err(corrupt("Netpbm data ended early."));
    }
    let mut samples = Vec::with_capacity(count);
    if kind <= 3 {
        for _ in 0..count {
            if kind == 1 {
                // Bitmaps may pack digits without separators.
                let digit = loop {
                    match data.get(cursor.position) {
                        Some(b'0') | Some(b'1') => break data[cursor.position],
                        Some(b'#') => {
                            cursor.token()?;
                            cursor.position -= 1;
                            continue;
                        }
                        Some(_) => cursor.position += 1,
                        None => return Err(corrupt("Netpbm data ended early.")),
                    }
                };
                cursor.position += 1;
                samples.push(digit as usize - b'0' as usize);
            } else {
                samples.push(cursor.number()?);
            }
        }
    } else {
        cursor.position += 1; // the single whitespace after the header
        let body = &data[cursor.position.min(data.len())..];
        if kind == 4 {
            let row_bytes = width.div_ceil(8);
            if body.len() < row_bytes * height {
                return Err(corrupt("Netpbm data ended early."));
            }
            for y in 0..height {
                for x in 0..width {
                    samples.push(((body[y * row_bytes + x / 8] >> (7 - x % 8)) & 1) as usize);
                }
            }
        } else if max < 256 {
            if body.len() < count {
                return Err(corrupt("Netpbm data ended early."));
            }
            samples.extend(body[..count].iter().map(|&b| b as usize));
        } else {
            if body.len() < 2 * count {
                return Err(corrupt("Netpbm data ended early."));
            }
            samples.extend((0..count).map(|i| u16::from_be_bytes([body[2 * i], body[2 * i + 1]]) as usize));
        }
    }
    if samples.iter().any(|&s| s > max) {
        return Err(corrupt("A Netpbm sample exceeds the maximum value."));
    }
    // Rescale to 8 bits (or keep 16-bit data at 16 bits); bitmaps map 1 (black) to 0.
    let (depth, target) = if max > 255 { (16, 65535usize) } else { (8, 255usize) };
    let values = samples
        .into_iter()
        .map(|s| if bitmap { (1 - s) as u16 * 255 } else { ((s * target + max / 2) / max) as u16 })
        .collect();
    RasterImage::new(width, height, channels, depth, values)
}

/// Encodes a one-channel image as binary PGM (P5) or a three-channel image as binary PPM (P6),
/// with maximum value 255 or 65535 according to the bit depth.
pub fn ppm_encode(image: &RasterImage) -> Result<Vec<u8>, ErrorsJSL> {
    let magic = match image.channels {
        1 => "P5",
        3 => "P6",
        _ => return Err(ErrorsJSL::InvalidInputRange("PGM/PPM need one or three channels (drop alpha first).")),
    };
    let max = if image.bit_depth == 16 { 65535 } else { 255 };
    let mut out = format!("{magic}\n{} {}\n{max}\n", image.width, image.height).into_bytes();
    if image.bit_depth == 16 {
        out.extend(image.samples.iter().flat_map(|s| s.to_be_bytes()));
    } else {
        out.extend(image.samples.iter().map(|&s| s as u8));
    }
    Ok(out)
}

/// Reads a Netpbm file.
pub fn read_ppm_file(path: impl AsRef<Path>) -> Result<RasterImage, ErrorsJSL> {
    let data = std::fs::read(path).map_err(|_| ErrorsJSL::RuntimeError("Could not read the Netpbm file."))?;
    ppm_decode(&data)
}

/// Writes a binary PGM or PPM file.
pub fn write_ppm_file(path: impl AsRef<Path>, image: &RasterImage) -> Result<(), ErrorsJSL> {
    std::fs::write(path, ppm_encode(image)?).map_err(|_| ErrorsJSL::RuntimeError("Could not write the Netpbm file."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headers_promising_more_data_than_the_file_holds_are_errors() {
        // These used to try to allocate hundreds of gigabytes and abort the process.
        assert!(ppm_decode(b"P5\n200000 200000\n255\nabc").is_err());
        assert!(ppm_decode(b"P2\n200000 200000\n255\n1 2 3").is_err());
        assert!(ppm_decode(b"P6\n4294967296 4294967296\n255\n").is_err());
    }

    #[test]
    fn reads_ascii_variants_with_comments() {
        let ppm = b"P3\n# a comment\n2 1\n# another\n255\n255 0 0   0 128 255\n";
        let image = ppm_decode(ppm).unwrap();
        assert_eq!((image.width, image.height, image.channels), (2, 1, 3));
        assert_eq!(image.samples, vec![255, 0, 0, 0, 128, 255]);
        // Maximum 15 rescales to 8 bits.
        let pgm = ppm_decode(b"P2 3 1 15 0 15 7").unwrap();
        assert_eq!(pgm.samples, vec![0, 255, 119]);
        // Bitmap digits may run together; 1 is black.
        let pbm = ppm_decode(b"P1\n4 2\n0101\n1 1 0 0\n").unwrap();
        assert_eq!(pbm.samples, vec![255, 0, 255, 0, 0, 0, 255, 255]);
    }

    #[test]
    fn binary_round_trips_and_packed_bitmaps() {
        let rgb = RasterImage::new(5, 3, 3, 8, (0..45).map(|i| (i * 5) as u16).collect()).unwrap();
        assert_eq!(ppm_decode(&ppm_encode(&rgb).unwrap()).unwrap(), rgb);
        let deep = RasterImage::new(4, 2, 1, 16, (0..8).map(|i| i * 9000).collect()).unwrap();
        assert_eq!(ppm_decode(&ppm_encode(&deep).unwrap()).unwrap(), deep);
        // P4: 10 pixels per row packed MSB first and padded to whole bytes.
        let mut p4 = b"P4\n10 1\n".to_vec();
        p4.extend([0b1010_0000, 0b0100_0000]);
        assert_eq!(ppm_decode(&p4).unwrap().samples, vec![0, 255, 0, 255, 255, 255, 255, 255, 255, 0]);
        assert!(ppm_decode(b"P6 2 2 255 abc").is_err() && ppm_decode(b"P7 1 1 1 0").is_err());
        assert!(ppm_encode(&RasterImage::new(1, 1, 4, 8, vec![1, 2, 3, 4]).unwrap()).is_err());
    }
}
