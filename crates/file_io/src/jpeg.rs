//! A simple baseline JPEG codec.
//!
//! This is a small, readable implementation intended as a worked example rather than a
//! production codec. It supports:
//!
//! - **Encoding** 8-bit grayscale or RGB images as baseline sequential JPEG (JFIF), using the
//!   standard Annex K quantization and Huffman tables, a quality factor in `1..=100`, and
//!   4:4:4 chroma (no subsampling).
//! - **Decoding** baseline sequential (SOF0) JPEG files with one or three components, 8-bit
//!   precision, any chroma subsampling (upsampled by replication), custom Huffman and
//!   quantization tables, and restart intervals.
//!
//! Progressive, arithmetic-coded, 12-bit, CMYK and multi-scan files are rejected with an error.
//!
//! The pipeline is the classic one: colour conversion, 8x8 forward DCT, quantization, zig-zag
//! ordering, differential DC coding and run-length AC coding, then Huffman coding; the decoder
//! runs it in reverse.

use lib_jsl_core::ErrorsJSL;
use std::f64::consts::PI;

/// A decoded raster image with interleaved 8-bit samples (`channels` is 1 for grayscale or 3 for
/// RGB, row-major, `width * height * channels` bytes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JpegImage {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub pixels: Vec<u8>,
}

// ---------------------------------------------------------------- tables

const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
    28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
    54, 47, 55, 62, 63,
];

const LUMA_QUANT: [u8; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51,
    87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
];

const CHROMA_QUANT: [u8; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99,
];

/// A Huffman table in JPEG form: counts of codes per length (1..=16) and the symbols in order.
struct HuffSpec {
    counts: [u8; 16],
    symbols: &'static [u8],
}

const DC_LUMA: HuffSpec = HuffSpec {
    counts: [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    symbols: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
};
const DC_CHROMA: HuffSpec = HuffSpec {
    counts: [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    symbols: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
};
const AC_LUMA: HuffSpec = HuffSpec {
    counts: [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d],
    symbols: &[
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71,
        0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72,
        0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37,
        0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83,
        0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3,
        0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
        0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
        0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
    ],
};
const AC_CHROMA: HuffSpec = HuffSpec {
    counts: [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77],
    symbols: &[
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22,
        0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1,
        0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36,
        0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
        0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a,
        0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a,
        0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba,
        0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
        0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
    ],
};

// ------------------------------------------------------------------- DCT

/// `COS[x][u] = c(u)/2 * cos((2x+1) u π / 16)`, so that the 2-D DCT is `C^T · block · C`.
fn dct_basis() -> [[f64; 8]; 8] {
    let mut c = [[0.0; 8]; 8];
    for (x, row) in c.iter_mut().enumerate() {
        for (u, v) in row.iter_mut().enumerate() {
            let scale = if u == 0 { (1.0f64 / 8.0).sqrt() } else { 0.5 };
            *v = scale * ((2 * x + 1) as f64 * u as f64 * PI / 16.0).cos();
        }
    }
    c
}

/// Forward 8x8 DCT-II (orthonormal).
fn fdct(block: &[f64; 64]) -> [f64; 64] {
    let c = dct_basis();
    let mut tmp = [0.0; 64];
    for y in 0..8 {
        for u in 0..8 {
            tmp[y * 8 + u] = (0..8).map(|x| block[y * 8 + x] * c[x][u]).sum();
        }
    }
    let mut out = [0.0; 64];
    for u in 0..8 {
        for v in 0..8 {
            out[v * 8 + u] = (0..8).map(|y| tmp[y * 8 + u] * c[y][v]).sum();
        }
    }
    out
}

/// Inverse 8x8 DCT.
fn idct(coef: &[f64; 64]) -> [f64; 64] {
    let c = dct_basis();
    let mut tmp = [0.0; 64];
    for v in 0..8 {
        for x in 0..8 {
            tmp[v * 8 + x] = (0..8).map(|u| coef[v * 8 + u] * c[x][u]).sum();
        }
    }
    let mut out = [0.0; 64];
    for x in 0..8 {
        for y in 0..8 {
            out[y * 8 + x] = (0..8).map(|v| tmp[v * 8 + x] * c[y][v]).sum();
        }
    }
    out
}

// --------------------------------------------------------------- encoder

fn scaled_quant(base: &[u8; 64], quality: u8) -> [u8; 64] {
    let q = quality.clamp(1, 100) as i32;
    let scale = if q < 50 { 5000 / q } else { 200 - 2 * q };
    let mut out = [0u8; 64];
    for (o, &b) in out.iter_mut().zip(base) {
        *o = ((b as i32 * scale + 50) / 100).clamp(1, 255) as u8;
    }
    out
}

/// Canonical Huffman code assignment: `codes[symbol] = (code, length)`.
fn encoder_codes(spec: &HuffSpec) -> Vec<Option<(u16, u8)>> {
    let mut table = vec![None; 256];
    let mut code = 0u16;
    let mut k = 0;
    for len in 1..=16u8 {
        for _ in 0..spec.counts[len as usize - 1] {
            table[spec.symbols[k] as usize] = Some((code, len));
            code += 1;
            k += 1;
        }
        code <<= 1;
    }
    table
}

struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    nbits: u32,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter { out: Vec::new(), acc: 0, nbits: 0 }
    }

    fn put(&mut self, value: u32, len: u32) {
        if len == 0 {
            return;
        }
        self.acc = (self.acc << len) | (value & ((1u32 << len) - 1));
        self.nbits += len;
        while self.nbits >= 8 {
            let byte = (self.acc >> (self.nbits - 8)) as u8;
            self.out.push(byte);
            if byte == 0xFF {
                self.out.push(0x00); // byte stuffing
            }
            self.nbits -= 8;
        }
        self.acc &= (1u32 << self.nbits) - 1;
    }

    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            let pad = 8 - self.nbits;
            self.put((1u32 << pad) - 1, pad); // pad with 1 bits
        }
        self.out
    }
}

/// Size category (number of bits) of a coefficient magnitude.
fn category(v: i32) -> u32 {
    32 - v.unsigned_abs().leading_zeros()
}

/// The `category` low bits that follow a Huffman symbol for value `v`.
fn magnitude_bits(v: i32, cat: u32) -> u32 {
    if v >= 0 { v as u32 } else { (v + (1 << cat) - 1) as u32 }
}

struct ComponentEncoder<'a> {
    quant: &'a [u8; 64],
    dc: &'a [Option<(u16, u8)>],
    ac: &'a [Option<(u16, u8)>],
    prev_dc: i32,
}

impl ComponentEncoder<'_> {
    fn encode_block(&mut self, samples: &[f64; 64], w: &mut BitWriter) {
        let mut shifted = [0.0; 64];
        for (s, &v) in shifted.iter_mut().zip(samples) {
            *s = v - 128.0;
        }
        let coef = fdct(&shifted);
        let mut q = [0i32; 64];
        for (i, &z) in ZIGZAG.iter().enumerate() {
            q[i] = (coef[z] / self.quant[z] as f64).round() as i32;
        }
        // DC difference.
        let diff = q[0] - self.prev_dc;
        self.prev_dc = q[0];
        let cat = category(diff);
        let (code, len) = self.dc[cat as usize].expect("DC symbol present in table");
        w.put(code as u32, len as u32);
        w.put(magnitude_bits(diff, cat), cat);
        // AC coefficients: (run, size) symbols.
        let mut run = 0;
        for &v in &q[1..] {
            if v == 0 {
                run += 1;
                continue;
            }
            while run >= 16 {
                let (code, len) = self.ac[0xF0].unwrap();
                w.put(code as u32, len as u32);
                run -= 16;
            }
            let cat = category(v);
            let (code, len) = self.ac[(run << 4) | cat as usize].expect("AC symbol present in table");
            w.put(code as u32, len as u32);
            w.put(magnitude_bits(v, cat), cat);
            run = 0;
        }
        if run > 0 {
            let (code, len) = self.ac[0x00].unwrap();
            w.put(code as u32, len as u32);
        }
    }
}

fn put_marker(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    out.extend_from_slice(&[0xFF, marker]);
    out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(payload);
}

fn put_dht(out: &mut Vec<u8>, class_and_id: u8, spec: &HuffSpec) {
    let mut payload = vec![class_and_id];
    payload.extend_from_slice(&spec.counts);
    payload.extend_from_slice(spec.symbols);
    put_marker(out, 0xC4, &payload);
}

fn put_dqt(out: &mut Vec<u8>, id: u8, table: &[u8; 64]) {
    let mut payload = vec![id];
    payload.extend(ZIGZAG.iter().map(|&z| table[z]));
    put_marker(out, 0xDB, &payload);
}

/// Encodes an 8-bit image as a baseline JPEG (JFIF) byte stream.
///
/// `pixels` holds `width * height * channels` interleaved samples, row-major, with `channels`
/// equal to 1 (grayscale) or 3 (RGB). `quality` runs from 1 (smallest file) to 100 (best fidelity).
/// Dimensions must be between 1 and 65535.
pub fn jpeg_encode(pixels: &[u8], width: usize, height: usize, channels: usize, quality: u8) -> Result<Vec<u8>, ErrorsJSL> {
    if channels != 1 && channels != 3 {
        return Err(ErrorsJSL::InvalidInputRange("JPEG encoding supports 1 (gray) or 3 (RGB) channels."));
    }
    if width == 0 || height == 0 || width > 65535 || height > 65535 {
        return Err(ErrorsJSL::InvalidInputRange("JPEG dimensions must be between 1 and 65535."));
    }
    if pixels.len() != width * height * channels {
        return Err(ErrorsJSL::IncompatibleArraySizes((pixels.len(), width * height * channels)));
    }
    let luma_q = scaled_quant(&LUMA_QUANT, quality);
    let chroma_q = scaled_quant(&CHROMA_QUANT, quality);

    // Split into planes of f64 samples (Y, Cb, Cr with the JFIF full-range conversion).
    let n = width * height;
    let mut planes: Vec<Vec<f64>> = vec![Vec::with_capacity(n); channels];
    for px in pixels.chunks_exact(channels) {
        if channels == 1 {
            planes[0].push(px[0] as f64);
        } else {
            let (r, g, b) = (px[0] as f64, px[1] as f64, px[2] as f64);
            planes[0].push(0.299 * r + 0.587 * g + 0.114 * b);
            planes[1].push(-0.168_736 * r - 0.331_264 * g + 0.5 * b + 128.0);
            planes[2].push(0.5 * r - 0.418_688 * g - 0.081_312 * b + 128.0);
        }
    }

    let dc_luma = encoder_codes(&DC_LUMA);
    let ac_luma = encoder_codes(&AC_LUMA);
    let dc_chroma = encoder_codes(&DC_CHROMA);
    let ac_chroma = encoder_codes(&AC_CHROMA);

    let mut out = vec![0xFF, 0xD8]; // SOI
    // JFIF APP0: version 1.01, no density units, 1:1 aspect, no thumbnail.
    put_marker(&mut out, 0xE0, &[b'J', b'F', b'I', b'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]);
    put_dqt(&mut out, 0, &luma_q);
    if channels == 3 {
        put_dqt(&mut out, 1, &chroma_q);
    }
    // SOF0: precision 8, height, width, components (id, sampling 1x1, quant table).
    let mut sof = vec![8];
    sof.extend_from_slice(&(height as u16).to_be_bytes());
    sof.extend_from_slice(&(width as u16).to_be_bytes());
    sof.push(channels as u8);
    for c in 0..channels {
        sof.extend_from_slice(&[c as u8 + 1, 0x11, if c == 0 { 0 } else { 1 }]);
    }
    put_marker(&mut out, 0xC0, &sof);
    put_dht(&mut out, 0x00, &DC_LUMA);
    put_dht(&mut out, 0x10, &AC_LUMA);
    if channels == 3 {
        put_dht(&mut out, 0x01, &DC_CHROMA);
        put_dht(&mut out, 0x11, &AC_CHROMA);
    }
    // SOS: all components, DC/AC table selectors, spectral range 0..63, no approximation.
    let mut sos = vec![channels as u8];
    for c in 0..channels {
        sos.extend_from_slice(&[c as u8 + 1, if c == 0 { 0x00 } else { 0x11 }]);
    }
    sos.extend_from_slice(&[0, 63, 0]);
    put_marker(&mut out, 0xDA, &sos);

    let mut encoders: Vec<ComponentEncoder> = (0..channels)
        .map(|c| ComponentEncoder {
            quant: if c == 0 { &luma_q } else { &chroma_q },
            dc: if c == 0 { &dc_luma } else { &dc_chroma },
            ac: if c == 0 { &ac_luma } else { &ac_chroma },
            prev_dc: 0,
        })
        .collect();
    let mut writer = BitWriter::new();
    for by in 0..height.div_ceil(8) {
        for bx in 0..width.div_ceil(8) {
            for (c, enc) in encoders.iter_mut().enumerate() {
                let mut block = [0.0; 64];
                for y in 0..8 {
                    for x in 0..8 {
                        // Replicate edge pixels into the padding.
                        let sx = (bx * 8 + x).min(width - 1);
                        let sy = (by * 8 + y).min(height - 1);
                        block[y * 8 + x] = planes[c][sy * width + sx];
                    }
                }
                enc.encode_block(&block, &mut writer);
            }
        }
    }
    out.extend(writer.finish());
    out.extend_from_slice(&[0xFF, 0xD9]); // EOI
    Ok(out)
}

// --------------------------------------------------------------- decoder

struct HuffDecoder {
    /// For each code length 1..=16: smallest code, largest code (or -1), index of first symbol.
    min_code: [i32; 17],
    max_code: [i32; 18],
    val_ptr: [i32; 17],
    symbols: Vec<u8>,
}

impl HuffDecoder {
    fn new(counts: &[u8; 16], symbols: &[u8]) -> Result<Self, ErrorsJSL> {
        let total: usize = counts.iter().map(|&c| c as usize).sum();
        if total != symbols.len() || total > 256 {
            return Err(ErrorsJSL::InvalidInputRange("Malformed JPEG Huffman table."));
        }
        let mut min_code = [0i32; 17];
        let mut max_code = [-1i32; 18];
        let mut val_ptr = [0i32; 17];
        let mut code = 0i32;
        let mut k = 0i32;
        for len in 1..=16usize {
            val_ptr[len] = k;
            min_code[len] = code;
            code += counts[len - 1] as i32;
            k += counts[len - 1] as i32;
            max_code[len] = if counts[len - 1] > 0 { code - 1 } else { -1 };
            if code > (1 << len) {
                return Err(ErrorsJSL::InvalidInputRange("Over-subscribed JPEG Huffman table."));
            }
            code <<= 1;
        }
        Ok(HuffDecoder { min_code, max_code, val_ptr, symbols: symbols.to_vec() })
    }
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    acc: u32,
    nbits: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        BitReader { data, pos, acc: 0, nbits: 0 }
    }

    fn bit(&mut self) -> Result<u32, ErrorsJSL> {
        if self.nbits == 0 {
            let b = *self.data.get(self.pos).ok_or(ErrorsJSL::InvalidInputRange("JPEG scan data ended unexpectedly."))?;
            if b == 0xFF {
                match self.data.get(self.pos + 1) {
                    Some(0x00) => self.pos += 2,
                    _ => return Err(ErrorsJSL::InvalidInputRange("Unexpected marker inside JPEG scan data.")),
                }
            } else {
                self.pos += 1;
            }
            self.acc = b as u32;
            self.nbits = 8;
        }
        self.nbits -= 1;
        Ok((self.acc >> self.nbits) & 1)
    }

    fn bits(&mut self, n: u32) -> Result<u32, ErrorsJSL> {
        let mut v = 0;
        for _ in 0..n {
            v = (v << 1) | self.bit()?;
        }
        Ok(v)
    }

    fn decode(&mut self, h: &HuffDecoder) -> Result<u8, ErrorsJSL> {
        let mut code = 0i32;
        for len in 1..=16usize {
            code = (code << 1) | self.bit()? as i32;
            if h.max_code[len] >= 0 && code <= h.max_code[len] && code >= h.min_code[len] {
                let idx = (h.val_ptr[len] + code - h.min_code[len]) as usize;
                return Ok(h.symbols[idx]);
            }
        }
        Err(ErrorsJSL::InvalidInputRange("Invalid JPEG Huffman code."))
    }

    /// Discards any partial byte and consumes an `RSTn` marker.
    fn restart(&mut self) -> Result<(), ErrorsJSL> {
        self.nbits = 0;
        if self.data.get(self.pos) == Some(&0xFF) && matches!(self.data.get(self.pos + 1), Some(0xD0..=0xD7)) {
            self.pos += 2;
            Ok(())
        } else {
            Err(ErrorsJSL::InvalidInputRange("Missing JPEG restart marker."))
        }
    }
}

/// Sign-extends an `n`-bit magnitude as in JPEG (the inverse of [`magnitude_bits`]).
fn extend(v: u32, n: u32) -> i32 {
    if n == 0 {
        0
    } else if v < (1 << (n - 1)) {
        v as i32 - (1 << n) + 1
    } else {
        v as i32
    }
}

struct Component {
    id: u8,
    h: usize,
    v: usize,
    quant_id: usize,
    dc_id: usize,
    ac_id: usize,
    plane: Vec<f64>,
    plane_w: usize,
}

fn read_u16(data: &[u8], pos: usize) -> Result<usize, ErrorsJSL> {
    match data.get(pos..pos + 2) {
        Some(b) => Ok(u16::from_be_bytes([b[0], b[1]]) as usize),
        None => Err(ErrorsJSL::InvalidInputRange("Truncated JPEG header.")),
    }
}

/// Decodes a baseline JPEG byte stream into an interleaved 8-bit image (grayscale stays 1
/// channel; three-component YCbCr files become RGB).
pub fn jpeg_decode(data: &[u8]) -> Result<JpegImage, ErrorsJSL> {
    if data.get(0..2) != Some(&[0xFF, 0xD8]) {
        return Err(ErrorsJSL::InvalidInputRange("Not a JPEG file (missing SOI)."));
    }
    let mut pos = 2;
    let mut quant: [Option<[u16; 64]>; 4] = [None; 4];
    let mut dc_tables: [Option<HuffDecoder>; 4] = [None, None, None, None];
    let mut ac_tables: [Option<HuffDecoder>; 4] = [None, None, None, None];
    let mut comps: Vec<Component> = Vec::new();
    let (mut width, mut height) = (0usize, 0usize);
    let mut restart_interval = 0usize;
    let mut scan_start = None;

    while scan_start.is_none() {
        if data.get(pos) != Some(&0xFF) {
            return Err(ErrorsJSL::InvalidInputRange("Expected a JPEG marker."));
        }
        while data.get(pos) == Some(&0xFF) {
            pos += 1; // skip fill bytes
        }
        let marker = *data.get(pos).ok_or(ErrorsJSL::InvalidInputRange("Truncated JPEG."))?;
        pos += 1;
        if marker == 0xD9 {
            return Err(ErrorsJSL::InvalidInputRange("JPEG ended before any image data."));
        }
        if marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
            continue; // standalone markers
        }
        let len = read_u16(data, pos)?;
        let seg = data.get(pos + 2..pos + len).ok_or(ErrorsJSL::InvalidInputRange("Truncated JPEG segment."))?;
        match marker {
            0xDB => {
                let mut i = 0;
                while i < seg.len() {
                    let (prec, id) = (seg[i] >> 4, (seg[i] & 15) as usize);
                    i += 1;
                    if id > 3 || prec > 1 {
                        return Err(ErrorsJSL::InvalidInputRange("Unsupported JPEG quantization table."));
                    }
                    let mut t = [0u16; 64];
                    for k in 0..64 {
                        let v = if prec == 0 {
                            *seg.get(i).ok_or(ErrorsJSL::InvalidInputRange("Truncated DQT."))? as u16
                        } else {
                            read_u16(seg, i)? as u16
                        };
                        i += if prec == 0 { 1 } else { 2 };
                        t[ZIGZAG[k]] = v;
                    }
                    quant[id] = Some(t);
                }
            }
            0xC4 => {
                let mut i = 0;
                while i + 17 <= seg.len() {
                    let (class, id) = (seg[i] >> 4, (seg[i] & 15) as usize);
                    let mut counts = [0u8; 16];
                    counts.copy_from_slice(&seg[i + 1..i + 17]);
                    let total: usize = counts.iter().map(|&c| c as usize).sum();
                    let symbols = seg.get(i + 17..i + 17 + total).ok_or(ErrorsJSL::InvalidInputRange("Truncated DHT."))?;
                    if id > 3 || class > 1 {
                        return Err(ErrorsJSL::InvalidInputRange("Unsupported JPEG Huffman table id."));
                    }
                    let table = HuffDecoder::new(&counts, symbols)?;
                    if class == 0 { dc_tables[id] = Some(table) } else { ac_tables[id] = Some(table) }
                    i += 17 + total;
                }
            }
            0xC0 | 0xC1 => {
                if seg.len() < 6 || seg[0] != 8 {
                    return Err(ErrorsJSL::InvalidInputRange("Only 8-bit baseline JPEG is supported."));
                }
                height = read_u16(seg, 1)?;
                width = read_u16(seg, 3)?;
                let n = seg[5] as usize;
                if (n != 1 && n != 3) || seg.len() < 6 + 3 * n || width == 0 || height == 0 {
                    return Err(ErrorsJSL::InvalidInputRange("Unsupported JPEG component layout."));
                }
                comps = (0..n)
                    .map(|i| {
                        let s = &seg[6 + 3 * i..9 + 3 * i];
                        Component {
                            id: s[0],
                            h: (s[1] >> 4) as usize,
                            v: (s[1] & 15) as usize,
                            quant_id: (s[2] & 3) as usize,
                            dc_id: 0,
                            ac_id: 0,
                            plane: Vec::new(),
                            plane_w: 0,
                        }
                    })
                    .collect();
                if comps.iter().any(|c| c.h == 0 || c.v == 0 || c.h > 4 || c.v > 4) {
                    return Err(ErrorsJSL::InvalidInputRange("Invalid JPEG sampling factors."));
                }
            }
            0xC2 | 0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
                return Err(ErrorsJSL::InvalidInputRange("Only baseline sequential Huffman JPEG is supported."));
            }
            0xDD => restart_interval = read_u16(seg, 0)?,
            0xDA => {
                if comps.is_empty() {
                    return Err(ErrorsJSL::InvalidInputRange("JPEG scan before frame header."));
                }
                let ns = *seg.first().ok_or(ErrorsJSL::InvalidInputRange("Truncated SOS."))? as usize;
                if ns != comps.len() || seg.len() < 1 + 2 * ns + 3 {
                    return Err(ErrorsJSL::InvalidInputRange("Only single-scan JPEG files are supported."));
                }
                for i in 0..ns {
                    let (cid, tabs) = (seg[1 + 2 * i], seg[2 + 2 * i]);
                    let comp = comps.iter_mut().find(|c| c.id == cid).ok_or(ErrorsJSL::InvalidInputRange("Unknown component in SOS."))?;
                    comp.dc_id = (tabs >> 4) as usize & 3;
                    comp.ac_id = (tabs & 15) as usize & 3;
                }
                scan_start = Some(pos + len);
            }
            _ => {}
        }
        pos += len;
    }

    let hmax = comps.iter().map(|c| c.h).max().unwrap();
    let vmax = comps.iter().map(|c| c.v).max().unwrap();
    let mcu_w = 8 * hmax;
    let mcu_h = 8 * vmax;
    let mcus_x = width.div_ceil(mcu_w);
    let mcus_y = height.div_ceil(mcu_h);
    // Every 8x8 block costs at least two bits of scan data (a DC code and an end-of-block code),
    // so a frame header claiming more blocks than the data could hold is rejected before the
    // planes are allocated.
    let scan = scan_start.ok_or(ErrorsJSL::InvalidInputRange("JPEG file has no scan."))?;
    let blocks = mcus_x * mcus_y * comps.iter().map(|c| c.h * c.v).sum::<usize>();
    if blocks / 4 > data.len().saturating_sub(scan) {
        return Err(ErrorsJSL::InvalidInputRange("JPEG scan data are shorter than the frame needs."));
    }
    for c in comps.iter_mut() {
        c.plane_w = mcus_x * c.h * 8;
        c.plane = vec![0.0; c.plane_w * mcus_y * c.v * 8];
    }

    let mut reader = BitReader::new(data, scan);
    let mut prev_dc = vec![0i32; comps.len()];
    let mut mcu_count = 0usize;
    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            if restart_interval > 0 && mcu_count > 0 && mcu_count.is_multiple_of(restart_interval) {
                reader.restart()?;
                prev_dc.iter_mut().for_each(|p| *p = 0);
            }
            mcu_count += 1;
            for (ci, comp) in comps.iter_mut().enumerate() {
                let q = quant[comp.quant_id].ok_or(ErrorsJSL::InvalidInputRange("Missing JPEG quantization table."))?;
                let dc = dc_tables[comp.dc_id].as_ref().ok_or(ErrorsJSL::InvalidInputRange("Missing JPEG DC table."))?;
                let ac = ac_tables[comp.ac_id].as_ref().ok_or(ErrorsJSL::InvalidInputRange("Missing JPEG AC table."))?;
                for bv in 0..comp.v {
                    for bh in 0..comp.h {
                        let mut coef = [0.0; 64];
                        let cat = reader.decode(dc)? as u32;
                        if cat > 11 {
                            return Err(ErrorsJSL::InvalidInputRange("Invalid JPEG DC category."));
                        }
                        prev_dc[ci] += extend(reader.bits(cat)?, cat);
                        coef[0] = (prev_dc[ci] * q[0] as i32) as f64;
                        let mut k = 1;
                        while k < 64 {
                            let sym = reader.decode(ac)?;
                            let (run, size) = ((sym >> 4) as usize, (sym & 15) as u32);
                            if size == 0 {
                                if run == 15 {
                                    k += 16;
                                    continue;
                                }
                                break; // end of block
                            }
                            k += run;
                            if k > 63 {
                                return Err(ErrorsJSL::InvalidInputRange("JPEG AC run exceeds the block."));
                            }
                            let z = ZIGZAG[k];
                            coef[z] = (extend(reader.bits(size)?, size) * q[z] as i32) as f64;
                            k += 1;
                        }
                        let samples = idct(&coef);
                        let x0 = (mx * comp.h + bh) * 8;
                        let y0 = (my * comp.v + bv) * 8;
                        for y in 0..8 {
                            for x in 0..8 {
                                comp.plane[(y0 + y) * comp.plane_w + x0 + x] = samples[y * 8 + x] + 128.0;
                            }
                        }
                    }
                }
            }
        }
    }

    // Upsample (replication) and colour-convert.
    let channels = comps.len();
    let mut pixels = vec![0u8; width * height * channels];
    let clamp = |v: f64| v.round().clamp(0.0, 255.0) as u8;
    for y in 0..height {
        for x in 0..width {
            let sample = |c: &Component| c.plane[(y * c.v / vmax) * c.plane_w + x * c.h / hmax];
            let o = (y * width + x) * channels;
            if channels == 1 {
                pixels[o] = clamp(sample(&comps[0]));
            } else {
                let (yy, cb, cr) = (sample(&comps[0]), sample(&comps[1]) - 128.0, sample(&comps[2]) - 128.0);
                pixels[o] = clamp(yy + 1.402 * cr);
                pixels[o + 1] = clamp(yy - 0.344_136 * cb - 0.714_136 * cr);
                pixels[o + 2] = clamp(yy + 1.772 * cb);
            }
        }
    }
    Ok(JpegImage { width, height, channels, pixels })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_larger_than_its_scan_data_is_an_error() {
        // A 65535 x 65535 frame header on a tiny file used to allocate 100 GB of planes.
        let mut bytes = jpeg_encode(&[128u8; 64], 8, 8, 1, 90).unwrap();
        let sof = bytes.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
        bytes[sof + 5..sof + 9].copy_from_slice(&[0xFF; 4]);
        assert!(jpeg_decode(&bytes).is_err());
    }

    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        let mse: f64 = a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).powi(2)).sum::<f64>() / a.len() as f64;
        if mse == 0.0 { f64::INFINITY } else { 10.0 * (255.0f64 * 255.0 / mse).log10() }
    }

    /// A smooth synthetic test image with gradients and a few edges.
    fn test_image(w: usize, h: usize, channels: usize) -> Vec<u8> {
        let mut p = Vec::with_capacity(w * h * channels);
        for y in 0..h {
            for x in 0..w {
                let base = (x * 255 / w.max(1)) as f64 * 0.6 + (y * 255 / h.max(1)) as f64 * 0.4;
                let edge = if (x / 8 + y / 8) % 4 == 0 { 40.0 } else { 0.0 };
                let v = (base + edge).min(255.0);
                if channels == 1 {
                    p.push(v as u8);
                } else {
                    p.push(v as u8);
                    p.push((255.0 - v) as u8);
                    p.push(((x * 7 + y * 3) % 256) as u8);
                }
            }
        }
        p
    }

    #[test]
    fn standard_huffman_tables_are_complete() {
        // AC tables must cover (run 0..=15, size 1..=10) plus EOB (0x00) and ZRL (0xF0).
        let mut expected: Vec<u8> = vec![0x00, 0xF0];
        for run in 0..16u8 {
            for size in 1..=10u8 {
                expected.push(run << 4 | size);
            }
        }
        expected.sort_unstable();
        for spec in [&AC_LUMA, &AC_CHROMA] {
            assert_eq!(spec.counts.iter().map(|&c| c as usize).sum::<usize>(), spec.symbols.len());
            let mut got = spec.symbols.to_vec();
            got.sort_unstable();
            assert_eq!(got, expected);
        }
        for spec in [&DC_LUMA, &DC_CHROMA] {
            assert_eq!(spec.counts.iter().map(|&c| c as usize).sum::<usize>(), 12);
        }
        // All tables must be valid prefix codes.
        for spec in [&DC_LUMA, &DC_CHROMA, &AC_LUMA, &AC_CHROMA] {
            assert!(HuffDecoder::new(&spec.counts, spec.symbols).is_ok());
        }
    }

    #[test]
    fn dct_round_trip_and_dc_gain() {
        let mut block = [0.0; 64];
        for (i, v) in block.iter_mut().enumerate() {
            *v = ((i * 37) % 101) as f64 - 50.0;
        }
        let back = idct(&fdct(&block));
        assert!(block.iter().zip(&back).all(|(a, b)| (a - b).abs() < 1e-9));
        // A constant block has only a DC term equal to 8 * value.
        let flat = fdct(&[10.0; 64]);
        assert!((flat[0] - 80.0).abs() < 1e-9 && flat[1..].iter().all(|c| c.abs() < 1e-9));
    }

    #[test]
    fn magnitude_coding_round_trips() {
        for v in -2047..=2047 {
            let cat = category(v);
            assert_eq!(extend(magnitude_bits(v, cat), cat), v);
        }
        assert_eq!(category(0), 0);
        assert_eq!(category(-1), 1);
        assert_eq!(category(255), 8);
    }

    #[test]
    fn gray_round_trip_quality() {
        let img = test_image(37, 29, 1); // not a multiple of 8
        let jpg = jpeg_encode(&img, 37, 29, 1, 90).unwrap();
        assert_eq!(&jpg[..2], &[0xFF, 0xD8]);
        assert_eq!(&jpg[jpg.len() - 2..], &[0xFF, 0xD9]);
        let dec = jpeg_decode(&jpg).unwrap();
        assert_eq!((dec.width, dec.height, dec.channels), (37, 29, 1));
        assert!(psnr(&img, &dec.pixels) > 35.0, "psnr {}", psnr(&img, &dec.pixels));
    }

    #[test]
    fn rgb_round_trip_quality_and_size_tradeoff() {
        let img = test_image(48, 40, 3);
        let high = jpeg_encode(&img, 48, 40, 3, 95).unwrap();
        let low = jpeg_encode(&img, 48, 40, 3, 20).unwrap();
        assert!(low.len() < high.len());
        let dh = jpeg_decode(&high).unwrap();
        let dl = jpeg_decode(&low).unwrap();
        assert_eq!(dh.channels, 3);
        assert!(psnr(&img, &dh.pixels) > psnr(&img, &dl.pixels));
        assert!(psnr(&img, &dh.pixels) > 28.0, "psnr {}", psnr(&img, &dh.pixels));
        assert!(img.len() > high.len(), "should compress a smooth image");
    }

    #[test]
    fn flat_image_is_exact_and_tiny() {
        let img = vec![200u8; 16 * 16];
        let jpg = jpeg_encode(&img, 16, 16, 1, 75).unwrap();
        assert!(jpg.len() < 700);
        let dec = jpeg_decode(&jpg).unwrap();
        assert!(dec.pixels.iter().all(|&p| (p as i32 - 200).abs() <= 1));
    }

    #[test]
    fn noisy_image_exercises_long_codes_and_byte_stuffing() {
        let mut state = 12345u64;
        let img: Vec<u8> = (0..64 * 64)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 56) as u8
            })
            .collect();
        let jpg = jpeg_encode(&img, 64, 64, 1, 100).unwrap();
        let dec = jpeg_decode(&jpg).unwrap();
        assert!(psnr(&img, &dec.pixels) > 38.0, "psnr {}", psnr(&img, &dec.pixels));
    }

    /// Smooth test pattern used to produce the libjpeg fixtures below:
    /// R rises left to right, G rises top to bottom, B is constant.
    fn gradient(w: usize, h: usize) -> Vec<u8> {
        let mut p = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                p.extend_from_slice(&[(x * 255 / (w - 1)) as u8, (y * 255 / (h - 1)) as u8, 128]);
            }
        }
        p
    }

    // Both files were written by libjpeg (Pillow / ImageMagick) from `gradient`, quality 90.
    //  - LIBJPEG_420_16X16: 4:2:0 chroma subsampling.
    //  - LIBJPEG_RESTART_24X16: 4:4:4 with a restart interval of 2 MCUs.
    const LIBJPEG_420_16X16: &[u8] = &[
    0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
    0x00, 0x01, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43, 0x00, 0x03, 0x02, 0x02, 0x03, 0x02, 0x02, 0x03,
    0x03, 0x03, 0x03, 0x04, 0x03, 0x03, 0x04, 0x05, 0x08, 0x05, 0x05, 0x04, 0x04, 0x05, 0x0a, 0x07,
    0x07, 0x06, 0x08, 0x0c, 0x0a, 0x0c, 0x0c, 0x0b, 0x0a, 0x0b, 0x0b, 0x0d, 0x0e, 0x12, 0x10, 0x0d,
    0x0e, 0x11, 0x0e, 0x0b, 0x0b, 0x10, 0x16, 0x10, 0x11, 0x13, 0x14, 0x15, 0x15, 0x15, 0x0c, 0x0f,
    0x17, 0x18, 0x16, 0x14, 0x18, 0x12, 0x14, 0x15, 0x14, 0xff, 0xdb, 0x00, 0x43, 0x01, 0x03, 0x04,
    0x04, 0x05, 0x04, 0x05, 0x09, 0x05, 0x05, 0x09, 0x14, 0x0d, 0x0b, 0x0d, 0x14, 0x14, 0x14, 0x14,
    0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
    0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
    0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0xff, 0xc0,
    0x00, 0x11, 0x08, 0x00, 0x10, 0x00, 0x10, 0x03, 0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11,
    0x01, 0xff, 0xc4, 0x00, 0x1f, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
    0x0a, 0x0b, 0xff, 0xc4, 0x00, 0xb5, 0x10, 0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05,
    0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7d, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21,
    0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23,
    0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17,
    0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a,
    0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a,
    0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a,
    0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99,
    0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7,
    0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5,
    0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1,
    0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xff, 0xc4, 0x00, 0x1f, 0x01, 0x00, 0x03,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
    0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0xff, 0xc4, 0x00, 0xb5, 0x11, 0x00,
    0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04, 0x07, 0x05, 0x04, 0x04, 0x00, 0x01, 0x02, 0x77, 0x00,
    0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13,
    0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15,
    0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26, 0x27,
    0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88,
    0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6,
    0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4,
    0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9,
    0xfa, 0xff, 0xda, 0x00, 0x0c, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3f, 0x00, 0xf9,
    0x7b, 0xc2, 0x3f, 0x09, 0x3e, 0xe7, 0xee, 0x7f, 0x4a, 0xf6, 0xcf, 0x08, 0x7c, 0x24, 0xfb, 0x9f,
    0xb9, 0xfd, 0x2b, 0xda, 0x3c, 0x21, 0xf0, 0x93, 0x84, 0xfd, 0xcf, 0xe9, 0x5e, 0xd9, 0xe1, 0x0f,
    0x84, 0x9c, 0xa7, 0xee, 0x7f, 0x4a, 0x79, 0xd7, 0x19, 0xef, 0xef, 0x07, 0x86, 0x3e, 0x20, 0xff,
    0x00, 0x0f, 0xdf, 0xec, 0x7f, 0xff, 0xd9,
    ];

    const LIBJPEG_RESTART_24X16: &[u8] = &[
    0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
    0x00, 0x01, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43, 0x00, 0x03, 0x02, 0x02, 0x03, 0x02, 0x02, 0x03,
    0x03, 0x03, 0x03, 0x04, 0x03, 0x03, 0x04, 0x05, 0x08, 0x05, 0x05, 0x04, 0x04, 0x05, 0x0a, 0x07,
    0x07, 0x06, 0x08, 0x0c, 0x0a, 0x0c, 0x0c, 0x0b, 0x0a, 0x0b, 0x0b, 0x0d, 0x0e, 0x12, 0x10, 0x0d,
    0x0e, 0x11, 0x0e, 0x0b, 0x0b, 0x10, 0x16, 0x10, 0x11, 0x13, 0x14, 0x15, 0x15, 0x15, 0x0c, 0x0f,
    0x17, 0x18, 0x16, 0x14, 0x18, 0x12, 0x14, 0x15, 0x14, 0xff, 0xdb, 0x00, 0x43, 0x01, 0x03, 0x04,
    0x04, 0x05, 0x04, 0x05, 0x09, 0x05, 0x05, 0x09, 0x14, 0x0d, 0x0b, 0x0d, 0x14, 0x14, 0x14, 0x14,
    0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
    0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
    0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0xff, 0xc0,
    0x00, 0x11, 0x08, 0x00, 0x10, 0x00, 0x18, 0x03, 0x01, 0x11, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11,
    0x01, 0xff, 0xc4, 0x00, 0x16, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0x06, 0x08, 0xff, 0xc4, 0x00, 0x1d, 0x10, 0x00, 0x00,
    0x06, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05,
    0x07, 0x23, 0x31, 0xa1, 0x06, 0x32, 0x41, 0x51, 0xff, 0xc4, 0x00, 0x18, 0x01, 0x01, 0x01, 0x00,
    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x07, 0x05,
    0x08, 0x09, 0xff, 0xc4, 0x00, 0x1a, 0x11, 0x00, 0x02, 0x02, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05, 0x04, 0x31, 0x06, 0x21, 0x22, 0xff, 0xdd,
    0x00, 0x04, 0x00, 0x02, 0xff, 0xda, 0x00, 0x0c, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00,
    0x3f, 0x00, 0xc9, 0x84, 0x09, 0x1c, 0x33, 0xe7, 0x05, 0x96, 0x43, 0xab, 0xd8, 0x51, 0x2e, 0x43,
    0x5d, 0x08, 0xf8, 0xfa, 0x47, 0x0c, 0xd0, 0x2f, 0x21, 0xd5, 0xec, 0xbb, 0x25, 0xc8, 0x6b, 0xa3,
    0xff, 0xd0, 0x9a, 0xc7, 0xd2, 0x3d, 0x59, 0xf3, 0x83, 0x2b, 0x21, 0xd5, 0xec, 0xdc, 0x14, 0xb9,
    0x0d, 0x74, 0x23, 0x90, 0x24, 0x70, 0xcd, 0x02, 0xf2, 0x1d, 0x5e, 0xce, 0x3d, 0xa5, 0xc8, 0x6b,
    0xa3, 0xff, 0xd1, 0x69, 0x20, 0x48, 0xe1, 0x9a, 0x01, 0x24, 0x3a, 0xbd, 0x93, 0x24, 0xb9, 0x0d,
    0x74, 0x23, 0x90, 0x24, 0x7a, 0xb3, 0x40, 0xbc, 0x87, 0x57, 0xb2, 0xee, 0x97, 0x21, 0xae, 0x8f,
    0xff, 0xd9,
    ];

    #[test]
    fn decodes_libjpeg_subsampled_file() {
        let img = jpeg_decode(LIBJPEG_420_16X16).unwrap();
        assert_eq!((img.width, img.height, img.channels), (16, 16, 3));
        let want = gradient(16, 16);
        assert!(psnr(&img.pixels, &want) > 30.0, "psnr {}", psnr(&img.pixels, &want));
    }

    #[test]
    fn decodes_libjpeg_restart_intervals() {
        assert!(LIBJPEG_RESTART_24X16.windows(2).any(|w| w == [0xFF, 0xDD]), "fixture has a DRI segment");
        let img = jpeg_decode(LIBJPEG_RESTART_24X16).unwrap();
        assert_eq!((img.width, img.height, img.channels), (24, 16, 3));
        let want = gradient(24, 16);
        assert!(psnr(&img.pixels, &want) > 35.0, "psnr {}", psnr(&img.pixels, &want));
    }

    #[test]
    fn rejects_malformed_input() {
        assert!(jpeg_decode(&[]).is_err());
        assert!(jpeg_decode(b"not a jpeg").is_err());
        assert!(jpeg_decode(&LIBJPEG_420_16X16[..60]).is_err()); // truncated in the headers
        let mut corrupt = LIBJPEG_RESTART_24X16.to_vec();
        let sos = corrupt.windows(2).position(|w| w == [0xFF, 0xDA]).unwrap();
        corrupt.truncate(sos + 20);
        assert!(jpeg_decode(&corrupt).is_err()); // scan data ends early
        let good = jpeg_encode(&test_image(16, 16, 1), 16, 16, 1, 80).unwrap();
        assert!(jpeg_decode(&good[..good.len() / 2]).is_err());
        assert!(jpeg_encode(&[0; 10], 4, 4, 1, 50).is_err());
        assert!(jpeg_encode(&[0; 16], 4, 4, 2, 50).is_err());
        assert!(jpeg_encode(&[], 0, 4, 1, 50).is_err());
    }
}
