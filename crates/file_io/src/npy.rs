//! NumPy `.npy` arrays (format versions 1.0, 2.0 and 3.0).
//!
//! A `.npy` file is the magic string `\x93NUMPY`, a version, a little-endian header length and a
//! Python-literal header dictionary (`descr`, `fortran_order`, `shape`), padded so the data
//! starts on a 64-byte boundary, followed by the raw array data. Reading supports booleans,
//! signed and unsigned integers of 1 to 8 bytes, 32- and 64-bit floats and 64- and 128-bit
//! complex numbers, in either byte order and in C or Fortran order; data are returned in C
//! (row-major) order. Writing produces version 1.0 C-order little-endian files.

use lib_jsl_core::ErrorsJSL;
use ndarray::{ArrayD, IxDyn, ShapeBuilder};
use num::Complex;
use std::path::Path;

/// The element data of an array, in C order.
#[derive(Debug, Clone, PartialEq)]
pub enum NpyData {
    Bool(Vec<bool>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    C64(Vec<Complex<f32>>),
    C128(Vec<Complex<f64>>),
}

impl NpyData {
    fn len(&self) -> usize {
        match self {
            NpyData::Bool(v) => v.len(),
            NpyData::I8(v) => v.len(),
            NpyData::I16(v) => v.len(),
            NpyData::I32(v) => v.len(),
            NpyData::I64(v) => v.len(),
            NpyData::U8(v) => v.len(),
            NpyData::U16(v) => v.len(),
            NpyData::U32(v) => v.len(),
            NpyData::U64(v) => v.len(),
            NpyData::F32(v) => v.len(),
            NpyData::F64(v) => v.len(),
            NpyData::C64(v) => v.len(),
            NpyData::C128(v) => v.len(),
        }
    }

    /// The `descr` code of the little-endian encoding.
    fn descr(&self) -> &'static str {
        match self {
            NpyData::Bool(_) => "|b1",
            NpyData::I8(_) => "|i1",
            NpyData::I16(_) => "<i2",
            NpyData::I32(_) => "<i4",
            NpyData::I64(_) => "<i8",
            NpyData::U8(_) => "|u1",
            NpyData::U16(_) => "<u2",
            NpyData::U32(_) => "<u4",
            NpyData::U64(_) => "<u8",
            NpyData::F32(_) => "<f4",
            NpyData::F64(_) => "<f8",
            NpyData::C64(_) => "<c8",
            NpyData::C128(_) => "<c16",
        }
    }
}

/// An n-dimensional array read from or written to `.npy`.
#[derive(Debug, Clone, PartialEq)]
pub struct NpyArray {
    /// The shape; empty for a scalar (one element).
    pub shape: Vec<usize>,
    pub data: NpyData,
}

impl NpyArray {
    pub fn new(shape: Vec<usize>, data: NpyData) -> Result<Self, ErrorsJSL> {
        let count: usize = shape.iter().product();
        if count != data.len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((data.len(), count)));
        }
        Ok(NpyArray { shape, data })
    }

    /// The array as `f64` (real types only; booleans become 0 or 1).
    pub fn to_f64(&self) -> Result<ArrayD<f64>, ErrorsJSL> {
        let values: Vec<f64> = match &self.data {
            NpyData::Bool(v) => v.iter().map(|&b| b as u8 as f64).collect(),
            NpyData::I8(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::I16(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::I32(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::I64(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::U8(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::U16(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::U32(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::U64(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::F32(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::F64(v) => v.clone(),
            _ => return Err(ErrorsJSL::InvalidInputRange("Complex arrays have no real f64 form.")),
        };
        ArrayD::from_shape_vec(IxDyn(&self.shape), values).map_err(|_| ErrorsJSL::RuntimeError("Shape mismatch."))
    }

    /// An `f64` array from ndarray (any memory layout).
    pub fn from_f64(array: &ArrayD<f64>) -> Self {
        NpyArray { shape: array.shape().to_vec(), data: NpyData::F64(array.iter().copied().collect()) }
    }
}

fn corrupt(message: &'static str) -> ErrorsJSL {
    ErrorsJSL::RuntimeError(message)
}

/// Extracts the value text after `'key':` in the header dictionary.
fn header_value<'a>(header: &'a str, key: &str) -> Result<&'a str, ErrorsJSL> {
    let pattern = format!("'{key}'");
    let start = header.find(&pattern).ok_or(corrupt("A .npy header key is missing."))? + pattern.len();
    let rest = header[start..].trim_start().strip_prefix(':').ok_or(corrupt("Malformed .npy header."))?.trim_start();
    Ok(rest)
}

/// Decodes `.npy` bytes.
pub fn npy_decode(bytes: &[u8]) -> Result<NpyArray, ErrorsJSL> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(corrupt("Not a .npy file."));
    }
    let (header_len, offset) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 if bytes.len() >= 12 => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12),
        _ => return Err(corrupt("Unsupported .npy version.")),
    };
    let header = std::str::from_utf8(bytes.get(offset..offset + header_len).ok_or(corrupt("Truncated .npy header."))?).map_err(|_| corrupt("Non-text .npy header."))?;
    let descr_text = header_value(header, "descr")?;
    let descr = descr_text.strip_prefix('\'').and_then(|s| s.split('\'').next()).ok_or(corrupt("Unsupported .npy dtype."))?;
    let fortran = header_value(header, "fortran_order")?.starts_with("True");
    let shape_text = header_value(header, "shape")?;
    let close = shape_text.find(')').ok_or(corrupt("Malformed .npy shape."))?;
    let shape: Vec<usize> = shape_text[1..close].split(',').map(str::trim).filter(|s| !s.is_empty()).map(|s| s.parse().map_err(|_| corrupt("Malformed .npy shape."))).collect::<Result<_, _>>()?;
    let count: usize = shape.iter().product();
    let (order, kind, size) = {
        let mut chars = descr.chars();
        let order = chars.next().ok_or(corrupt("Empty dtype."))?;
        let kind = chars.next().ok_or(corrupt("Empty dtype."))?;
        let size: usize = chars.as_str().parse().map_err(|_| corrupt("Unsupported .npy dtype."))?;
        (order, kind, size)
    };
    let big = order == '>';
    let body = bytes.get(offset + header_len..offset + header_len + count * size).ok_or(corrupt("Truncated .npy data."))?;
    macro_rules! read {
        ($t:ty, $n:expr) => {
            body.chunks_exact($n).map(|c| { let a: [u8; $n] = c.try_into().expect("exact"); if big { <$t>::from_be_bytes(a) } else { <$t>::from_le_bytes(a) } }).collect::<Vec<$t>>()
        };
    }
    let data = match (kind, size) {
        ('b', 1) => NpyData::Bool(body.iter().map(|&b| b != 0).collect()),
        ('i', 1) => NpyData::I8(read!(i8, 1)),
        ('i', 2) => NpyData::I16(read!(i16, 2)),
        ('i', 4) => NpyData::I32(read!(i32, 4)),
        ('i', 8) => NpyData::I64(read!(i64, 8)),
        ('u', 1) => NpyData::U8(body.to_vec()),
        ('u', 2) => NpyData::U16(read!(u16, 2)),
        ('u', 4) => NpyData::U32(read!(u32, 4)),
        ('u', 8) => NpyData::U64(read!(u64, 8)),
        ('f', 4) => NpyData::F32(read!(f32, 4)),
        ('f', 8) => NpyData::F64(read!(f64, 8)),
        ('c', 8) => NpyData::C64(read!(f32, 4).chunks_exact(2).map(|p| Complex::new(p[0], p[1])).collect()),
        ('c', 16) => NpyData::C128(read!(f64, 8).chunks_exact(2).map(|p| Complex::new(p[0], p[1])).collect()),
        _ => return Err(corrupt("Unsupported .npy dtype.")),
    };
    let data = if fortran && shape.len() > 1 { to_c_order(data, &shape) } else { data };
    NpyArray::new(shape, data)
}

/// Reorders Fortran-order elements into C order.
fn to_c_order(data: NpyData, shape: &[usize]) -> NpyData {
    fn reorder<T: Clone>(v: Vec<T>, shape: &[usize]) -> Vec<T> {
        let array = ArrayD::from_shape_vec(IxDyn(shape).f(), v).expect("shape matches");
        array.iter().cloned().collect()
    }
    match data {
        NpyData::Bool(v) => NpyData::Bool(reorder(v, shape)),
        NpyData::I8(v) => NpyData::I8(reorder(v, shape)),
        NpyData::I16(v) => NpyData::I16(reorder(v, shape)),
        NpyData::I32(v) => NpyData::I32(reorder(v, shape)),
        NpyData::I64(v) => NpyData::I64(reorder(v, shape)),
        NpyData::U8(v) => NpyData::U8(reorder(v, shape)),
        NpyData::U16(v) => NpyData::U16(reorder(v, shape)),
        NpyData::U32(v) => NpyData::U32(reorder(v, shape)),
        NpyData::U64(v) => NpyData::U64(reorder(v, shape)),
        NpyData::F32(v) => NpyData::F32(reorder(v, shape)),
        NpyData::F64(v) => NpyData::F64(reorder(v, shape)),
        NpyData::C64(v) => NpyData::C64(reorder(v, shape)),
        NpyData::C128(v) => NpyData::C128(reorder(v, shape)),
    }
}

/// Encodes an array as a version 1.0 `.npy` file (little-endian, C order).
pub fn npy_encode(array: &NpyArray) -> Result<Vec<u8>, ErrorsJSL> {
    if array.shape.iter().product::<usize>() != array.data.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((array.data.len(), array.shape.iter().product())));
    }
    let shape = match array.shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", array.shape[0]),
        _ => format!("({})", array.shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut header = format!("{{'descr': '{}', 'fortran_order': False, 'shape': {shape}, }}", array.data.descr());
    // Pad with spaces so that magic + version + length + header (with its newline) is a
    // multiple of 64 bytes.
    let total = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - total % 64) % 64));
    header.push('\n');
    if header.len() > u16::MAX as usize {
        return Err(ErrorsJSL::InvalidInputRange("The .npy header is too long for version 1.0."));
    }
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    match &array.data {
        NpyData::Bool(v) => out.extend(v.iter().map(|&b| b as u8)),
        NpyData::I8(v) => out.extend(v.iter().map(|&x| x as u8)),
        NpyData::I16(v) => out.extend(v.iter().flat_map(|x| x.to_le_bytes())),
        NpyData::I32(v) => out.extend(v.iter().flat_map(|x| x.to_le_bytes())),
        NpyData::I64(v) => out.extend(v.iter().flat_map(|x| x.to_le_bytes())),
        NpyData::U8(v) => out.extend_from_slice(v),
        NpyData::U16(v) => out.extend(v.iter().flat_map(|x| x.to_le_bytes())),
        NpyData::U32(v) => out.extend(v.iter().flat_map(|x| x.to_le_bytes())),
        NpyData::U64(v) => out.extend(v.iter().flat_map(|x| x.to_le_bytes())),
        NpyData::F32(v) => out.extend(v.iter().flat_map(|x| x.to_le_bytes())),
        NpyData::F64(v) => out.extend(v.iter().flat_map(|x| x.to_le_bytes())),
        NpyData::C64(v) => out.extend(v.iter().flat_map(|c| [c.re.to_le_bytes(), c.im.to_le_bytes()].concat())),
        NpyData::C128(v) => out.extend(v.iter().flat_map(|c| [c.re.to_le_bytes(), c.im.to_le_bytes()].concat())),
    }
    Ok(out)
}

/// Reads a `.npy` file.
pub fn read_npy_file(path: impl AsRef<Path>) -> Result<NpyArray, ErrorsJSL> {
    let data = std::fs::read(path).map_err(|_| ErrorsJSL::RuntimeError("Could not read the .npy file."))?;
    npy_decode(&data)
}

/// Writes a `.npy` file.
pub fn write_npy_file(path: impl AsRef<Path>, array: &NpyArray) -> Result<(), ErrorsJSL> {
    std::fs::write(path, npy_encode(array)?).map_err(|_| ErrorsJSL::RuntimeError("Could not write the .npy file."))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Files written by NumPy 2.3 (`np.save`).
    const F8_C_2X3: &[u8] = &[
        147, 78, 85, 77, 80, 89, 1, 0, 118, 0, 123, 39, 100, 101, 115, 99, 114, 39, 58, 32, 39, 60,
        102, 56, 39, 44, 32, 39, 102, 111, 114, 116, 114, 97, 110, 95, 111, 114, 100, 101, 114, 39,
        58, 32, 70, 97, 108, 115, 101, 44, 32, 39, 115, 104, 97, 112, 101, 39, 58, 32, 40, 50, 44,
        32, 51, 41, 44, 32, 125, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 10, 0, 0, 0, 0, 0,
        0, 0, 192, 0, 0, 0, 0, 0, 0, 224, 191, 0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 4, 64,
        0, 0, 0, 0, 0, 0, 16, 64, 0, 0, 0, 0, 0, 0, 22, 64,
    ];
    const I4_BIG_FORTRAN_2X3: &[u8] = &[
        147, 78, 85, 77, 80, 89, 1, 0, 118, 0, 123, 39, 100, 101, 115, 99, 114, 39, 58, 32, 39, 62,
        105, 52, 39, 44, 32, 39, 102, 111, 114, 116, 114, 97, 110, 95, 111, 114, 100, 101, 114, 39,
        58, 32, 84, 114, 117, 101, 44, 32, 39, 115, 104, 97, 112, 101, 39, 58, 32, 40, 50, 44, 32,
        51, 41, 44, 32, 125, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 10, 0, 0, 0, 1, 0,
        0, 156, 64, 255, 255, 255, 254, 255, 255, 255, 251, 0, 0, 0, 3, 0, 0, 0, 6,
    ];
    const C16_3: &[u8] = &[
        147, 78, 85, 77, 80, 89, 1, 0, 118, 0, 123, 39, 100, 101, 115, 99, 114, 39, 58, 32, 39, 60,
        99, 49, 54, 39, 44, 32, 39, 102, 111, 114, 116, 114, 97, 110, 95, 111, 114, 100, 101, 114,
        39, 58, 32, 70, 97, 108, 115, 101, 44, 32, 39, 115, 104, 97, 112, 101, 39, 58, 32, 40, 51,
        44, 41, 44, 32, 125, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 10, 0, 0, 0, 0, 0,
        0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 224, 191,
        0, 0, 0, 0, 0, 0, 8, 64, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    const BOOL_4: &[u8] = &[
        147, 78, 85, 77, 80, 89, 1, 0, 118, 0, 123, 39, 100, 101, 115, 99, 114, 39, 58, 32, 39, 124,
        98, 49, 39, 44, 32, 39, 102, 111, 114, 116, 114, 97, 110, 95, 111, 114, 100, 101, 114, 39,
        58, 32, 70, 97, 108, 115, 101, 44, 32, 39, 115, 104, 97, 112, 101, 39, 58, 32, 40, 52, 44,
        41, 44, 32, 125, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 10, 1, 0, 0, 1,
    ];
    const U1_SCALAR: &[u8] = &[
        147, 78, 85, 77, 80, 89, 1, 0, 118, 0, 123, 39, 100, 101, 115, 99, 114, 39, 58, 32, 39, 124,
        117, 49, 39, 44, 32, 39, 102, 111, 114, 116, 114, 97, 110, 95, 111, 114, 100, 101, 114, 39,
        58, 32, 70, 97, 108, 115, 101, 44, 32, 39, 115, 104, 97, 112, 101, 39, 58, 32, 40, 41, 44,
        32, 125, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 10, 7,
    ];
    #[test]
    fn reads_files_written_by_numpy() {
        let a = npy_decode(F8_C_2X3).unwrap();
        assert_eq!(a.shape, vec![2, 3]);
        assert_eq!(a.data, NpyData::F64(vec![-2.0, -0.5, 1.0, 2.5, 4.0, 5.5]));
        // Big-endian, Fortran order: returned in C order.
        let b = npy_decode(I4_BIG_FORTRAN_2X3).unwrap();
        assert_eq!(b.data, NpyData::I32(vec![1, -2, 3, 40000, -5, 6]));
        let c = npy_decode(C16_3).unwrap();
        assert_eq!(c.data, NpyData::C128(vec![Complex::new(1.0, 2.0), Complex::new(0.0, -0.5), Complex::new(3.0, 0.0)]));
        assert_eq!(npy_decode(BOOL_4).unwrap().data, NpyData::Bool(vec![true, false, false, true]));
        let s = npy_decode(U1_SCALAR).unwrap();
        assert!(s.shape.is_empty() && s.data == NpyData::U8(vec![7]));
        assert_eq!(a.to_f64().unwrap()[[1, 2]], 5.5);
    }

    #[test]
    fn writes_files_numpy_layout_and_round_trips() {
        // Our encoding of NumPy's own array reproduces NumPy's bytes exactly.
        let a = npy_decode(F8_C_2X3).unwrap();
        assert_eq!(npy_encode(&a).unwrap(), F8_C_2X3);
        assert_eq!(npy_encode(&npy_decode(BOOL_4).unwrap()).unwrap(), BOOL_4);
        let arrays = [
            NpyArray::new(vec![2, 2, 2], NpyData::I16((0..8).map(|i| i * -300).collect())).unwrap(),
            NpyArray::new(vec![3], NpyData::C64(vec![Complex::new(1.5, -2.0); 3])).unwrap(),
            NpyArray::new(vec![0, 4], NpyData::F32(vec![])).unwrap(),
            NpyArray::new(vec![], NpyData::U64(vec![u64::MAX])).unwrap(),
        ];
        for array in &arrays {
            let bytes = npy_encode(array).unwrap();
            assert_eq!((bytes.len() - array_bytes(array)) % 64, 0);
            assert_eq!(&npy_decode(&bytes).unwrap(), array);
        }
        let from = NpyArray::from_f64(&ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap().reversed_axes());
        assert_eq!(from.data, NpyData::F64(vec![1.0, 3.0, 2.0, 4.0]));
        assert!(NpyArray::new(vec![2, 2], NpyData::F64(vec![1.0])).is_err() && npy_decode(b"\x93NUMPY").is_err());
    }

    fn array_bytes(a: &NpyArray) -> usize {
        let size = match &a.data {
            NpyData::Bool(_) | NpyData::I8(_) | NpyData::U8(_) => 1,
            NpyData::I16(_) | NpyData::U16(_) => 2,
            NpyData::I32(_) | NpyData::U32(_) | NpyData::F32(_) => 4,
            NpyData::C64(_) | NpyData::I64(_) | NpyData::U64(_) | NpyData::F64(_) => 8,
            NpyData::C128(_) => 16,
        };
        a.data.len() * size
    }
}
