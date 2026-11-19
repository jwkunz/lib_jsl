//! MATLAB level-5 `.mat` files (SciPy's `io.loadmat`, `io.savemat` and `io.whosmat`).
//!
//! A level-5 file is a 128-byte header (descriptive text, a version and an endian indicator)
//! followed by data elements, each a tag (type and byte count) and its data padded to 8 bytes;
//! small elements of at most 4 bytes pack the tag and data into 8 bytes. Each variable is a
//! `miMATRIX` element holding array flags (class, complex, logical), dimensions, a name and the
//! data, or a `miCOMPRESSED` element holding one zlib-compressed `miMATRIX`. Numeric data are
//! stored column-major in any integer or floating type.
//!
//! Reading supports both byte orders, compression, the numeric, logical, character, sparse,
//! cell and structure classes, and nesting; writing produces little-endian files, optionally
//! compressed. MATLAB objects, function handles and v7.3 (HDF5) files are not supported.
//!
//! See the book entry [Scientific data files](https://jwkunz.github.io/lib_jsl/book/foundations/scientific-files.html#matlab-level-5-files).

use crate::deflate::{zlib_compress, zlib_decompress};
use lib_jsl_core::ErrorsJSL;
use std::path::Path;

/// The MATLAB class of a numeric array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MatClass {
    Double,
    Single,
    Int8,
    Uint8,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Int64,
    Uint64,
}

impl MatClass {
    fn code(self) -> u8 {
        match self {
            MatClass::Double => 6,
            MatClass::Single => 7,
            MatClass::Int8 => 8,
            MatClass::Uint8 => 9,
            MatClass::Int16 => 10,
            MatClass::Uint16 => 11,
            MatClass::Int32 => 12,
            MatClass::Uint32 => 13,
            MatClass::Int64 => 14,
            MatClass::Uint64 => 15,
        }
    }

    fn from_code(c: u8) -> Option<Self> {
        Some(match c {
            6 => MatClass::Double,
            7 => MatClass::Single,
            8 => MatClass::Int8,
            9 => MatClass::Uint8,
            10 => MatClass::Int16,
            11 => MatClass::Uint16,
            12 => MatClass::Int32,
            13 => MatClass::Uint32,
            14 => MatClass::Int64,
            15 => MatClass::Uint64,
            _ => return None,
        })
    }

    /// The data type MATLAB stores this class in.
    fn mi_type(self) -> u32 {
        match self {
            MatClass::Double => MI_DOUBLE,
            MatClass::Single => MI_SINGLE,
            MatClass::Int8 => MI_INT8,
            MatClass::Uint8 => MI_UINT8,
            MatClass::Int16 => MI_INT16,
            MatClass::Uint16 => MI_UINT16,
            MatClass::Int32 => MI_INT32,
            MatClass::Uint32 => MI_UINT32,
            MatClass::Int64 => MI_INT64,
            MatClass::Uint64 => MI_UINT64,
        }
    }
}

/// A MATLAB value. Arrays are column-major with their MATLAB dimensions (at least two).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum MatValue {
    /// A numeric array; values converted to `f64` (64-bit integers beyond 2⁵³ lose precision).
    Numeric { class: MatClass, dims: Vec<usize>, real: Vec<f64>, imag: Option<Vec<f64>> },
    /// A logical array.
    Logical { dims: Vec<usize>, data: Vec<bool> },
    /// A character array, column-major; a row vector reads as a string with [`MatValue::as_string`].
    Char { dims: Vec<usize>, chars: Vec<char> },
    /// A sparse matrix in compressed sparse column form: row indices, column pointers (length
    /// `ncols + 1`) and values.
    Sparse { nrows: usize, ncols: usize, row_indices: Vec<usize>, col_ptr: Vec<usize>, real: Vec<f64>, imag: Option<Vec<f64>>, logical: bool },
    /// A cell array of values, column-major.
    Cell { dims: Vec<usize>, cells: Vec<MatValue> },
    /// A structure array: field names, and for each element (column-major) one value per field.
    Struct { dims: Vec<usize>, fields: Vec<String>, elements: Vec<Vec<MatValue>> },
}

impl MatValue {
    /// A real `f64` matrix from rows.
    pub fn from_rows(rows: &[Vec<f64>]) -> Self {
        let m = rows.len();
        let n = rows.first().map_or(0, |r| r.len());
        let real = (0..n).flat_map(|j| rows.iter().map(move |r| r[j])).collect();
        MatValue::Numeric { class: MatClass::Double, dims: vec![m, n], real, imag: None }
    }

    /// A real `f64` row vector.
    pub fn from_slice(v: &[f64]) -> Self {
        MatValue::Numeric { class: MatClass::Double, dims: vec![1, v.len()], real: v.to_vec(), imag: None }
    }

    /// A character row vector.
    pub fn from_text(s: &str) -> Self {
        let chars: Vec<char> = s.chars().collect();
        MatValue::Char { dims: vec![1, chars.len()], chars }
    }

    /// A character row vector (or a column of one character) as a string.
    pub fn as_string(&self) -> Option<String> {
        match self {
            MatValue::Char { dims, chars } if dims.iter().filter(|&&d| d != 1).count() <= 1 => Some(chars.iter().collect()),
            _ => None,
        }
    }

    /// A real numeric matrix as rows (`None` for other values or more than two dimensions).
    pub fn to_rows(&self) -> Option<Vec<Vec<f64>>> {
        match self {
            MatValue::Numeric { dims, real, .. } if dims.len() == 2 => Some((0..dims[0]).map(|i| (0..dims[1]).map(|j| real[i + j * dims[0]]).collect()).collect()),
            _ => None,
        }
    }
}

const MI_INT8: u32 = 1;
const MI_UINT8: u32 = 2;
const MI_INT16: u32 = 3;
const MI_UINT16: u32 = 4;
const MI_INT32: u32 = 5;
const MI_UINT32: u32 = 6;
const MI_SINGLE: u32 = 7;
const MI_DOUBLE: u32 = 9;
const MI_INT64: u32 = 12;
const MI_UINT64: u32 = 13;
const MI_MATRIX: u32 = 14;
const MI_COMPRESSED: u32 = 15;
const MI_UTF8: u32 = 16;
const MI_UTF16: u32 = 17;
const MI_UTF32: u32 = 18;

fn bad(msg: &'static str) -> ErrorsJSL {
    ErrorsJSL::InvalidInputRange(msg)
}

/// A cursor over little- or big-endian bytes.
struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
    swap: bool,
}

impl<'a> Reader<'a> {
    fn u32_at(&self, at: usize) -> Result<u32, ErrorsJSL> {
        let s = self.b.get(at..at + 4).ok_or(bad("MAT-file: truncated"))?;
        let a = [s[0], s[1], s[2], s[3]];
        Ok(if self.swap { u32::from_be_bytes(a) } else { u32::from_le_bytes(a) })
    }

    /// The next element: its type and data bytes; advances past the padding.
    fn element(&mut self) -> Result<(u32, &'a [u8]), ErrorsJSL> {
        let first = self.u32_at(self.pos)?;
        if first >> 16 != 0 {
            // Small data element: 2-byte count, 2-byte type, 4 bytes of data.
            let (ty, n) = (first & 0xffff, (first >> 16) as usize);
            if n > 4 {
                return Err(bad("MAT-file: bad small element"));
            }
            let data = self.b.get(self.pos + 4..self.pos + 4 + n).ok_or(bad("MAT-file: truncated"))?;
            self.pos += 8;
            return Ok((ty, data));
        }
        let n = self.u32_at(self.pos + 4)? as usize;
        let start = self.pos + 8;
        let data = self.b.get(start..start + n).ok_or(bad("MAT-file: truncated element"))?;
        // Compressed elements are not padded.
        self.pos = if first == MI_COMPRESSED { start + n } else { start + n.div_ceil(8) * 8 };
        Ok((first, data))
    }

    fn done(&self) -> bool {
        self.pos + 8 > self.b.len()
    }
}

/// Numbers of a given element type as `f64`.
fn numbers(ty: u32, d: &[u8], swap: bool) -> Result<Vec<f64>, ErrorsJSL> {
    macro_rules! conv {
        ($t:ty, $n:expr) => {
            d.chunks_exact($n)
                .map(|c| {
                    let a: [u8; $n] = c.try_into().expect("chunk size");
                    (if swap { <$t>::from_be_bytes(a) } else { <$t>::from_le_bytes(a) }) as f64
                })
                .collect()
        };
    }
    Ok(match ty {
        MI_INT8 => d.iter().map(|&v| v as i8 as f64).collect(),
        MI_UINT8 | MI_UTF8 => d.iter().map(|&v| v as f64).collect(),
        MI_INT16 => conv!(i16, 2),
        MI_UINT16 | MI_UTF16 => conv!(u16, 2),
        MI_INT32 => conv!(i32, 4),
        MI_UINT32 | MI_UTF32 => conv!(u32, 4),
        MI_SINGLE => conv!(f32, 4),
        MI_DOUBLE => conv!(f64, 8),
        MI_INT64 => conv!(i64, 8),
        MI_UINT64 => conv!(u64, 8),
        _ => return Err(bad("MAT-file: unsupported data type")),
    })
}

fn parse_matrix(data: &[u8], swap: bool) -> Result<(String, MatValue), ErrorsJSL> {
    let mut r = Reader { b: data, pos: 0, swap };
    if data.is_empty() {
        // An empty miMATRIX: an empty double array.
        return Ok((String::new(), MatValue::Numeric { class: MatClass::Double, dims: vec![0, 0], real: vec![], imag: None }));
    }
    let (_, flags) = r.element()?;
    let flags_r = Reader { b: flags, pos: 0, swap };
    let f = flags_r.u32_at(0)?;
    let class = (f & 0xff) as u8;
    let (complex, logical) = (f & 0x0800 != 0, f & 0x0200 != 0);
    let (ty, dd) = r.element()?;
    let dims: Vec<usize> = numbers(ty, dd, swap)?.into_iter().map(|v| v as usize).collect();
    let (_, nm) = r.element()?;
    let name = String::from_utf8_lossy(nm).into_owned();
    let count: usize = dims.iter().product();
    let value = match class {
        1 => {
            let mut cells = Vec::with_capacity(count);
            for _ in 0..count {
                let (ty, d) = r.element()?;
                if ty != MI_MATRIX {
                    return Err(bad("MAT-file: a cell is not a matrix"));
                }
                cells.push(parse_matrix(d, swap)?.1);
            }
            MatValue::Cell { dims, cells }
        }
        2 => {
            let (ty, d) = r.element()?;
            let len = *numbers(ty, d, swap)?.first().ok_or(bad("MAT-file: missing field name length"))? as usize;
            let (_, names) = r.element()?;
            let fields: Vec<String> = if len == 0 {
                Vec::new()
            } else {
                names.chunks(len).map(|c| String::from_utf8_lossy(c).trim_end_matches('\0').to_string()).collect()
            };
            let mut elements = Vec::with_capacity(count);
            for _ in 0..count {
                let mut vals = Vec::with_capacity(fields.len());
                for _ in 0..fields.len() {
                    let (_, d) = r.element()?;
                    vals.push(parse_matrix(d, swap)?.1);
                }
                elements.push(vals);
            }
            MatValue::Struct { dims, fields, elements }
        }
        4 => {
            let chars: Vec<char> = if count == 0 {
                Vec::new()
            } else {
                let (ty, d) = r.element()?;
                if ty == MI_UTF8 {
                    String::from_utf8_lossy(d).chars().collect()
                } else {
                    numbers(ty, d, swap)?.into_iter().map(|v| char::from_u32(v as u32).unwrap_or('\u{fffd}')).collect()
                }
            };
            MatValue::Char { dims, chars }
        }
        5 => {
            let (ty, d) = r.element()?;
            let ir: Vec<usize> = numbers(ty, d, swap)?.into_iter().map(|v| v as usize).collect();
            let (ty, d) = r.element()?;
            let jc: Vec<usize> = numbers(ty, d, swap)?.into_iter().map(|v| v as usize).collect();
            let nnz = *jc.last().unwrap_or(&0);
            let (ty, d) = r.element()?;
            let mut real = numbers(ty, d, swap)?;
            real.truncate(nnz);
            let imag = if complex {
                let (ty, d) = r.element()?;
                let mut im = numbers(ty, d, swap)?;
                im.truncate(nnz);
                Some(im)
            } else {
                None
            };
            MatValue::Sparse { nrows: dims[0], ncols: dims[1], row_indices: ir[..nnz.min(ir.len())].to_vec(), col_ptr: jc, real, imag, logical }
        }
        6..=15 => {
            let cls = MatClass::from_code(class).expect("numeric class");
            let real = if count == 0 && r.done() {
                Vec::new()
            } else {
                let (t, d) = r.element()?;
                numbers(t, d, swap)?
            };
            let imag = if complex {
                let (t, d) = r.element()?;
                Some(numbers(t, d, swap)?)
            } else {
                None
            };
            if logical {
                MatValue::Logical { dims, data: real.iter().map(|&v| v != 0.0).collect() }
            } else {
                MatValue::Numeric { class: cls, dims, real, imag }
            }
        }
        _ => return Err(bad("MAT-file: unsupported class (objects and function handles)")),
    };
    Ok((name, value))
}

// ANCHOR: loadmat
/// Parses a level-5 MAT-file (SciPy's `loadmat`): the variables in file order, as
/// `(name, value)` pairs. The header's endian indicator `IM` or `MI` gives the byte order.
pub fn mat_decode(bytes: &[u8]) -> Result<Vec<(String, MatValue)>, ErrorsJSL> {
    if bytes.len() < 128 {
        return Err(bad("MAT-file: shorter than the 128-byte header"));
    }
    let swap = match &bytes[126..128] {
        b"IM" => false,
        b"MI" => true,
        _ => return Err(bad("MAT-file: not a level-5 MAT-file (v7.3 files are HDF5)")),
    };
    let mut r = Reader { b: bytes, pos: 128, swap };
    let mut out = Vec::new();
    while !r.done() {
        let (ty, d) = r.element()?;
        let (name, value) = match ty {
            MI_MATRIX => parse_matrix(d, swap)?,
            MI_COMPRESSED => {
                let raw = zlib_decompress(d)?;
                let mut inner = Reader { b: &raw, pos: 0, swap };
                let (ty2, d2) = inner.element()?;
                if ty2 != MI_MATRIX {
                    return Err(bad("MAT-file: a compressed element does not hold a matrix"));
                }
                parse_matrix(d2, swap)?
            }
            _ => continue,
        };
        out.push((name, value));
    }
    Ok(out)
}
// ANCHOR_END: loadmat

/// The names, dimensions and classes of the variables (SciPy's `whosmat`), without keeping their
/// data.
pub fn mat_whos(bytes: &[u8]) -> Result<Vec<(String, Vec<usize>, String)>, ErrorsJSL> {
    Ok(mat_decode(bytes)?
        .into_iter()
        .map(|(name, v)| {
            let (dims, class) = match &v {
                MatValue::Numeric { class, dims, .. } => (dims.clone(), format!("{class:?}").to_lowercase()),
                MatValue::Logical { dims, .. } => (dims.clone(), "logical".into()),
                MatValue::Char { dims, .. } => (dims.clone(), "char".into()),
                MatValue::Sparse { nrows, ncols, .. } => (vec![*nrows, *ncols], "sparse".into()),
                MatValue::Cell { dims, .. } => (dims.clone(), "cell".into()),
                MatValue::Struct { dims, .. } => (dims.clone(), "struct".into()),
            };
            (name, dims, class)
        })
        .collect())
}

// ------------------------------------------------------------------------------------- writing

fn put_element(out: &mut Vec<u8>, ty: u32, data: &[u8]) {
    if data.len() <= 4 && ty != MI_MATRIX {
        out.extend_from_slice(&((data.len() as u32) << 16 | ty).to_le_bytes());
        let mut d = data.to_vec();
        d.resize(4, 0);
        out.extend_from_slice(&d);
        return;
    }
    out.extend_from_slice(&ty.to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(data);
    out.resize(out.len().next_multiple_of(8), 0);
}

fn encode_numbers(class: MatClass, v: &[f64]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 8);
    for &x in v {
        match class {
            MatClass::Double => b.extend_from_slice(&x.to_le_bytes()),
            MatClass::Single => b.extend_from_slice(&(x as f32).to_le_bytes()),
            MatClass::Int8 => b.push(x as i8 as u8),
            MatClass::Uint8 => b.push(x as u8),
            MatClass::Int16 => b.extend_from_slice(&(x as i16).to_le_bytes()),
            MatClass::Uint16 => b.extend_from_slice(&(x as u16).to_le_bytes()),
            MatClass::Int32 => b.extend_from_slice(&(x as i32).to_le_bytes()),
            MatClass::Uint32 => b.extend_from_slice(&(x as u32).to_le_bytes()),
            MatClass::Int64 => b.extend_from_slice(&(x as i64).to_le_bytes()),
            MatClass::Uint64 => b.extend_from_slice(&(x as u64).to_le_bytes()),
        }
    }
    b
}

fn i32s(v: &[usize]) -> Vec<u8> {
    v.iter().flat_map(|&x| (x as i32).to_le_bytes()).collect()
}

fn flags(class: u8, complex: bool, logical: bool, nzmax: usize) -> Vec<u8> {
    let f = class as u32 | if complex { 0x0800 } else { 0 } | if logical { 0x0200 } else { 0 };
    [f.to_le_bytes(), (nzmax as u32).to_le_bytes()].concat()
}

fn encode_matrix(name: &str, v: &MatValue) -> Result<Vec<u8>, ErrorsJSL> {
    let mut body = Vec::new();
    let check = |dims: &[usize], n: usize| if dims.iter().product::<usize>() == n && dims.len() >= 2 { Ok(()) } else { Err(bad("MAT-file: the dimensions do not match the data")) };
    match v {
        MatValue::Numeric { class, dims, real, imag } => {
            check(dims, real.len())?;
            put_element(&mut body, MI_UINT32, &flags(class.code(), imag.is_some(), false, 0));
            put_element(&mut body, MI_INT32, &i32s(dims));
            put_element(&mut body, MI_INT8, name.as_bytes());
            put_element(&mut body, class.mi_type(), &encode_numbers(*class, real));
            if let Some(im) = imag {
                check(dims, im.len())?;
                put_element(&mut body, class.mi_type(), &encode_numbers(*class, im));
            }
        }
        MatValue::Logical { dims, data } => {
            check(dims, data.len())?;
            put_element(&mut body, MI_UINT32, &flags(9, false, true, 0));
            put_element(&mut body, MI_INT32, &i32s(dims));
            put_element(&mut body, MI_INT8, name.as_bytes());
            put_element(&mut body, MI_UINT8, &data.iter().map(|&b| b as u8).collect::<Vec<_>>());
        }
        MatValue::Char { dims, chars } => {
            check(dims, chars.len())?;
            put_element(&mut body, MI_UINT32, &flags(4, false, false, 0));
            put_element(&mut body, MI_INT32, &i32s(dims));
            put_element(&mut body, MI_INT8, name.as_bytes());
            if chars.iter().any(|&c| c as u32 > 0xffff) {
                return Err(bad("MAT-file: characters beyond the Basic Multilingual Plane"));
            }
            put_element(&mut body, MI_UINT16, &chars.iter().flat_map(|&c| (c as u16).to_le_bytes()).collect::<Vec<_>>());
        }
        MatValue::Sparse { nrows, ncols, row_indices, col_ptr, real, imag, logical } => {
            if col_ptr.len() != ncols + 1 || row_indices.len() != real.len() || *col_ptr.last().unwrap_or(&0) != real.len() {
                return Err(bad("MAT-file: inconsistent sparse matrix"));
            }
            let nnz = real.len();
            put_element(&mut body, MI_UINT32, &flags(5, imag.is_some(), *logical, nnz.max(1)));
            put_element(&mut body, MI_INT32, &i32s(&[*nrows, *ncols]));
            put_element(&mut body, MI_INT8, name.as_bytes());
            put_element(&mut body, MI_INT32, &i32s(row_indices));
            put_element(&mut body, MI_INT32, &i32s(col_ptr));
            put_element(&mut body, MI_DOUBLE, &encode_numbers(MatClass::Double, real));
            if let Some(im) = imag {
                put_element(&mut body, MI_DOUBLE, &encode_numbers(MatClass::Double, im));
            }
        }
        MatValue::Cell { dims, cells } => {
            check(dims, cells.len())?;
            put_element(&mut body, MI_UINT32, &flags(1, false, false, 0));
            put_element(&mut body, MI_INT32, &i32s(dims));
            put_element(&mut body, MI_INT8, name.as_bytes());
            for c in cells {
                body.extend(encode_matrix("", c)?);
            }
        }
        MatValue::Struct { dims, fields, elements } => {
            check(dims, elements.len())?;
            put_element(&mut body, MI_UINT32, &flags(2, false, false, 0));
            put_element(&mut body, MI_INT32, &i32s(dims));
            put_element(&mut body, MI_INT8, name.as_bytes());
            let len = fields.iter().map(|f| f.len() + 1).max().unwrap_or(1).max(1);
            if len > 64 {
                return Err(bad("MAT-file: field names are limited to 63 characters"));
            }
            put_element(&mut body, MI_INT32, &(len as i32).to_le_bytes());
            let mut names = Vec::new();
            for f in fields {
                let mut b = f.as_bytes().to_vec();
                b.resize(len, 0);
                names.extend(b);
            }
            put_element(&mut body, MI_INT8, &names);
            for el in elements {
                if el.len() != fields.len() {
                    return Err(bad("MAT-file: a struct element has the wrong number of fields"));
                }
                for f in el {
                    body.extend(encode_matrix("", f)?);
                }
            }
        }
    }
    let mut out = Vec::new();
    put_element(&mut out, MI_MATRIX, &body);
    Ok(out)
}

// ANCHOR: savemat
/// Writes a level-5 MAT-file (SciPy's `savemat` with `format="5"`), little-endian, each variable
/// zlib-compressed if `compress` (SciPy's `do_compression`).
pub fn mat_encode(variables: &[(String, MatValue)], compress: bool) -> Result<Vec<u8>, ErrorsJSL> {
    let mut out = Vec::new();
    let mut text = b"MATLAB 5.0 MAT-file Platform: lib_jsl".to_vec();
    text.resize(116, b' ');
    out.extend(text);
    out.extend([0u8; 8]);
    out.extend(0x0100u16.to_le_bytes());
    out.extend(b"IM");
    for (name, v) in variables {
        if name.is_empty() || name.len() > 63 || !name.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(bad("MAT-file: variable names must start with a letter and use letters, digits and underscores"));
        }
        let m = encode_matrix(name, v)?;
        if compress {
            let z = zlib_compress(&m, 6);
            out.extend(MI_COMPRESSED.to_le_bytes());
            out.extend((z.len() as u32).to_le_bytes());
            out.extend(z);
        } else {
            out.extend(m);
        }
    }
    Ok(out)
}
// ANCHOR_END: savemat

/// Reads a MAT-file.
pub fn read_mat_file(path: impl AsRef<Path>) -> Result<Vec<(String, MatValue)>, ErrorsJSL> {
    mat_decode(&std::fs::read(path).map_err(|_| ErrorsJSL::RuntimeError("could not read the MAT-file"))?)
}

/// Writes a MAT-file.
pub fn write_mat_file(path: impl AsRef<Path>, variables: &[(String, MatValue)], compress: bool) -> Result<(), ErrorsJSL> {
    std::fs::write(path, mat_encode(variables, compress)?).map_err(|_| ErrorsJSL::RuntimeError("could not write the MAT-file"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_every_class() {
        let vars = vec![
            ("a".to_string(), MatValue::from_rows(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.5]])),
            ("z".into(), MatValue::Numeric { class: MatClass::Double, dims: vec![1, 2], real: vec![1.0, -2.0], imag: Some(vec![0.5, 3.0]) }),
            ("i16".into(), MatValue::Numeric { class: MatClass::Int16, dims: vec![2, 1], real: vec![-7.0, 300.0], imag: None }),
            ("flag".into(), MatValue::Logical { dims: vec![1, 3], data: vec![true, false, true] }),
            ("s".into(), MatValue::from_text("héllo")),
            ("sp".into(), MatValue::Sparse { nrows: 3, ncols: 2, row_indices: vec![0, 2, 1], col_ptr: vec![0, 2, 3], real: vec![1.0, 2.0, 3.0], imag: None, logical: false }),
            ("c".into(), MatValue::Cell { dims: vec![1, 2], cells: vec![MatValue::from_slice(&[1.0]), MatValue::from_text("x")] }),
            (
                "st".into(),
                MatValue::Struct { dims: vec![1, 1], fields: vec!["alpha".into(), "beta".into()], elements: vec![vec![MatValue::from_slice(&[2.5, 3.5]), MatValue::from_text("b")]] },
            ),
            ("e".into(), MatValue::Numeric { class: MatClass::Double, dims: vec![0, 0], real: vec![], imag: None }),
        ];
        for compress in [false, true] {
            let bytes = mat_encode(&vars, compress).unwrap();
            let back = mat_decode(&bytes).unwrap();
            assert_eq!(back, vars, "compress = {compress}");
        }
        assert_eq!(back_string(&vars), "héllo");
        assert_eq!(mat_whos(&mat_encode(&vars, false).unwrap()).unwrap()[3], ("flag".to_string(), vec![1, 3], "logical".to_string()));
    }

    fn back_string(v: &[(String, MatValue)]) -> String {
        v[4].1.as_string().unwrap()
    }
}
