//! Matrix Market exchange files (SciPy's `io.mmread`, `io.mmwrite` and `io.mminfo`).
//!
//! A Matrix Market file is text: a banner `%%MatrixMarket matrix <format> <field> <symmetry>`,
//! comment lines starting with `%`, a size line, and the entries. The `coordinate` format lists
//! `row column value` triplets (1-based) of a sparse matrix; the `array` format lists every entry
//! of a dense matrix in column-major order. The field is `real`, `integer`, `complex` or
//! `pattern` (no values); the symmetry `general`, `symmetric`, `skew-symmetric` or `hermitian`,
//! in which case only the lower triangle is stored.
//!
//! See the book entry [Scientific data files](https://jwkunz.github.io/lib_jsl/book/foundations/scientific-files.html#matrix-market).

use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;
use num_complex::Complex;
use std::path::Path;

type C64 = Complex<f64>;

/// Sparse triplets or a dense array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MmFormat {
    Coordinate,
    Array,
}

/// The type of the entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MmField {
    Real,
    Integer,
    Complex,
    /// Only the positions of the non-zeros (coordinate format).
    Pattern,
}

/// Which part of the matrix is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MmSymmetry {
    General,
    Symmetric,
    SkewSymmetric,
    Hermitian,
}

/// A matrix read from (or to be written to) a Matrix Market file. The entries are always the
/// full matrix — symmetric files are expanded on reading, as SciPy's `mmread` does — as 0-based
/// triplets; values are complex so one type serves every field (the imaginary part is zero
/// unless the field is complex, and the value is 1 for pattern files).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct MatrixMarket {
    pub nrows: usize,
    pub ncols: usize,
    pub format: MmFormat,
    pub field: MmField,
    pub symmetry: MmSymmetry,
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub values: Vec<C64>,
    /// The comment lines, without their leading `%`.
    pub comment: String,
}

impl MatrixMarket {
    /// A sparse real matrix from 0-based triplets (general, real).
    pub fn from_triplets(nrows: usize, ncols: usize, rows: Vec<usize>, cols: Vec<usize>, values: Vec<f64>) -> Result<Self, ErrorsJSL> {
        if rows.len() != cols.len() || rows.len() != values.len() {
            return Err(ErrorsJSL::IncompatibleArraySizes((rows.len(), values.len())));
        }
        if rows.iter().any(|&r| r >= nrows) || cols.iter().any(|&c| c >= ncols) {
            return Err(ErrorsJSL::InvalidInputRange("MatrixMarket: an index is out of range"));
        }
        Ok(MatrixMarket {
            nrows,
            ncols,
            format: MmFormat::Coordinate,
            field: MmField::Real,
            symmetry: MmSymmetry::General,
            rows,
            cols,
            values: values.into_iter().map(|v| C64::new(v, 0.0)).collect(),
            comment: String::new(),
        })
    }

    /// A dense real matrix (array format, general).
    pub fn from_dense(a: &Array2<f64>) -> Self {
        let (nrows, ncols) = a.dim();
        let (mut rows, mut cols, mut values) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..ncols {
            for i in 0..nrows {
                rows.push(i);
                cols.push(j);
                values.push(C64::new(a[(i, j)], 0.0));
            }
        }
        MatrixMarket { nrows, ncols, format: MmFormat::Array, field: MmField::Real, symmetry: MmSymmetry::General, rows, cols, values, comment: String::new() }
    }

    /// The same matrix with a different field or symmetry declared for writing (the entries
    /// must be consistent with it: integers for `Integer`, and for a symmetry the full matrix,
    /// of which the lower triangle is written).
    pub fn with_field(mut self, field: MmField) -> Self {
        self.field = field;
        self
    }

    pub fn with_symmetry(mut self, symmetry: MmSymmetry) -> Self {
        self.symmetry = symmetry;
        self
    }

    pub fn with_comment(mut self, comment: &str) -> Self {
        self.comment = comment.to_string();
        self
    }

    /// The matrix as a dense real array (real parts; duplicates summed, as SciPy's conversion
    /// of a COO matrix does).
    pub fn to_dense(&self) -> Array2<f64> {
        let mut a = Array2::zeros((self.nrows, self.ncols));
        for k in 0..self.values.len() {
            a[(self.rows[k], self.cols[k])] += self.values[k].re;
        }
        a
    }

    /// The matrix as a dense complex array.
    pub fn to_dense_complex(&self) -> Array2<C64> {
        let mut a = Array2::from_elem((self.nrows, self.ncols), C64::new(0.0, 0.0));
        for k in 0..self.values.len() {
            a[(self.rows[k], self.cols[k])] += self.values[k];
        }
        a
    }
}

/// The header information of a Matrix Market file (SciPy's `mminfo`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct MmInfo {
    pub nrows: usize,
    pub ncols: usize,
    /// Stored entries (for symmetric files, before expansion).
    pub entries: usize,
    pub format: MmFormat,
    pub field: MmField,
    pub symmetry: MmSymmetry,
}

fn bad(msg: &'static str) -> ErrorsJSL {
    ErrorsJSL::InvalidInputRange(msg)
}

fn parse_banner(line: &str) -> Result<(MmFormat, MmField, MmSymmetry), ErrorsJSL> {
    let words: Vec<String> = line.split_whitespace().map(|w| w.to_ascii_lowercase()).collect();
    if words.len() != 5 || words[0] != "%%matrixmarket" || words[1] != "matrix" {
        return Err(bad("Matrix Market: expected the banner %%MatrixMarket matrix <format> <field> <symmetry>"));
    }
    let format = match words[2].as_str() {
        "coordinate" => MmFormat::Coordinate,
        "array" => MmFormat::Array,
        _ => return Err(bad("Matrix Market: unknown format")),
    };
    let field = match words[3].as_str() {
        "real" | "double" => MmField::Real,
        "integer" => MmField::Integer,
        "complex" => MmField::Complex,
        "pattern" => MmField::Pattern,
        _ => return Err(bad("Matrix Market: unknown field")),
    };
    let symmetry = match words[4].as_str() {
        "general" => MmSymmetry::General,
        "symmetric" => MmSymmetry::Symmetric,
        "skew-symmetric" => MmSymmetry::SkewSymmetric,
        "hermitian" => MmSymmetry::Hermitian,
        _ => return Err(bad("Matrix Market: unknown symmetry")),
    };
    if format == MmFormat::Array && field == MmField::Pattern {
        return Err(bad("Matrix Market: the array format has no pattern field"));
    }
    Ok((format, field, symmetry))
}

/// Splits the text into banner, comments, size line and the remaining tokens.
/// The banner, comments, size line and the remaining tokens of a file.
type Parts<'a> = ((MmFormat, MmField, MmSymmetry), String, Vec<usize>, std::str::SplitWhitespace<'a>);

fn split(text: &str) -> Result<Parts<'_>, ErrorsJSL> {
    let mut rest = text;
    let mut take_line = || -> Option<&str> {
        if rest.is_empty() {
            return None;
        }
        let (line, tail) = match rest.find('\n') {
            Some(p) => (&rest[..p], &rest[p + 1..]),
            None => (rest, ""),
        };
        rest = tail;
        Some(line.trim_end_matches('\r'))
    };
    let banner = parse_banner(take_line().ok_or(bad("Matrix Market: empty file"))?)?;
    let mut comment = String::new();
    let size_line = loop {
        let line = take_line().ok_or(bad("Matrix Market: missing size line"))?;
        let t = line.trim();
        if let Some(c) = t.strip_prefix('%') {
            if !comment.is_empty() {
                comment.push('\n');
            }
            comment.push_str(c);
        } else if !t.is_empty() {
            break t.to_string();
        }
    };
    let size: Vec<usize> = size_line.split_whitespace().map(|w| w.parse().map_err(|_| bad("Matrix Market: bad size line"))).collect::<Result<_, _>>()?;
    let expected = if banner.0 == MmFormat::Coordinate { 3 } else { 2 };
    if size.len() != expected {
        return Err(bad("Matrix Market: wrong number of values on the size line"));
    }
    Ok((banner, comment, size, rest.split_whitespace()))
}

/// The header of a Matrix Market file (SciPy's `mminfo`).
pub fn mm_info(text: &str) -> Result<MmInfo, ErrorsJSL> {
    let ((format, field, symmetry), _, size, _) = split(text)?;
    let entries = if format == MmFormat::Coordinate {
        size[2]
    } else {
        let (m, n) = (size[0], size[1]);
        match symmetry {
            MmSymmetry::General => m * n,
            MmSymmetry::SkewSymmetric => n * (n.saturating_sub(1)) / 2,
            _ => n * (n + 1) / 2,
        }
    };
    Ok(MmInfo { nrows: size[0], ncols: size[1], entries, format, field, symmetry })
}

// ANCHOR: mm_decode
/// Parses a Matrix Market file (SciPy's `mmread`). Symmetric, skew-symmetric and Hermitian
/// matrices are expanded to the full matrix: each stored off-diagonal entry `(i, j, v)` adds
/// `(j, i, v)`, `(j, i, −v)` or `(j, i, v̄)`.
pub fn mm_decode(text: &str) -> Result<MatrixMarket, ErrorsJSL> {
    let ((format, field, symmetry), comment, size, mut tokens) = split(text)?;
    let (nrows, ncols) = (size[0], size[1]);
    if symmetry != MmSymmetry::General && nrows != ncols {
        return Err(bad("Matrix Market: a symmetric matrix must be square"));
    }
    let mut next_f64 = |what: &'static str| -> Result<f64, ErrorsJSL> {
        let t = tokens.next().ok_or(bad(what))?;
        t.parse::<f64>().or_else(|_| t.replace(['d', 'D'], "e").parse::<f64>()).map_err(|_| bad("Matrix Market: a value is not a number"))
    };
    let value = |next: &mut dyn FnMut(&'static str) -> Result<f64, ErrorsJSL>| -> Result<C64, ErrorsJSL> {
        Ok(match field {
            MmField::Pattern => C64::new(1.0, 0.0),
            MmField::Complex => {
                let re = next("Matrix Market: missing value")?;
                C64::new(re, next("Matrix Market: missing imaginary part")?)
            }
            _ => C64::new(next("Matrix Market: missing value")?, 0.0),
        })
    };
    let (mut rows, mut cols, mut values) = (Vec::new(), Vec::new(), Vec::new());
    let mut push = |i: usize, j: usize, v: C64| {
        rows.push(i);
        cols.push(j);
        values.push(v);
        if i != j {
            let mirrored = match symmetry {
                MmSymmetry::General => return,
                MmSymmetry::Symmetric => v,
                MmSymmetry::SkewSymmetric => -v,
                MmSymmetry::Hermitian => v.conj(),
            };
            rows.push(j);
            cols.push(i);
            values.push(mirrored);
        }
    };
    match format {
        MmFormat::Coordinate => {
            for _ in 0..size[2] {
                let i = next_f64("Matrix Market: missing row index")?;
                let j = next_f64("Matrix Market: missing column index")?;
                if !(i >= 1.0 && j >= 1.0 && i <= nrows as f64 && j <= ncols as f64 && i.fract() == 0.0 && j.fract() == 0.0) {
                    return Err(bad("Matrix Market: an index is out of range"));
                }
                let v = value(&mut next_f64)?;
                push(i as usize - 1, j as usize - 1, v);
            }
        }
        MmFormat::Array => {
            for j in 0..ncols {
                let start = match symmetry {
                    MmSymmetry::General => 0,
                    MmSymmetry::SkewSymmetric => j + 1,
                    _ => j,
                };
                for i in start..nrows {
                    let v = value(&mut next_f64)?;
                    push(i, j, v);
                }
                if symmetry == MmSymmetry::SkewSymmetric {
                    push(j, j, C64::new(0.0, 0.0));
                }
            }
        }
    }
    Ok(MatrixMarket { nrows, ncols, format, field, symmetry, rows, cols, values, comment })
}
// ANCHOR_END: mm_decode

fn fmt_real(v: f64) -> String {
    // The shortest representation that reads back exactly, in exponent form like SciPy's.
    if v == 0.0 { "0.0000000000000000e+00".to_string() } else { format!("{v:e}") }
}

/// Writes a Matrix Market file (SciPy's `mmwrite`): the coordinate or array format as declared,
/// with only the lower triangle for a symmetric matrix (the strictly lower one for a
/// skew-symmetric one). Values are written in the shortest form that reads back exactly.
pub fn mm_encode(m: &MatrixMarket) -> Result<String, ErrorsJSL> {
    let sym = |s: MmSymmetry| match s {
        MmSymmetry::General => "general",
        MmSymmetry::Symmetric => "symmetric",
        MmSymmetry::SkewSymmetric => "skew-symmetric",
        MmSymmetry::Hermitian => "hermitian",
    };
    let fld = match m.field {
        MmField::Real => "real",
        MmField::Integer => "integer",
        MmField::Complex => "complex",
        MmField::Pattern => "pattern",
    };
    if m.format == MmFormat::Array && m.field == MmField::Pattern {
        return Err(bad("Matrix Market: the array format has no pattern field"));
    }
    if m.symmetry != MmSymmetry::General && m.nrows != m.ncols {
        return Err(bad("Matrix Market: a symmetric matrix must be square"));
    }
    let fmt = match m.format {
        MmFormat::Coordinate => "coordinate",
        MmFormat::Array => "array",
    };
    let mut out = format!("%%MatrixMarket matrix {fmt} {fld} {}\n", sym(m.symmetry));
    for line in m.comment.lines() {
        out.push('%');
        out.push_str(line);
        out.push('\n');
    }
    let val = |v: C64| -> String {
        match m.field {
            MmField::Pattern => String::new(),
            MmField::Integer => format!(" {}", v.re as i64),
            MmField::Complex => format!(" {} {}", fmt_real(v.re), fmt_real(v.im)),
            MmField::Real => format!(" {}", fmt_real(v.re)),
        }
    };
    let keep = |i: usize, j: usize| match m.symmetry {
        MmSymmetry::General => true,
        MmSymmetry::SkewSymmetric => i > j,
        _ => i >= j,
    };
    match m.format {
        MmFormat::Coordinate => {
            let idx: Vec<usize> = (0..m.values.len()).filter(|&k| keep(m.rows[k], m.cols[k])).collect();
            out.push_str(&format!("{} {} {}\n", m.nrows, m.ncols, idx.len()));
            for k in idx {
                out.push_str(&format!("{} {}{}\n", m.rows[k] + 1, m.cols[k] + 1, val(m.values[k])));
            }
        }
        MmFormat::Array => {
            let a = m.to_dense_complex();
            out.push_str(&format!("{} {}\n", m.nrows, m.ncols));
            for j in 0..m.ncols {
                for i in 0..m.nrows {
                    if keep(i, j) {
                        out.push_str(val(a[(i, j)]).trim_start());
                        out.push('\n');
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Reads a Matrix Market file.
pub fn read_mm_file(path: impl AsRef<Path>) -> Result<MatrixMarket, ErrorsJSL> {
    let text = std::fs::read_to_string(path).map_err(|_| ErrorsJSL::RuntimeError("could not read the Matrix Market file"))?;
    mm_decode(&text)
}

/// Writes a Matrix Market file.
pub fn write_mm_file(path: impl AsRef<Path>, m: &MatrixMarket) -> Result<(), ErrorsJSL> {
    std::fs::write(path, mm_encode(m)?).map_err(|_| ErrorsJSL::RuntimeError("could not write the Matrix Market file"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_symmetry_expansion() {
        let text = "%%MatrixMarket matrix coordinate real symmetric\n% a comment\n3 3 4\n1 1 2.0\n2 1 -1\n3 2 -1.5e0\n3 3 4\n";
        let m = mm_decode(text).unwrap();
        assert_eq!(m.comment, " a comment");
        let a = m.to_dense();
        assert_eq!(a[(0, 1)], -1.0);
        assert_eq!(a[(1, 0)], -1.0);
        assert_eq!(a[(1, 2)], -1.5);
        assert_eq!(mm_info(text).unwrap().entries, 4);
        let back = mm_decode(&mm_encode(&m).unwrap()).unwrap();
        assert_eq!(back.to_dense(), a);
        // Dense, skew-symmetric array.
        let s = "%%MatrixMarket matrix array real skew-symmetric\n3 3\n1\n2\n3\n";
        let d = mm_decode(s).unwrap().to_dense();
        assert_eq!(d[(1, 0)], 1.0);
        assert_eq!(d[(0, 1)], -1.0);
        assert_eq!(d[(2, 1)], 3.0);
        // Complex Hermitian and pattern.
        let h = mm_decode("%%MatrixMarket matrix coordinate complex hermitian\n2 2 2\n1 1 1 0\n2 1 0.5 -2\n").unwrap().to_dense_complex();
        assert_eq!(h[(0, 1)], C64::new(0.5, 2.0));
        let p = mm_decode("%%MatrixMarket matrix coordinate pattern general\n2 3 2\n1 3\n2 1\n").unwrap();
        assert_eq!(p.to_dense()[(0, 2)], 1.0);
        // Exact round trip of awkward values in array format.
        let x = Array2::from_shape_vec((2, 2), vec![0.1, -1e-300, 1.0 / 3.0, 6.02e23]).unwrap();
        assert_eq!(mm_decode(&mm_encode(&MatrixMarket::from_dense(&x)).unwrap()).unwrap().to_dense(), x);
    }
}
