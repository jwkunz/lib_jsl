//! Harwell-Boeing sparse matrix files (SciPy's `io.hb_read` and `io.hb_write`).
//!
//! A Harwell-Boeing file is fixed-width Fortran text: a title and key, the numbers of lines in
//! each section, the matrix type (`RUA`: real, unsymmetric, assembled), the dimensions and
//! number of entries, the Fortran formats of the sections, and then the 1-based column pointers,
//! row indices and values of the matrix in compressed sparse column form.
//!
//! Reading supports real and pattern matrices, unsymmetric, rectangular, symmetric and
//! skew-symmetric (expanded to the full matrix, as for Matrix Market), with any integer format
//! and `E`, `D` or `F` real formats; writing produces `RUA` files laid out as SciPy's.
//! Complex and elemental matrices and right-hand sides are not supported (nor by SciPy).
//!
//! See the book entry [Scientific data files](https://jwkunz.github.io/lib_jsl/book/foundations/scientific-files.html#harwell-boeing).

use lib_jsl_core::ErrorsJSL;
use std::path::Path;

/// A sparse matrix in compressed sparse column form, as read from or written to a
/// Harwell-Boeing file (0-based).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct HarwellBoeing {
    pub title: String,
    pub key: String,
    pub nrows: usize,
    pub ncols: usize,
    /// `ncols + 1` column pointers.
    pub col_ptr: Vec<usize>,
    pub row_indices: Vec<usize>,
    /// The values (1 for a pattern matrix).
    pub values: Vec<f64>,
}

impl HarwellBoeing {
    /// A matrix from compressed sparse column arrays, with SciPy's default title and key.
    pub fn new(nrows: usize, ncols: usize, col_ptr: Vec<usize>, row_indices: Vec<usize>, values: Vec<f64>) -> Result<Self, ErrorsJSL> {
        if col_ptr.len() != ncols + 1 || row_indices.len() != values.len() || col_ptr[ncols] != values.len() || col_ptr.windows(2).any(|w| w[1] < w[0]) || row_indices.iter().any(|&r| r >= nrows) {
            return Err(ErrorsJSL::InvalidInputRange("HarwellBoeing: inconsistent compressed sparse column arrays"));
        }
        Ok(HarwellBoeing { title: "Default title".into(), key: "0".into(), nrows, ncols, col_ptr, row_indices, values })
    }

    /// The matrix as dense rows.
    #[allow(clippy::needless_range_loop)]
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut a = vec![vec![0.0; self.ncols]; self.nrows];
        for j in 0..self.ncols {
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                a[self.row_indices[k]][j] += self.values[k];
            }
        }
        a
    }
}

fn bad(msg: &'static str) -> ErrorsJSL {
    ErrorsJSL::InvalidInputRange(msg)
}

/// A Fortran edit descriptor `(r Xw.d)`, with an optional scale factor `kP` ignored.
#[derive(Clone, Copy, Debug)]
struct FortranFormat {
    repeat: usize,
    width: usize,
}

fn parse_format(f: &str) -> Result<FortranFormat, ErrorsJSL> {
    let s: String = f.trim().trim_start_matches('(').trim_end_matches(')').to_ascii_uppercase();
    // Drop a scale factor such as "1P," or "1P".
    let s = match s.find('P') {
        Some(p) if s[..p].chars().all(|c| c.is_ascii_digit() || c == '-') => s[p + 1..].trim_start_matches(',').to_string(),
        _ => s,
    };
    let letter = s.find(|c: char| c.is_ascii_alphabetic()).ok_or(bad("Harwell-Boeing: bad format"))?;
    let repeat = if letter == 0 { 1 } else { s[..letter].parse().map_err(|_| bad("Harwell-Boeing: bad format"))? };
    let rest = &s[letter + 1..];
    let width_str = rest.split('.').next().unwrap_or("");
    let width = width_str.parse().map_err(|_| bad("Harwell-Boeing: bad format width"))?;
    if !matches!(&s[letter..letter + 1], "I" | "E" | "D" | "F" | "G") {
        return Err(bad("Harwell-Boeing: unsupported format"));
    }
    Ok(FortranFormat { repeat, width })
}

/// Reads `count` fixed-width fields from the given lines.
fn read_fields<'a>(lines: &mut impl Iterator<Item = &'a str>, nlines: usize, fmt: FortranFormat, count: usize) -> Result<Vec<String>, ErrorsJSL> {
    let mut out = Vec::with_capacity(count);
    for _ in 0..nlines {
        let line = lines.next().ok_or(bad("Harwell-Boeing: truncated"))?;
        let chars: Vec<char> = line.chars().collect();
        for k in 0..fmt.repeat {
            if out.len() == count {
                break;
            }
            let (a, b) = (k * fmt.width, ((k + 1) * fmt.width).min(chars.len()));
            if a >= chars.len() {
                break;
            }
            let field: String = chars[a..b].iter().collect();
            if !field.trim().is_empty() {
                out.push(field.trim().to_string());
            }
        }
    }
    if out.len() != count {
        return Err(bad("Harwell-Boeing: wrong number of entries in a section"));
    }
    Ok(out)
}

fn int_at(line: &str, from: usize, width: usize) -> Result<usize, ErrorsJSL> {
    let s: String = line.chars().skip(from).take(width).collect();
    let s = s.trim();
    if s.is_empty() {
        return Ok(0);
    }
    s.parse().map_err(|_| bad("Harwell-Boeing: bad integer in the header"))
}

// ANCHOR: hb_read
/// Parses a Harwell-Boeing file (SciPy's `hb_read`). Symmetric and skew-symmetric matrices,
/// which store one triangle, are expanded to the full matrix.
pub fn hb_decode(text: &str) -> Result<HarwellBoeing, ErrorsJSL> {
    let mut lines = text.lines();
    let l1 = lines.next().ok_or(bad("Harwell-Boeing: empty file"))?;
    let title: String = l1.chars().take(72).collect::<String>().trim_end().to_string();
    let key: String = l1.chars().skip(72).take(8).collect::<String>().trim().to_string();
    let l2 = lines.next().ok_or(bad("Harwell-Boeing: missing line 2"))?;
    let (ptrcrd, indcrd, valcrd, rhscrd) = (int_at(l2, 14, 14)?, int_at(l2, 28, 14)?, int_at(l2, 42, 14)?, int_at(l2, 56, 14)?);
    let l3 = lines.next().ok_or(bad("Harwell-Boeing: missing line 3"))?;
    let mxtype: String = l3.chars().take(3).collect::<String>().to_ascii_uppercase();
    let (nrows, ncols, nnz) = (int_at(l3, 14, 14)?, int_at(l3, 28, 14)?, int_at(l3, 42, 14)?);
    let l4 = lines.next().ok_or(bad("Harwell-Boeing: missing line 4"))?;
    let col = |a: usize, w: usize| -> String { l4.chars().skip(a).take(w).collect() };
    let (ptrfmt, indfmt, valfmt) = (parse_format(&col(0, 16))?, parse_format(&col(16, 16))?, col(32, 20));
    if rhscrd > 0 {
        lines.next();
    }
    let mt: Vec<char> = mxtype.chars().collect();
    if mt.len() != 3 || mt[2] != 'A' {
        return Err(bad("Harwell-Boeing: only assembled matrices are supported"));
    }
    if !matches!(mt[0], 'R' | 'P') {
        return Err(bad("Harwell-Boeing: complex matrices are not supported"));
    }
    let ptr: Vec<usize> = read_fields(&mut lines, ptrcrd, ptrfmt, ncols + 1)?.iter().map(|s| s.parse::<usize>().map_err(|_| bad("Harwell-Boeing: bad pointer"))).collect::<Result<_, _>>()?;
    let ind: Vec<usize> = read_fields(&mut lines, indcrd, indfmt, nnz)?.iter().map(|s| s.parse::<usize>().map_err(|_| bad("Harwell-Boeing: bad index"))).collect::<Result<_, _>>()?;
    let vals: Vec<f64> = if mt[0] == 'P' {
        vec![1.0; nnz]
    } else {
        read_fields(&mut lines, valcrd, parse_format(&valfmt)?, nnz)?
            .iter()
            .map(|s| {
                let s = s.replace(['D', 'd'], "E");
                // Fortran allows the 'E' to be omitted before a signed exponent: 1.5-03.
                s.parse::<f64>().or_else(|_| {
                    let p = s[1..].rfind(['+', '-']).map(|p| p + 1).ok_or(bad("Harwell-Boeing: bad value"))?;
                    format!("{}E{}", &s[..p], &s[p..]).parse::<f64>().map_err(|_| bad("Harwell-Boeing: bad value"))
                })
            })
            .collect::<Result<_, _>>()?
    };
    if ptr.first() != Some(&1) || ptr.windows(2).any(|w| w[1] < w[0]) || *ptr.last().expect("non-empty") != nnz + 1 || ind.iter().any(|&r| r == 0 || r > nrows) {
        return Err(bad("Harwell-Boeing: inconsistent pointers or indices"));
    }
    let col_ptr: Vec<usize> = ptr.iter().map(|p| p - 1).collect();
    let row_indices: Vec<usize> = ind.iter().map(|r| r - 1).collect();
    let mut hb = HarwellBoeing { title, key, nrows, ncols, col_ptr, row_indices, values: vals };
    if matches!(mt[1], 'S' | 'Z' | 'H') {
        // Expand the stored triangle into full compressed columns.
        let sign = if mt[1] == 'Z' { -1.0 } else { 1.0 };
        let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); ncols];
        for j in 0..ncols {
            for k in hb.col_ptr[j]..hb.col_ptr[j + 1] {
                let (i, v) = (hb.row_indices[k], hb.values[k]);
                cols[j].push((i, v));
                if i != j {
                    cols[i].push((j, sign * v));
                }
            }
        }
        let mut cp = vec![0];
        let (mut ri, mut vv) = (Vec::new(), Vec::new());
        for c in cols.iter_mut() {
            c.sort_by_key(|e| e.0);
            for &(i, v) in c.iter() {
                ri.push(i);
                vv.push(v);
            }
            cp.push(ri.len());
        }
        hb.col_ptr = cp;
        hb.row_indices = ri;
        hb.values = vv;
    }
    Ok(hb)
}
// ANCHOR_END: hb_read

/// `v` as Fortran writes it with `E25.16` under Python formatting: one digit before the point,
/// sixteen after, and a signed exponent of at least two digits.
fn fmt_e(v: f64) -> String {
    let s = format!("{v:.16E}");
    let (m, e) = s.split_once('E').expect("exponent form");
    let e: i32 = e.parse().expect("integer exponent");
    format!("{m}E{}{:02}", if e < 0 { '-' } else { '+' }, e.abs())
}

/// Writes a Harwell-Boeing file (SciPy's `hb_write`), as a real unsymmetric assembled (`RUA`)
/// matrix with SciPy's choice of formats: integers as wide as the largest needs plus one, as
/// many per 80-column line as fit, and values in `(3E25.16)`.
pub fn hb_encode(hb: &HarwellBoeing) -> Result<String, ErrorsJSL> {
    let nnz = hb.values.len();
    if hb.col_ptr.len() != hb.ncols + 1 || hb.row_indices.len() != nnz || hb.col_ptr[hb.ncols] != nnz {
        return Err(bad("HarwellBoeing: inconsistent compressed sparse column arrays"));
    }
    let int_format = |max: usize| -> (usize, usize) {
        let w = max.to_string().len() + 1;
        (80 / w, w)
    };
    let ptr: Vec<usize> = hb.col_ptr.iter().map(|p| p + 1).collect();
    let ind: Vec<usize> = hb.row_indices.iter().map(|r| r + 1).collect();
    let (pn, pw) = int_format(*ptr.iter().max().unwrap_or(&1));
    let (inn, iw) = int_format(*ind.iter().max().unwrap_or(&1));
    let (vn, vw) = (3, 25);
    let lines_for = |count: usize, per: usize| count.div_ceil(per);
    let (ptrcrd, indcrd, valcrd) = (lines_for(ptr.len(), pn), lines_for(ind.len(), inn), lines_for(nnz, vn));
    let mut out = String::new();
    out.push_str(&format!("{:<72}{:<8}\n", hb.title.chars().take(72).collect::<String>(), hb.key.chars().take(8).collect::<String>()));
    out.push_str(&format!("{:>14}{:>14}{:>14}{:>14}\n", ptrcrd + indcrd + valcrd, ptrcrd, indcrd, valcrd));
    out.push_str(&format!("{:<14}{:>14}{:>14}{:>14}{:>14}\n", "RUA", hb.nrows, hb.ncols, nnz, 0));
    out.push_str(&format!("{:<16}{:<16}{:<20}{:<20}\n", format!("({pn}I{pw})"), format!("({inn}I{iw})"), format!("({vn}E{vw}.16)"), ""));
    for chunk in ptr.chunks(pn) {
        out.push_str(&chunk.iter().map(|p| format!("{p:>pw$}")).collect::<String>());
        out.push('\n');
    }
    for chunk in ind.chunks(inn) {
        out.push_str(&chunk.iter().map(|p| format!("{p:>iw$}")).collect::<String>());
        out.push('\n');
    }
    for chunk in hb.values.chunks(vn) {
        out.push_str(&chunk.iter().map(|&v| format!("{:>vw$}", fmt_e(v))).collect::<String>());
        out.push('\n');
    }
    Ok(out)
}

/// Reads a Harwell-Boeing file.
pub fn read_hb_file(path: impl AsRef<Path>) -> Result<HarwellBoeing, ErrorsJSL> {
    hb_decode(&std::fs::read_to_string(path).map_err(|_| ErrorsJSL::RuntimeError("could not read the Harwell-Boeing file"))?)
}

/// Writes a Harwell-Boeing file.
pub fn write_hb_file(path: impl AsRef<Path>, hb: &HarwellBoeing) -> Result<(), ErrorsJSL> {
    std::fs::write(path, hb_encode(hb)?).map_err(|_| ErrorsJSL::RuntimeError("could not write the Harwell-Boeing file"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_symmetric() {
        let hb = HarwellBoeing::new(4, 3, vec![0, 2, 3, 5], vec![0, 3, 1, 0, 2], vec![1.5, -2.0, 1e-300, 6.02e23, 1.0 / 3.0]).unwrap();
        let text = hb_encode(&hb).unwrap();
        assert_eq!(hb_decode(&text).unwrap(), hb);
        assert!(text.lines().nth(2).unwrap().starts_with("RUA"));
        // A symmetric file storing the lower triangle, with D exponents.
        let s = "symmetric test                                                          KEY     \n\
                 \x20            3             1             1             1\n\
                 RSA                        3             3             4             0\n\
                 (4I3)           (4I3)           (4D12.4)            \n\
                 \x20 1  3  4  5\n\
                 \x20 1  2  2  3\n\
                 \x20 2.0000D+00 -1.0000D+00  3.0000D+00  4.0000D+00\n";
        let m = hb_decode(s).unwrap();
        assert_eq!(m.key, "KEY");
        assert_eq!(m.to_dense(), vec![vec![2.0, -1.0, 0.0], vec![-1.0, 3.0, 0.0], vec![0.0, 0.0, 4.0]]);
    }
}
