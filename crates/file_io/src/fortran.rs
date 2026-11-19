//! Fortran sequential unformatted files (SciPy's `io.FortranFile`).
//!
//! Each record written by a Fortran `WRITE` statement to an unformatted sequential file is
//! framed by its length in bytes, as a 4-byte (gfortran's default, SciPy's `header_dtype`
//! `uint32`) or 8-byte integer, before and after the data. The data are raw machine values,
//! here little-endian.
//!
//! See the book entry [Scientific data files](https://jwkunz.github.io/lib_jsl/book/foundations/scientific-files.html#fortran-records-and-arff).

use lib_jsl_core::ErrorsJSL;
use std::path::Path;

fn bad(msg: &'static str) -> ErrorsJSL {
    ErrorsJSL::InvalidInputRange(msg)
}

// ANCHOR: fortran
/// Splits a Fortran unformatted sequential file into its records (SciPy's
/// `FortranFile.read_record`), with `marker_bytes` (4 or 8) per length marker. Each record's
/// leading and trailing markers must agree.
pub fn fortran_records_decode(bytes: &[u8], marker_bytes: usize) -> Result<Vec<Vec<u8>>, ErrorsJSL> {
    if !matches!(marker_bytes, 4 | 8) {
        return Err(bad("Fortran file: record markers are 4 or 8 bytes"));
    }
    let marker = |at: usize| -> Result<usize, ErrorsJSL> {
        let s = bytes.get(at..at + marker_bytes).ok_or(bad("Fortran file: truncated record marker"))?;
        Ok(if marker_bytes == 4 { u32::from_le_bytes(s.try_into().expect("4 bytes")) as usize } else { u64::from_le_bytes(s.try_into().expect("8 bytes")) as usize })
    };
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < bytes.len() {
        let n = marker(pos)?;
        let start = pos + marker_bytes;
        let data = bytes.get(start..start + n).ok_or(bad("Fortran file: truncated record"))?;
        if marker(start + n)? != n {
            return Err(bad("Fortran file: the record markers disagree"));
        }
        out.push(data.to_vec());
        pos = start + n + marker_bytes;
    }
    Ok(out)
}

/// Frames records for a Fortran unformatted sequential file (SciPy's `FortranFile.write_record`).
pub fn fortran_records_encode(records: &[Vec<u8>], marker_bytes: usize) -> Result<Vec<u8>, ErrorsJSL> {
    if !matches!(marker_bytes, 4 | 8) {
        return Err(bad("Fortran file: record markers are 4 or 8 bytes"));
    }
    let mut out = Vec::new();
    for r in records {
        let m: Vec<u8> = if marker_bytes == 4 {
            u32::try_from(r.len()).map_err(|_| bad("Fortran file: a record is too long for 4-byte markers"))?.to_le_bytes().to_vec()
        } else {
            (r.len() as u64).to_le_bytes().to_vec()
        };
        out.extend(&m);
        out.extend(r);
        out.extend(&m);
    }
    Ok(out)
}
// ANCHOR_END: fortran

/// A record's bytes as little-endian `f64` values (`read_reals(dtype=float)`).
pub fn record_to_f64(r: &[u8]) -> Result<Vec<f64>, ErrorsJSL> {
    if !r.len().is_multiple_of(8) {
        return Err(bad("Fortran file: the record is not a whole number of f64 values"));
    }
    Ok(r.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes"))).collect())
}

/// A record's bytes as little-endian `f32` values.
pub fn record_to_f32(r: &[u8]) -> Result<Vec<f32>, ErrorsJSL> {
    if !r.len().is_multiple_of(4) {
        return Err(bad("Fortran file: the record is not a whole number of f32 values"));
    }
    Ok(r.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes"))).collect())
}

/// A record's bytes as little-endian `i32` values (`read_ints()`).
pub fn record_to_i32(r: &[u8]) -> Result<Vec<i32>, ErrorsJSL> {
    if !r.len().is_multiple_of(4) {
        return Err(bad("Fortran file: the record is not a whole number of i32 values"));
    }
    Ok(r.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().expect("4 bytes"))).collect())
}

/// `f64` values as a record's bytes.
pub fn record_from_f64(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// `f32` values as a record's bytes.
pub fn record_from_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// `i32` values as a record's bytes.
pub fn record_from_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Reads the records of a Fortran unformatted sequential file.
pub fn read_fortran_file(path: impl AsRef<Path>, marker_bytes: usize) -> Result<Vec<Vec<u8>>, ErrorsJSL> {
    fortran_records_decode(&std::fs::read(path).map_err(|_| ErrorsJSL::RuntimeError("could not read the Fortran file"))?, marker_bytes)
}

/// Writes records as a Fortran unformatted sequential file.
pub fn write_fortran_file(path: impl AsRef<Path>, records: &[Vec<u8>], marker_bytes: usize) -> Result<(), ErrorsJSL> {
    std::fs::write(path, fortran_records_encode(records, marker_bytes)?).map_err(|_| ErrorsJSL::RuntimeError("could not write the Fortran file"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_round_trip() {
        let recs = vec![record_from_i32(&[3, -4]), record_from_f64(&[1.5, -2.25, 1e300]), Vec::new(), record_from_f32(&[0.5])];
        for m in [4, 8] {
            let b = fortran_records_encode(&recs, m).unwrap();
            assert_eq!(b.len(), recs.iter().map(|r| r.len() + 2 * m).sum::<usize>());
            let back = fortran_records_decode(&b, m).unwrap();
            assert_eq!(back, recs);
            assert_eq!(record_to_f64(&back[1]).unwrap(), vec![1.5, -2.25, 1e300]);
            assert_eq!(record_to_i32(&back[0]).unwrap(), vec![3, -4]);
        }
        let mut b = fortran_records_encode(&recs, 4).unwrap();
        b[12] = 99; // the trailing marker of the first record
        assert!(fortran_records_decode(&b, 4).is_err());
    }
}
