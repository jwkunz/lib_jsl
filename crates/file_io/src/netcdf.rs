//! NetCDF-3 files, classic and 64-bit offset (SciPy's `io.netcdf_file`).
//!
//! A NetCDF-3 file is big-endian: the magic `CDF` and a version byte (1 classic, 2 with 64-bit
//! offsets), the number of records, then lists of dimensions (one may be unlimited, the record
//! dimension), global attributes and variables (name, dimensions, attributes, type, size and the
//! offset of the data). Fixed-size variables are stored one after another; variables along the
//! record dimension are interleaved record by record. Types are bytes, characters, 16- and
//! 32-bit integers and 32- and 64-bit floats.
//!
//! See the book entry [Scientific data files](https://jwkunz.github.io/lib_jsl/book/foundations/scientific-files.html#netcdf-3).

use lib_jsl_core::ErrorsJSL;
use std::path::Path;

/// Values of one NetCDF type, in C order.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum NcValue {
    Byte(Vec<i8>),
    Char(String),
    Short(Vec<i16>),
    Int(Vec<i32>),
    Float(Vec<f32>),
    Double(Vec<f64>),
}

impl NcValue {
    fn type_code(&self) -> u32 {
        match self {
            NcValue::Byte(_) => 1,
            NcValue::Char(_) => 2,
            NcValue::Short(_) => 3,
            NcValue::Int(_) => 4,
            NcValue::Float(_) => 5,
            NcValue::Double(_) => 6,
        }
    }

    /// The number of elements (bytes, for characters).
    pub fn len(&self) -> usize {
        match self {
            NcValue::Byte(v) => v.len(),
            NcValue::Char(s) => s.len(),
            NcValue::Short(v) => v.len(),
            NcValue::Int(v) => v.len(),
            NcValue::Float(v) => v.len(),
            NcValue::Double(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The values as `f64` (characters as their byte values).
    pub fn to_f64(&self) -> Vec<f64> {
        match self {
            NcValue::Byte(v) => v.iter().map(|&x| x as f64).collect(),
            NcValue::Char(s) => s.bytes().map(|b| b as f64).collect(),
            NcValue::Short(v) => v.iter().map(|&x| x as f64).collect(),
            NcValue::Int(v) => v.iter().map(|&x| x as f64).collect(),
            NcValue::Float(v) => v.iter().map(|&x| x as f64).collect(),
            NcValue::Double(v) => v.clone(),
        }
    }

    fn encode(&self) -> Vec<u8> {
        match self {
            NcValue::Byte(v) => v.iter().map(|&x| x as u8).collect(),
            NcValue::Char(s) => s.as_bytes().to_vec(),
            NcValue::Short(v) => v.iter().flat_map(|x| x.to_be_bytes()).collect(),
            NcValue::Int(v) => v.iter().flat_map(|x| x.to_be_bytes()).collect(),
            NcValue::Float(v) => v.iter().flat_map(|x| x.to_be_bytes()).collect(),
            NcValue::Double(v) => v.iter().flat_map(|x| x.to_be_bytes()).collect(),
        }
    }

    fn element_size(code: u32) -> Option<usize> {
        Some(match code {
            1 | 2 => 1,
            3 => 2,
            4 | 5 => 4,
            6 => 8,
            _ => return None,
        })
    }

    fn decode(code: u32, b: &[u8]) -> Result<NcValue, ErrorsJSL> {
        Ok(match code {
            1 => NcValue::Byte(b.iter().map(|&x| x as i8).collect()),
            2 => NcValue::Char(String::from_utf8_lossy(b).into_owned()),
            3 => NcValue::Short(b.chunks_exact(2).map(|c| i16::from_be_bytes([c[0], c[1]])).collect()),
            4 => NcValue::Int(b.chunks_exact(4).map(|c| i32::from_be_bytes(c.try_into().expect("4 bytes"))).collect()),
            5 => NcValue::Float(b.chunks_exact(4).map(|c| f32::from_be_bytes(c.try_into().expect("4 bytes"))).collect()),
            6 => NcValue::Double(b.chunks_exact(8).map(|c| f64::from_be_bytes(c.try_into().expect("8 bytes"))).collect()),
            _ => return Err(bad("NetCDF: unknown type")),
        })
    }

    fn empty_like(code: u32) -> NcValue {
        NcValue::decode(code, &[]).expect("valid code")
    }

    fn append(&mut self, other: NcValue) {
        match (self, other) {
            (NcValue::Byte(a), NcValue::Byte(b)) => a.extend(b),
            (NcValue::Char(a), NcValue::Char(b)) => a.push_str(&b),
            (NcValue::Short(a), NcValue::Short(b)) => a.extend(b),
            (NcValue::Int(a), NcValue::Int(b)) => a.extend(b),
            (NcValue::Float(a), NcValue::Float(b)) => a.extend(b),
            (NcValue::Double(a), NcValue::Double(b)) => a.extend(b),
            _ => unreachable!("record slabs share their variable's type"),
        }
    }
}

/// A variable: its dimensions (indices into the file's dimension list), attributes and data in
/// C order (the record dimension, if any, first).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct NcVariable {
    pub name: String,
    pub dims: Vec<usize>,
    pub attributes: Vec<(String, NcValue)>,
    pub data: NcValue,
}

impl NcVariable {
    pub fn new(name: &str, dims: Vec<usize>, data: NcValue) -> Self {
        NcVariable { name: name.to_string(), dims, attributes: Vec::new(), data }
    }

    pub fn with_attribute(mut self, name: &str, value: NcValue) -> Self {
        self.attributes.push((name.to_string(), value));
        self
    }
}

/// The contents of a NetCDF-3 file.
#[derive(Clone, Debug, PartialEq, Default)]
#[non_exhaustive]
pub struct NetCdf {
    /// Dimension names and lengths; `None` for the unlimited (record) dimension.
    pub dimensions: Vec<(String, Option<usize>)>,
    pub attributes: Vec<(String, NcValue)>,
    pub variables: Vec<NcVariable>,
    /// The number of records along the unlimited dimension.
    pub numrecs: usize,
    /// 1 for the classic format, 2 for 64-bit offsets.
    pub version: u8,
}

impl NetCdf {
    pub fn new() -> Self {
        NetCdf { version: 1, ..Default::default() }
    }

    pub fn with_dimension(mut self, name: &str, length: Option<usize>) -> Self {
        self.dimensions.push((name.to_string(), length));
        self
    }

    pub fn with_attribute(mut self, name: &str, value: NcValue) -> Self {
        self.attributes.push((name.to_string(), value));
        self
    }

    /// Adds a variable; along the unlimited dimension the number of records follows its data.
    pub fn with_variable(mut self, v: NcVariable) -> Self {
        self.variables.push(v);
        self
    }

    /// The variable with this name.
    pub fn variable(&self, name: &str) -> Option<&NcVariable> {
        self.variables.iter().find(|v| v.name == name)
    }

    /// The lengths of a variable's dimensions (the record dimension as `numrecs`).
    pub fn shape(&self, v: &NcVariable) -> Vec<usize> {
        v.dims.iter().map(|&d| self.dimensions[d].1.unwrap_or(self.numrecs)).collect()
    }
}

fn bad(msg: &'static str) -> ErrorsJSL {
    ErrorsJSL::InvalidInputRange(msg)
}

struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], ErrorsJSL> {
        let s = self.b.get(self.pos..self.pos + n).ok_or(bad("NetCDF: truncated header"))?;
        self.pos += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, ErrorsJSL> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().expect("4 bytes")))
    }
    fn u64(&mut self) -> Result<u64, ErrorsJSL> {
        Ok(u64::from_be_bytes(self.take(8)?.try_into().expect("8 bytes")))
    }
    fn name(&mut self) -> Result<String, ErrorsJSL> {
        let n = self.u32()? as usize;
        let s = String::from_utf8_lossy(self.take(n)?).into_owned();
        self.take((4 - n % 4) % 4)?;
        Ok(s)
    }
    fn attributes(&mut self) -> Result<Vec<(String, NcValue)>, ErrorsJSL> {
        let tag = self.u32()?;
        let n = self.u32()? as usize;
        if tag == 0 {
            return Ok(Vec::new());
        }
        if tag != 0x0C {
            return Err(bad("NetCDF: expected an attribute list"));
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let name = self.name()?;
            let ty = self.u32()?;
            let count = self.u32()? as usize;
            let size = NcValue::element_size(ty).ok_or(bad("NetCDF: unknown attribute type"))? * count;
            let v = NcValue::decode(ty, self.take(size)?)?;
            self.take((4 - size % 4) % 4)?;
            out.push((name, v));
        }
        Ok(out)
    }
}

// ANCHOR: netcdf_decode
/// Parses a NetCDF-3 file (SciPy's `netcdf_file(..., "r")`), classic or 64-bit offset.
pub fn netcdf_decode(bytes: &[u8]) -> Result<NetCdf, ErrorsJSL> {
    let mut c = Cursor { b: bytes, pos: 0 };
    let magic = c.take(4)?;
    if &magic[..3] != b"CDF" || !matches!(magic[3], 1 | 2) {
        return Err(bad("NetCDF: not a NetCDF-3 file (NetCDF-4 files are HDF5)"));
    }
    let version = magic[3];
    let numrecs_raw = c.u32()?;
    let tag = c.u32()?;
    let ndims = c.u32()? as usize;
    if tag != 0 && tag != 0x0A {
        return Err(bad("NetCDF: expected a dimension list"));
    }
    let mut dimensions = Vec::with_capacity(ndims);
    for _ in 0..ndims {
        let name = c.name()?;
        let len = c.u32()? as usize;
        dimensions.push((name, if len == 0 { None } else { Some(len) }));
    }
    let attributes = c.attributes()?;
    let tag = c.u32()?;
    let nvars = c.u32()? as usize;
    if tag != 0 && tag != 0x0B {
        return Err(bad("NetCDF: expected a variable list"));
    }
    struct Raw {
        name: String,
        dims: Vec<usize>,
        attributes: Vec<(String, NcValue)>,
        ty: u32,
        vsize: usize,
        begin: usize,
    }
    let mut raws = Vec::with_capacity(nvars);
    for _ in 0..nvars {
        let name = c.name()?;
        let nd = c.u32()? as usize;
        let dims: Vec<usize> = (0..nd).map(|_| c.u32().map(|d| d as usize)).collect::<Result<_, _>>()?;
        if dims.iter().any(|&d| d >= dimensions.len()) {
            return Err(bad("NetCDF: a variable refers to an unknown dimension"));
        }
        let attributes = c.attributes()?;
        let ty = c.u32()?;
        let vsize = c.u32()? as usize;
        let begin = if version == 1 { c.u32()? as usize } else { c.u64()? as usize };
        raws.push(Raw { name, dims, attributes, ty, vsize, begin });
    }
    let is_record = |r: &Raw| r.dims.first().is_some_and(|&d| dimensions[d].1.is_none());
    let rec_vars: Vec<usize> = (0..raws.len()).filter(|&i| is_record(&raws[i])).collect();
    // The stride between records: the sum of the record variables' sizes, unpadded when there
    // is only one record variable.
    let recsize: usize = rec_vars.iter().map(|&i| raws[i].vsize).sum();
    let numrecs = if numrecs_raw == u32::MAX {
        // Streaming: infer the count from the file length.
        match rec_vars.first() {
            Some(&i) if recsize > 0 => (bytes.len().saturating_sub(raws[i].begin)) / recsize,
            _ => 0,
        }
    } else {
        numrecs_raw as usize
    };
    let mut variables = Vec::with_capacity(raws.len());
    for r in &raws {
        let size = NcValue::element_size(r.ty).ok_or(bad("NetCDF: unknown variable type"))?;
        let per: usize = r.dims.iter().filter(|&&d| dimensions[d].1.is_some()).map(|&d| dimensions[d].1.expect("fixed")).product();
        let nbytes = per * size;
        let data = if is_record(r) {
            let mut v = NcValue::empty_like(r.ty);
            for k in 0..numrecs {
                let at = r.begin + k * recsize;
                let slab = bytes.get(at..at + nbytes).ok_or(bad("NetCDF: truncated record data"))?;
                v.append(NcValue::decode(r.ty, slab)?);
            }
            v
        } else {
            NcValue::decode(r.ty, bytes.get(r.begin..r.begin + nbytes).ok_or(bad("NetCDF: truncated data"))?)?
        };
        variables.push(NcVariable { name: r.name.clone(), dims: r.dims.clone(), attributes: r.attributes.clone(), data });
    }
    Ok(NetCdf { dimensions, attributes, variables, numrecs, version })
}
// ANCHOR_END: netcdf_decode

fn put_name(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u32).to_be_bytes());
    out.extend(s.as_bytes());
    out.resize(out.len().next_multiple_of(4), 0);
}

fn put_attributes(out: &mut Vec<u8>, atts: &[(String, NcValue)]) {
    if atts.is_empty() {
        out.extend([0u8; 8]);
        return;
    }
    out.extend(0x0Cu32.to_be_bytes());
    out.extend((atts.len() as u32).to_be_bytes());
    for (name, v) in atts {
        put_name(out, name);
        out.extend(v.type_code().to_be_bytes());
        out.extend((v.len() as u32).to_be_bytes());
        out.extend(v.encode());
        out.resize(out.len().next_multiple_of(4), 0);
    }
}

// ANCHOR: netcdf_encode
/// Writes a NetCDF-3 file (SciPy's `netcdf_file(..., "w")`) in the declared version: the header,
/// then the fixed-size variables, then the records. The number of records is taken from the
/// record variables' data, which must agree.
pub fn netcdf_encode(nc: &NetCdf) -> Result<Vec<u8>, ErrorsJSL> {
    let version = if nc.version == 2 { 2 } else { 1 };
    if nc.dimensions.iter().filter(|d| d.1.is_none()).count() > 1 {
        return Err(bad("NetCDF: at most one unlimited dimension"));
    }
    let fixed = |v: &NcVariable| -> usize { v.dims.iter().filter_map(|&d| nc.dimensions[d].1).product() };
    let is_rec = |v: &NcVariable| v.dims.first().is_some_and(|&d| nc.dimensions[d].1.is_none());
    for v in &nc.variables {
        if v.dims.iter().any(|&d| d >= nc.dimensions.len()) || v.dims.iter().skip(1).any(|&d| nc.dimensions[d].1.is_none()) {
            return Err(bad("NetCDF: the unlimited dimension may only be a variable's first"));
        }
    }
    // The number of records.
    let mut numrecs = None;
    for v in nc.variables.iter().filter(|v| is_rec(v)) {
        let per = fixed(v);
        if per == 0 || v.data.len() % per != 0 {
            return Err(bad("NetCDF: record variable data is not a whole number of records"));
        }
        let n = v.data.len() / per;
        if numrecs.is_some_and(|m| m != n) {
            return Err(bad("NetCDF: record variables have different numbers of records"));
        }
        numrecs = Some(n);
    }
    let numrecs = numrecs.unwrap_or(0);
    for v in nc.variables.iter().filter(|v| !is_rec(v)) {
        if v.data.len() != fixed(v) {
            return Err(bad("NetCDF: a variable's data does not match its dimensions"));
        }
    }
    let size_of = |v: &NcVariable| NcValue::element_size(v.data.type_code()).expect("valid type") * fixed(v);
    let rec_count = nc.variables.iter().filter(|v| is_rec(v)).count();
    let vsize = |v: &NcVariable| if is_rec(v) && rec_count == 1 { size_of(v) } else { size_of(v).next_multiple_of(4) };
    // The header, first with placeholder offsets to learn its length.
    let header = |begins: &[usize]| -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(b"CDF");
        out.push(version);
        out.extend((numrecs as u32).to_be_bytes());
        if nc.dimensions.is_empty() {
            out.extend([0u8; 8]);
        } else {
            out.extend(0x0Au32.to_be_bytes());
            out.extend((nc.dimensions.len() as u32).to_be_bytes());
            for (name, len) in &nc.dimensions {
                put_name(&mut out, name);
                out.extend((len.unwrap_or(0) as u32).to_be_bytes());
            }
        }
        put_attributes(&mut out, &nc.attributes);
        if nc.variables.is_empty() {
            out.extend([0u8; 8]);
        } else {
            out.extend(0x0Bu32.to_be_bytes());
            out.extend((nc.variables.len() as u32).to_be_bytes());
            for (k, v) in nc.variables.iter().enumerate() {
                put_name(&mut out, &v.name);
                out.extend((v.dims.len() as u32).to_be_bytes());
                for &d in &v.dims {
                    out.extend((d as u32).to_be_bytes());
                }
                put_attributes(&mut out, &v.attributes);
                out.extend(v.data.type_code().to_be_bytes());
                out.extend((vsize(v) as u32).to_be_bytes());
                if version == 1 {
                    out.extend((begins[k] as u32).to_be_bytes());
                } else {
                    out.extend((begins[k] as u64).to_be_bytes());
                }
            }
        }
        out
    };
    let hlen = header(&vec![0; nc.variables.len()]).len();
    let mut begins = vec![0usize; nc.variables.len()];
    let mut at = hlen;
    for (k, v) in nc.variables.iter().enumerate().filter(|(_, v)| !is_rec(v)) {
        begins[k] = at;
        at += vsize(v);
    }
    let rec_start = at;
    let mut off = rec_start;
    for (k, v) in nc.variables.iter().enumerate().filter(|(_, v)| is_rec(v)) {
        begins[k] = off;
        off += vsize(v);
    }
    let recsize: usize = nc.variables.iter().filter(|v| is_rec(v)).map(vsize).sum();
    if version == 1 && rec_start + recsize * numrecs > u32::MAX as usize {
        return Err(bad("NetCDF: too large for the classic format; use version 2"));
    }
    let mut out = header(&begins);
    for v in nc.variables.iter().filter(|v| !is_rec(v)) {
        let mut b = v.data.encode();
        b.resize(vsize(v), 0);
        out.extend(b);
    }
    let encoded: Vec<(usize, Vec<u8>, usize)> = nc.variables.iter().filter(|v| is_rec(v)).map(|v| (size_of(v), v.data.encode(), vsize(v))).collect();
    for r in 0..numrecs {
        for (size, bytes, vs) in &encoded {
            let mut slab = bytes[r * size..(r + 1) * size].to_vec();
            slab.resize(*vs, 0);
            out.extend(slab);
        }
    }
    Ok(out)
}
// ANCHOR_END: netcdf_encode

/// Reads a NetCDF-3 file.
pub fn read_netcdf_file(path: impl AsRef<Path>) -> Result<NetCdf, ErrorsJSL> {
    netcdf_decode(&std::fs::read(path).map_err(|_| ErrorsJSL::RuntimeError("could not read the NetCDF file"))?)
}

/// Writes a NetCDF-3 file.
pub fn write_netcdf_file(path: impl AsRef<Path>, nc: &NetCdf) -> Result<(), ErrorsJSL> {
    std::fs::write(path, netcdf_encode(nc)?).map_err(|_| ErrorsJSL::RuntimeError("could not write the NetCDF file"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_with_records() {
        for version in [1u8, 2] {
            let mut nc = NetCdf::new()
                .with_dimension("time", None)
                .with_dimension("x", Some(3))
                .with_attribute("title", NcValue::Char("test".into()))
                .with_variable(NcVariable::new("x", vec![1], NcValue::Double(vec![0.5, 1.5, 2.5])).with_attribute("units", NcValue::Char("m".into())))
                .with_variable(NcVariable::new("temp", vec![0, 1], NcValue::Float(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])))
                .with_variable(NcVariable::new("flag", vec![0], NcValue::Short(vec![7, -8])))
                .with_variable(NcVariable::new("b", vec![1], NcValue::Byte(vec![1, -2, 3])));
            nc.version = version;
            let bytes = netcdf_encode(&nc).unwrap();
            let back = netcdf_decode(&bytes).unwrap();
            assert_eq!(back.numrecs, 2);
            assert_eq!(back.variables, nc.variables);
            assert_eq!(back.attributes, nc.attributes);
            assert_eq!(back.shape(back.variable("temp").unwrap()), vec![2, 3]);
        }
    }
}
