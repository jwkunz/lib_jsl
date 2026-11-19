//! The table for release 2.12.0 (foundations chapter): Matrix Market, MATLAB, Harwell-Boeing,
//! NetCDF-3, Fortran unformatted and ARFF files exchanged with SciPy in both directions. The
//! library reads the files SciPy wrote (`data/io/scipy_*`), writes its own (`data/io/library_*`),
//! and `data/make_io_refs.py` records whether SciPy reads those back.

use crate::refs::Refs;
use crate::spectral::md_table;
use crate::write;
use lib_jsl_file_io::*;
use num_complex::Complex;
use std::collections::HashMap;

fn io_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join("io")
}

fn grid(v: &[f64], m: usize, n: usize) -> Vec<Vec<f64>> {
    (0..m).map(|i| v[i * n..(i + 1) * n].to_vec()).collect()
}

fn dense_rows(a: &ndarray::Array2<f64>) -> Vec<Vec<f64>> {
    a.rows().into_iter().map(|r| r.to_vec()).collect()
}

fn triplets(a: &[Vec<f64>]) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for (i, row) in a.iter().enumerate() {
        for (j, &x) in row.iter().enumerate() {
            if x != 0.0 {
                r.push(i);
                c.push(j);
                v.push(x);
            }
        }
    }
    (r, c, v)
}

/// Compressed sparse columns of a dense matrix.
fn csc(a: &[Vec<f64>]) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let (m, n) = (a.len(), a[0].len());
    let (mut ptr, mut ri, mut v) = (vec![0], Vec::new(), Vec::new());
    for j in 0..n {
        for (i, row) in a.iter().enumerate().take(m) {
            if row[j] != 0.0 {
                ri.push(i);
                v.push(row[j]);
            }
        }
        ptr.push(ri.len());
    }
    (ptr, ri, v)
}

fn yes(b: bool) -> String {
    if b { "yes".into() } else { "**no**".into() }
}

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/io_reference.txt"));
    let dir = io_dir();
    let read = |name: &str| std::fs::read(dir.join(name)).unwrap_or_else(|_| panic!("missing {name}; run make_io_refs.py"));
    let text = |name: &str| String::from_utf8(read(name)).expect("text file");
    let dense = grid(refs.values("dense"), 4, 5);
    let sparse = grid(refs.values("sparse"), 30, 20);
    let sym = grid(refs.values("sym"), 12, 12);
    let (cre, cim) = (grid(refs.values("cplx re"), 6, 6), grid(refs.values("cplx im"), 6, 6));
    let nc_temp: Vec<f32> = refs.values("nc temp").iter().map(|&v| v as f32).collect();
    let nc_time = refs.values("nc time").to_vec();
    let scipy_reads: HashMap<String, bool> = std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/io_scipy_reads.txt"))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| l.split_once('|').map(|(k, v)| (k.to_string(), v.trim() == "1")))
        .collect();
    let back = |name: &str| scipy_reads.get(name).map_or("pending".to_string(), |&ok| yes(ok));
    let mut rows = Vec::new();

    // Matrix Market.
    let general = mm_decode(&text("scipy_general.mtx")).unwrap();
    let symm = mm_decode(&text("scipy_symmetric.mtx")).unwrap();
    let dm = mm_decode(&text("scipy_dense.mtx")).unwrap();
    let cm = mm_decode(&text("scipy_complex.mtx")).unwrap().to_dense_complex();
    let complex_ok = (0..6).all(|i| (0..6).all(|j| cm[(i, j)] == Complex::new(cre[i][j], cim[i][j])));
    let (r, c, v) = triplets(&sparse);
    write_mm_file(dir.join("library_general.mtx"), &MatrixMarket::from_triplets(30, 20, r, c, v).unwrap()).unwrap();
    let (r, c, v) = triplets(&sym);
    write_mm_file(dir.join("library_symmetric.mtx"), &MatrixMarket::from_triplets(12, 12, r, c, v).unwrap().with_symmetry(MmSymmetry::Symmetric)).unwrap();
    let da = ndarray::Array2::from_shape_fn((4, 5), |(i, j)| dense[i][j]);
    write_mm_file(dir.join("library_dense.mtx"), &MatrixMarket::from_dense(&da)).unwrap();
    let (mut cr, mut cc, mut cv) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..6 {
        for j in 0..6 {
            if cre[i][j] != 0.0 || cim[i][j] != 0.0 {
                cr.push(i);
                cc.push(j);
                cv.push(Complex::new(cre[i][j], cim[i][j]));
            }
        }
    }
    let mut cmm = MatrixMarket::from_triplets(6, 6, cr, cc, vec![0.0; cv.len()]).unwrap().with_field(MmField::Complex);
    cmm.values = cv;
    write_mm_file(dir.join("library_complex.mtx"), &cmm).unwrap();
    rows.push(vec!["Matrix Market, coordinate real general".into(), "30 × 20, 60 entries".into(), yes(dense_rows(&general.to_dense()) == sparse), back("library_general.mtx")]);
    rows.push(vec!["Matrix Market, coordinate real symmetric".into(), "12 × 12, lower triangle stored".into(), yes(dense_rows(&symm.to_dense()) == sym), back("library_symmetric.mtx")]);
    rows.push(vec!["Matrix Market, array real general".into(), "4 × 5 dense".into(), yes(dense_rows(&dm.to_dense()) == dense), back("library_dense.mtx")]);
    rows.push(vec!["Matrix Market, coordinate complex general".into(), "6 × 6".into(), yes(complex_ok), back("library_complex.mtx")]);

    // Harwell-Boeing.
    let hb = hb_decode(&text("scipy.hb")).unwrap();
    let (ptr, ri, v) = csc(&sparse);
    write_hb_file(dir.join("library.hb"), &HarwellBoeing::new(30, 20, ptr, ri, v).unwrap()).unwrap();
    rows.push(vec!["Harwell-Boeing (RUA)".into(), "30 × 20, 60 entries".into(), yes(hb.to_dense() == sparse), back("library.hb")]);

    // MATLAB.
    let mat_ok = |vars: &[(String, MatValue)]| -> bool {
        let get = |n: &str| vars.iter().find(|(k, _)| k == n).map(|(_, v)| v);
        let a_ok = get("a").and_then(|v| v.to_rows()) == Some(dense.clone());
        let z_ok = matches!(get("z"), Some(MatValue::Numeric { real, imag: Some(im), .. }) if real == &vec![1.0, 3.0] && im == &vec![-2.0, 0.5]);
        let i16_ok = matches!(get("i16"), Some(MatValue::Numeric { class: MatClass::Int16, real, .. }) if real == &vec![-7.0, 300.0]);
        let flag_ok = matches!(get("flag"), Some(MatValue::Logical { data, .. }) if data == &vec![true, false, true]);
        let s_ok = get("s").and_then(|v| v.as_string()).as_deref() == Some("hello world");
        let sp_ok = match get("sp") {
            Some(MatValue::Sparse { nrows, ncols, row_indices, col_ptr, real, .. }) => {
                let mut d = vec![vec![0.0; *ncols]; *nrows];
                for j in 0..*ncols {
                    for k in col_ptr[j]..col_ptr[j + 1] {
                        d[row_indices[k]][j] = real[k];
                    }
                }
                d == sym
            }
            _ => false,
        };
        let c_ok = matches!(get("c"), Some(MatValue::Cell { cells, .. }) if cells.len() == 2 && cells[1].as_string().as_deref() == Some("x"));
        let st_ok = matches!(get("st"), Some(MatValue::Struct { fields, elements, .. }) if fields == &vec!["alpha".to_string(), "beta".to_string()] && elements[0][1].as_string().as_deref() == Some("b"));
        let nd_ok = matches!(get("nd"), Some(MatValue::Numeric { dims, real, .. }) if dims == &vec![2, 3, 4] && real[1] == 12.0 && real[2] == 4.0);
        a_ok && z_ok && i16_ok && flag_ok && s_ok && sp_ok && c_ok && st_ok && nd_ok
    };
    let plain = mat_decode(&read("scipy_plain.mat")).unwrap();
    let comp = mat_decode(&read("scipy_compressed.mat")).unwrap();
    let (ptr, ri, v) = csc(&sym);
    let vars: Vec<(String, MatValue)> = vec![
        ("a".into(), MatValue::from_rows(&dense)),
        ("z".into(), MatValue::Numeric { class: MatClass::Double, dims: vec![1, 2], real: vec![1.0, 3.0], imag: Some(vec![-2.0, 0.5]) }),
        ("i16".into(), MatValue::Numeric { class: MatClass::Int16, dims: vec![2, 1], real: vec![-7.0, 300.0], imag: None }),
        ("flag".into(), MatValue::Logical { dims: vec![1, 3], data: vec![true, false, true] }),
        ("s".into(), MatValue::from_text("hello world")),
        ("sp".into(), MatValue::Sparse { nrows: 12, ncols: 12, row_indices: ri, col_ptr: ptr, real: v, imag: None, logical: false }),
        ("c".into(), MatValue::Cell { dims: vec![1, 2], cells: vec![MatValue::from_slice(&[1.0]), MatValue::from_text("x")] }),
        ("st".into(), MatValue::Struct { dims: vec![1, 1], fields: vec!["alpha".into(), "beta".into()], elements: vec![vec![MatValue::from_slice(&[2.5, 3.5]), MatValue::from_text("b")]] }),
        ("nd".into(), MatValue::Numeric { class: MatClass::Double, dims: vec![2, 3, 4], real: (0..24).map(|k| { let (i, j, l) = (k % 2, (k / 2) % 3, k / 6); (i * 12 + j * 4 + l) as f64 }).collect(), imag: None }),
    ];
    write_mat_file(dir.join("library_plain.mat"), &vars, false).unwrap();
    write_mat_file(dir.join("library_compressed.mat"), &vars, true).unwrap();
    let contents = "9 variables: double, complex, int16, logical, char, sparse, cell, struct, 3-D";
    rows.push(vec!["MATLAB level 5".into(), contents.into(), yes(mat_ok(&plain)), back("library_plain.mat")]);
    rows.push(vec!["MATLAB level 5, compressed".into(), contents.into(), yes(mat_ok(&comp)), back("library_compressed.mat")]);

    // NetCDF-3.
    let nc = netcdf_decode(&read("scipy.nc")).unwrap();
    let nc_ok = nc.numrecs == 5
        && nc.variable("temp").map(|v| &v.data) == Some(&NcValue::Float(nc_temp.clone()))
        && nc.variable("time").map(|v| &v.data) == Some(&NcValue::Double(nc_time.clone()))
        && nc.variable("time").is_some_and(|v| v.attributes.iter().any(|(k, x)| k == "units" && *x == NcValue::Char("days".into())))
        && nc.variable("station_id").map(|v| &v.data) == Some(&NcValue::Int(vec![11, 12, 13, 14]));
    let out = NetCdf::new()
        .with_dimension("time", None)
        .with_dimension("station", Some(4))
        .with_attribute("history", NcValue::Char("written by lib_jsl".into()))
        .with_variable(NcVariable::new("time", vec![0], NcValue::Double(nc_time.clone())).with_attribute("units", NcValue::Char("days".into())))
        .with_variable(NcVariable::new("temp", vec![0, 1], NcValue::Float(nc_temp.clone())))
        .with_variable(NcVariable::new("station_id", vec![1], NcValue::Int(vec![11, 12, 13, 14])))
        .with_variable(NcVariable::new("flag", vec![0], NcValue::Short(vec![1, 0, 1, 1, 0])));
    write_netcdf_file(dir.join("library.nc"), &out).unwrap();
    rows.push(vec!["NetCDF-3 classic".into(), "unlimited dimension, 3 record and 1 fixed variable, attributes".into(), yes(nc_ok), back("library.nc")]);

    // Fortran unformatted.
    let recs = fortran_records_decode(&read("scipy_fortran.dat"), 4).unwrap();
    let flat: Vec<f64> = dense.iter().flatten().copied().collect();
    let f_ok = recs.len() == 3 && record_to_i32(&recs[0]).unwrap() == vec![3, -4, 5] && record_to_f64(&recs[1]).unwrap() == flat && record_to_f32(&recs[2]).unwrap() == vec![0.5, 1.5];
    write_fortran_file(dir.join("library_fortran.dat"), &[record_from_i32(&[3, -4, 5]), record_from_f64(&flat), record_from_f32(&[0.5, 1.5])], 4).unwrap();
    rows.push(vec!["Fortran unformatted, 4-byte markers".into(), "3 records: int32, float64, float32".into(), yes(f_ok), back("library_fortran.dat")]);

    // ARFF.
    let a = arff_decode(&text("weather.arff")).unwrap();
    let same_nan = |x: &[f64], y: &[f64]| x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p == q || (p.is_nan() && q.is_nan()));
    let arff_ok = same_nan(&a.column_f64(1).unwrap(), refs.values("arff temperature")) && same_nan(&a.column_f64(2).unwrap(), refs.values("arff humidity")) && a.rows[4][4] == ArffValue::Missing;
    rows.push(vec!["ARFF".into(), "5 attributes (nominal and numeric), missing values".into(), yes(arff_ok), "(SciPy has no writer)".into()]);

    write("foundations", "io_formats.md", &md_table(&["Format", "Contents", "Library reads SciPy's file", "SciPy reads the library's file"], &rows));
}
