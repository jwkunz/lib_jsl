# Scientific data files: Matrix Market, MATLAB, Harwell-Boeing, NetCDF, Fortran and ARFF

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Exchange matrices and labelled arrays with MATLAB, Fortran programs, climate and ocean models, sparse-matrix collections and machine-learning tools, in the formats `scipy.io` reads and writes |
| **Devised by** | Ronald Boisvert, Roldan Pozo and Karin Remington (1996), Matrix Market; Iain Duff, Roger Grimes and John Lewis (1989), the Harwell-Boeing collection; The MathWorks (1999), the level-5 MAT-file; Russ Rew and Glenn Davis at Unidata (1990), NetCDF; the University of Waikato (1999), ARFF for Weka |
| **Accuracy** | Values round-trip exactly; SciPy reads every file the library writes, and the library reads every file SciPy writes (table below) |
| **Cost** | $O(\text{size})$; compressed MAT-files add deflate's cost |
| **Read first** | [Data files: CSV and NumPy .npy](data-files.md) |
| **API** | [`matrix_market`](../../api/lib_jsl_file_io/matrix_market/index.html) (`mm_decode`, `mm_encode`, `mm_info`, `MatrixMarket`); [`matlab`](../../api/lib_jsl_file_io/matlab/index.html) (`mat_decode`, `mat_encode`, `mat_whos`, `MatValue`); [`harwell_boeing`](../../api/lib_jsl_file_io/harwell_boeing/index.html) (`hb_decode`, `hb_encode`); [`netcdf`](../../api/lib_jsl_file_io/netcdf/index.html) (`netcdf_decode`, `netcdf_encode`, `NetCdf`); [`fortran`](../../api/lib_jsl_file_io/fortran/index.html) (`fortran_records_decode`, `fortran_records_encode`); [`arff`](../../api/lib_jsl_file_io/arff/index.html) (`arff_decode`, `arff_encode`); each with `read_*_file` and `write_*_file` |

</div>

## Overview

`scipy.io` reads and writes the file formats that scientific computing has accumulated, and this entry brings them to the library:

- **Matrix Market** (`.mtx`): text files for sparse and dense matrices, the format of the SuiteSparse Matrix Collection and the common currency of sparse linear algebra.
- **MATLAB MAT-files** (level 5, `.mat`): binary files of named variables, the way MATLAB saves and loads data.
- **Harwell-Boeing** (`.hb`, `.rua`): the older fixed-width Fortran format of the original Harwell-Boeing sparse matrix collection.
- **NetCDF-3** (`.nc`): self-describing binary files of named, dimensioned variables with attributes. It is the standard for climate, ocean and atmospheric data.
- **Fortran unformatted sequential files**: the raw binary records Fortran programs write.
- **ARFF** (`.arff`): Weka's text format for machine-learning data sets, with numeric and nominal attributes.

## History

The Harwell-Boeing collection of sparse test matrices (Duff, Grimes and Lewis, 1989) fixed its storage format in 80-column Fortran records. Ronald Boisvert, Roldan Pozo and Karin Remington at NIST replaced it with the simpler Matrix Market format in 1996. The MathWorks introduced the level-5 MAT-file with MATLAB 5 (1999) and added compression in MATLAB 7. Version 7.3 files are HDF5, a different format altogether.

NetCDF was designed at Unidata by Russ Rew and Glenn Davis (1990), after NASA's Common Data Format. Its classic format is still the most widely shared. Fortran's record markers date back to the earliest unformatted I/O. The University of Waikato defined ARFF for the Weka machine-learning workbench.

## The idea

### Matrix Market

A banner line declares the object, the format (`coordinate` for sparse triplets, `array` for dense column-major values), the field (`real`, `integer`, `complex`, `pattern`) and the symmetry (`general`, `symmetric`, `skew-symmetric`, `hermitian`). After comment lines and a size line come the entries, with 1-based indices. A symmetric file stores only the lower triangle. A reader rebuilds the rest by mirroring each off-diagonal entry, negating it for skew-symmetric matrices and conjugating it for Hermitian ones.

### MATLAB level-5 files

A 128-byte header ends with a version and the characters `IM`, read as `MI` on a machine of the other byte order. Then come **data elements**: an 8-byte tag (type, byte count) and the data, padded to 8 bytes. Elements of at most 4 bytes pack tag and data into 8 bytes.

A variable is a `miMATRIX` element containing nested elements: the array flags (class, complex, logical), the dimensions, the name, and the data in column-major order. Each part is stored in whatever integer or floating type is smallest. Sparse matrices store their compressed-column arrays, cells and structures store nested `miMATRIX` elements, and `miCOMPRESSED` wraps one `miMATRIX` in a zlib stream.

### Harwell-Boeing

The header gives the number of 80-column lines in each section, the matrix type (`RUA`: real, unsymmetric, assembled), the dimensions, and Fortran edit descriptors such as `(26I3)` and `(3E25.16)` for the sections. The body is the 1-based compressed-sparse-column pointers, row indices and values, in those fixed-width fields.

### NetCDF-3

The header lists:

- the dimensions, one of which may be unlimited (the record dimension);
- the global attributes;
- the variables, each with its dimensions, attributes, type, size and file offset.

Fixed-size variables follow one after another. Variables along the record dimension are interleaved: record 0 of every record variable, then record 1, and so on. That is what lets a file grow by appending records. Everything is big-endian and padded to 4 bytes. Version 2 widens the offsets to 64 bits.

### Fortran records and ARFF

A Fortran unformatted sequential write frames each record with its byte count, before and after, so a reader can skip records in either direction. ARFF is plain text: `@relation`, one `@attribute` per column, then `@data` with comma-separated rows and `?` for missing values.

## Implementation

```rust,ignore
{{#include ../../../crates/file_io/src/matlab.rs:loadmat}}
```

```rust,ignore
{{#include ../../../crates/file_io/src/netcdf.rs:netcdf_decode}}
```

```rust,ignore
{{#include ../../../crates/file_io/src/matrix_market.rs:mm_decode}}
```

- The readers take bytes or text, and the writers return them, so they work in memory. `read_*_file` and `write_*_file` wrap them for files.
- Matrix Market and Harwell-Boeing matrices come back as triplets or compressed sparse columns, ready for the library's sparse matrices.
- MATLAB numeric data are returned as `f64` with their class recorded. Values of 64-bit integers beyond $2^{53}$ lose precision.
- Not supported:
  - MATLAB v4 files, v7.3 (HDF5) files, objects and function handles;
  - NetCDF-4 (HDF5) files;
  - complex and elemental Harwell-Boeing matrices;
  - sparse ARFF rows;
  - IDL `.sav` files.

## In practice

The files were written by SciPy 1.17.1 with `book/generator/data/make_io_refs.py`, read by the library, written again by the library, and read back by SciPy:

<p class="caption" id="tab-26-10"><strong>Table 26.10.</strong> File exchange with `scipy.io` in both directions.</p>

{{#include ../generated/foundations/io_formats.md}}

Every value survives both directions exactly: floating-point numbers are written in binary, or as text with enough digits to read back the same value. SciPy reads MATLAB logical arrays as `uint8` whether the library or SciPy wrote them.

**Strengths**

- Pure Rust readers and writers, with no HDF5, NetCDF or MATLAB libraries.
- Symmetric Matrix Market and Harwell-Boeing files are expanded on reading, and symmetric matrices can be written in half the space.
- MAT-files of either byte order, compressed or not, with nested cells and structures.

**Weaknesses**

- Whole files are read into memory; there is no memory mapping or partial reading of large NetCDF variables.
- No HDF5-based formats (MATLAB v7.3, NetCDF-4), which need an HDF5 implementation.
- No IDL `.sav` reader.

## Exercises

1. How many entries does the `array` format store for a symmetric $n \times n$ matrix, and for a skew-symmetric one?
2. Why can a NetCDF file grow along the unlimited dimension without rewriting the variables already written?
3. A MAT-file element of type `miDOUBLE` holds 12 bytes. How many bytes does it occupy in the file?
4. Why does a Fortran unformatted record carry its length at both ends?

<details>
<summary>Hints and answers</summary>

1. $n(n+1)/2$, the lower triangle with the diagonal, and $n(n-1)/2$ for skew-symmetric matrices, whose diagonal is zero.
2. Records are stored after all fixed-size data, and each new record appends one slab of every record variable at the end of the file. Only the record count in the header changes.
3. 8 bytes of tag plus 12 bytes of data padded to 16: 24 bytes.
4. So that a program can `BACKSPACE`: from the trailing marker it can find the start of the record it has just read.

</details>

## Further reading

- R. F. Boisvert, R. Pozo and K. A. Remington, *The Matrix Market Exchange Formats: Initial Design*, NIST report NISTIR 5935 (1996).
- I. S. Duff, R. G. Grimes and J. G. Lewis, "Sparse matrix test problems", *ACM Transactions on Mathematical Software* 15 (1989) 1–14.
- The MathWorks, *MAT-File Format*, version 7 (2023).
- R. Rew and G. Davis, "NetCDF: an interface for scientific data access", *IEEE Computer Graphics and Applications* 10 (1990) 76–82; Unidata, *The NetCDF Classic Format Specification*.
- T. A. Davis and Y. Hu, "The University of Florida sparse matrix collection", *ACM Transactions on Mathematical Software* 38 (2011) 1:1–1:25.
