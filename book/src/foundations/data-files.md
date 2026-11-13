# Data files: CSV and NumPy .npy

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Exchange tables and arrays with spreadsheets, databases and Python: delimited text that any program can read, and a binary array format that keeps types and shapes exactly |
| **Devised by** | Delimited text since the 1970s, standardised by Yakov Shafranovich in RFC 4180 (2005); Robert Kern and the NumPy developers (2007), the `.npy` format |
| **Accuracy** | Python's `csv` module and NumPy read the library's files identically, and the library reads every tested `.npy` dtype, byte order, memory order and format version identically (tables below) |
| **Cost** | $O(\text{size})$ |
| **API** | [`parse_csv`](../../api/lib_jsl_file_io/csv/fn.parse_csv.html), [`format_csv`](../../api/lib_jsl_file_io/csv/fn.format_csv.html), [`format_csv_f64`](../../api/lib_jsl_file_io/csv/fn.format_csv_f64.html), [`read_csv_file`](../../api/lib_jsl_file_io/csv/fn.read_csv_file.html), [`write_csv_file`](../../api/lib_jsl_file_io/csv/fn.write_csv_file.html), [`CsvTable`](../../api/lib_jsl_file_io/csv/struct.CsvTable.html) (`column_index`, `column_f64`, `to_f64_rows`), [`npy_decode`](../../api/lib_jsl_file_io/npy/fn.npy_decode.html), [`npy_encode`](../../api/lib_jsl_file_io/npy/fn.npy_encode.html), [`read_npy_file`](../../api/lib_jsl_file_io/npy/fn.read_npy_file.html), [`write_npy_file`](../../api/lib_jsl_file_io/npy/fn.write_npy_file.html), [`NpyArray`](../../api/lib_jsl_file_io/npy/struct.NpyArray.html) (`to_f64`, `from_f64`), [`NpyData`](../../api/lib_jsl_file_io/npy/enum.NpyData.html) |

</div>

## Overview

Scientific work moves data between programs constantly. **CSV** (comma-separated values) is the universal table format. Every spreadsheet, database and plotting tool reads it, and a person can read it too. Its weaknesses are that numbers become text and that fields containing commas, quotes or line breaks must be quoted. **`.npy`** is NumPy's binary format for one array: a small text header giving the element type, the memory order and the shape, then the raw bytes. It round-trips floats exactly, loads fast, and is the easiest way to hand arrays to and from Python.

## History

Comma-separated lists of values appear in IBM's FORTRAN list-directed input of 1972, and the format spread with spreadsheets from VisiCalc (1979) onwards. It went unstandardised for decades, which is why programs still disagree on quoting, delimiters and line endings. Yakov Shafranovich's RFC 4180 (2005) wrote down the common rules: fields separated by commas, records by line breaks, and fields with special characters wrapped in double quotes, with embedded quotes doubled. NumPy's `.npy` format was proposed by Robert Kern in NEP 1 (2007) as a simple, documented replacement for pickling arrays. Version 2.0 (2014) allowed larger headers, and version 3.0 (2018) UTF-8 field names in structured types.

## The idea

### CSV

The parser reads characters into fields, with a flag for whether it is inside quotes. Outside quotes, the delimiter ends a field and a line break (LF or CRLF) ends a record. Inside quotes, everything is literal except a doubled quote, which stands for one quote character, and a single quote, which closes the field. A quote in the middle of an unquoted field, or a quoted field never closed, is an error rather than a guess. The writer quotes exactly the fields that need it:

```rust,ignore
{{#include ../../../crates/file_io/src/csv.rs:write_field}}
```

For numbers, `format_csv_f64` writes Rust's shortest representation that parses back to the same `f64`, so numeric tables round-trip exactly, and writes `NaN` as an empty field. `column_f64` and `to_f64_rows` read empty fields back as `NaN`. A blank line is therefore a record with one empty field, not nothing, unlike Python's `csv` module, so that one-column tables with missing values survive a round trip.

### NumPy .npy

The file starts with the magic string `\x93NUMPY`, a major and minor version, and a little-endian header length: two bytes in version 1.0, four in 2.0 and 3.0. The header is a Python dictionary literal such as `{'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }`, padded with spaces so the data start on a 64-byte boundary. The `descr` code gives the byte order (`<` little, `>` big, `|` not applicable), the kind (`b` boolean, `i` signed, `u` unsigned, `f` float, `c` complex) and the size in bytes.

```rust,ignore
{{#include ../../../crates/file_io/src/npy.rs:npy_decode}}
```

The reader handles booleans, signed and unsigned integers of 1 to 8 bytes, 32- and 64-bit floats and 64- and 128-bit complex numbers, in either byte order and in C or Fortran order, and returns elements in C (row-major) order as an [`NpyData`](../../api/lib_jsl_file_io/npy/enum.NpyData.html) variant of the right type. `to_f64` converts any real type to an `ndarray` of `f64`. The writer produces version 1.0, little-endian and C order, which every NumPy reads.

### Reading and writing other programs' files

Files written by Python's libraries, read by this one:

<p class="caption" id="tab-26-8"><strong>Table 26.8.</strong> Columns: file written by Python, bytes, read by the library.</p>

{{#include ../generated/foundations/reading_data.md}}

Files written by this library, read by Python's:

<p class="caption" id="tab-26-9"><strong>Table 26.9.</strong> Columns: file written by the library (read by), result.</p>

{{#include ../generated/foundations/interop_data.md}}

## Algorithm

**Parsing CSV.** The parser reads one character at a time and is in one of two states.

- *Outside quotes*: a delimiter ends the current field; a line ending ends the current field
  and the record; a quote at the start of a field enters the quoted state; any other
  character is added to the field.
- *Inside quotes*: two quotes in a row add one quote to the field; a single quote leaves the
  quoted state; any other character, including a delimiter or a line ending, is added to the
  field.

When reading a file, a byte-order mark at the very start is removed first.

**Writing CSV**: for each field, if it contains the delimiter, a quote, a carriage return or a
line feed, write it between quotes with every quote inside it doubled; otherwise write it as
it is.

**Reading an `.npy` file.**

1. Check the six-byte magic string and read the version.
2. Read the length of the header: two bytes in version 1, four in later versions.
3. Parse the header, a Python dictionary written as text, for the three keys `descr` (the
   element type and byte order), `fortran_order` and `shape`.
4. Check that the amount of data that follows equals the product of the shape times the
   element size.
5. Read the elements, swapping bytes if the file's byte order differs from the machine's.
6. If the file is in Fortran (column-major) order, transpose to C (row-major) order.

**Writing an `.npy` file**: build the header dictionary; pad it with spaces and a final
newline so that the data begins at a multiple of 64 bytes; write the magic string, version,
header length and header; append the elements in little-endian order.

## Implementation

- `CsvTable` holds an optional header and rows of `String` fields; rows may have different lengths, and `to_f64_rows` checks them.
- `NpyArray::new` checks that the shape matches the data length; an empty shape is a scalar.

## In practice

```rust,ignore
{{#include ../../../crates/file_io/examples/file_io_data.rs}}
```

```text
header Some(["city", "population", "note"])
  ["Paris", "2102650", "capital, France"]
  ["Saint-Étienne", "172718", "say \"Sainté\""]
total population 2275368
written back:
city;population;note
Paris;2102650;capital, France
Saint-Étienne;172718;"say ""Sainté"""
numbers (NaN as an empty field):
x,y
0.1,1e-300
,-2.5
.npy: 176 bytes, header "{'descr': '<f8', 'fortran_order': False, 'shape':"
shape [2, 3], identical: true
uint16 array as f64: [1.0, 2.0, 3.0, 4.0]
```

Writing back with semicolons quotes only the field containing quotes. Its comma no longer needs protecting.

**Strengths**

- RFC 4180 CSV with any delimiter, exact numeric round trips, and strict errors on malformed quoting.
- `.npy` reading for the common numeric dtypes in any byte order, memory order and format version, checked against NumPy.

**Weaknesses**

- No type inference, comment lines or dialect detection for CSV, and the whole file is read into memory.
- No `.npz` archives, half-precision floats, structured (record) dtypes, strings or memory mapping.

## Exercises

1. Write the CSV state machine as a table of (state, character) transitions, and check how it handles `"a""b",c`.
2. Show that writing $x$ with the shortest round-trip representation and parsing it back gives exactly $x$.
3. Work out the header padding of a version 1.0 `.npy` file for a `(3, 4)` float64 array.
4. Convert a Fortran-order index $(i, j, k)$ of an array of shape $(a, b, c)$ to its C-order position.
5. Why does NumPy put the data on a 64-byte boundary?

<details>
<summary>Hints and answers</summary>

1. The states are: field start, in unquoted field, in quoted field, quote seen inside a quoted field. In `"a""b",c` the first quote opens a quoted field, the doubled quote yields a literal `"`, the next quote closes the field, and the comma ends it, giving fields `a"b` and `c`.
2. The shortest round-trip representation is the shortest decimal string that parses back to the same double; by definition parsing it recovers $x$ exactly.
3. The header string for shape `(3, 4)` is `{'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }`, 59 characters. The preamble is 10 bytes, plus the newline: 70 bytes, padded with spaces to 128, so the header length field is 118.
4. In Fortran order the element $(i, j, k)$ is at offset $i + a(j + bk)$; in C order it is at $(ib + j)c + k$.
5. A 64-byte boundary matches the cache-line size and the width of AVX-512 vectors, so the array data can be memory-mapped and used with aligned loads.

</details>

## Further reading

- Y. Shafranovich, "Common format and MIME type for comma-separated values (CSV) files", RFC 4180, 2005.
- R. Kern, "A simple file format for NumPy arrays", NumPy Enhancement Proposal 1, 2007.
- C. R. Harris et al., "Array programming with NumPy", *Nature* 585 (2020) 357–362.
