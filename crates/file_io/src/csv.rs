//! Delimited text tables (CSV, RFC 4180).
//!
//! Fields may be quoted with `"`, and quoted fields may contain the delimiter, line breaks and
//! doubled quotes (`""` for a literal `"`). Records end with LF or CRLF; a trailing line break is
//! optional. Writing quotes exactly the fields that need it.

use lib_jsl_core::ErrorsJSL;
use std::path::Path;

/// A parsed table: an optional header row and the records, all as strings.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CsvTable {
    pub header: Option<Vec<String>>,
    pub rows: Vec<Vec<String>>,
}

impl CsvTable {
    /// The index of the column named `name` in the header.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.header.as_ref()?.iter().position(|h| h == name)
    }

    /// Column `index` parsed as numbers (empty fields become NaN).
    pub fn column_f64(&self, index: usize) -> Result<Vec<f64>, ErrorsJSL> {
        self.rows.iter().map(|row| parse_number(row.get(index).map(String::as_str).unwrap_or(""))).collect()
    }

    /// The whole table parsed as numbers, row-major (empty fields become NaN). Every row must
    /// have the same number of fields.
    pub fn to_f64_rows(&self) -> Result<Vec<Vec<f64>>, ErrorsJSL> {
        let width = self.rows.first().map_or(0, |r| r.len());
        self.rows
            .iter()
            .map(|row| {
                if row.len() != width {
                    return Err(ErrorsJSL::IncompatibleArraySizes((row.len(), width)));
                }
                row.iter().map(|f| parse_number(f)).collect()
            })
            .collect()
    }
}

fn parse_number(field: &str) -> Result<f64, ErrorsJSL> {
    let t = field.trim();
    if t.is_empty() {
        return Ok(f64::NAN);
    }
    t.parse::<f64>().map_err(|_| ErrorsJSL::RuntimeError("A CSV field is not a number."))
}

/// Parses delimited text. With `has_header` the first record becomes the header.
pub fn parse_csv(text: &str, delimiter: char, has_header: bool) -> Result<CsvTable, ErrorsJSL> {
    if delimiter == '"' || delimiter == '\n' || delimiter == '\r' {
        return Err(ErrorsJSL::InvalidInputRange("The delimiter cannot be a quote or a line break."));
    }
    let mut records: Vec<Vec<String>> = Vec::new();
    let mut record = Vec::new();
    let mut field = String::new();
    let mut chars = text.chars().peekable();
    let mut in_quotes = false;
    let mut field_started = false;
    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            } else {
                field.push(c);
            }
            continue;
        }
        match c {
            '"' if field.is_empty() && !field_started => {
                in_quotes = true;
                field_started = true;
            }
            '"' => return Err(ErrorsJSL::RuntimeError("A quote appears inside an unquoted CSV field.")),
            c if c == delimiter => {
                record.push(std::mem::take(&mut field));
                field_started = false;
            }
            '\r' if chars.peek() == Some(&'\n') => {}
            '\n' | '\r' => {
                record.push(std::mem::take(&mut field));
                records.push(std::mem::take(&mut record));
                field_started = false;
            }
            c => {
                field.push(c);
                field_started = true;
            }
        }
    }
    if in_quotes {
        return Err(ErrorsJSL::RuntimeError("Unterminated quoted CSV field."));
    }
    if field_started || !field.is_empty() || !record.is_empty() {
        record.push(field);
        records.push(record);
    }
    let header = if has_header && !records.is_empty() { Some(records.remove(0)) } else { None };
    Ok(CsvTable { header, rows: records })
}

fn write_field(out: &mut String, field: &str, delimiter: char) {
    if field.contains(delimiter) || field.contains('"') || field.contains('\n') || field.contains('\r') || field.starts_with(' ') || field.ends_with(' ') {
        out.push('"');
        out.push_str(&field.replace('"', "\"\""));
        out.push('"');
    } else {
        out.push_str(field);
    }
}

/// Formats a table as delimited text with LF line endings.
pub fn format_csv(table: &CsvTable, delimiter: char) -> String {
    let mut out = String::new();
    for record in table.header.iter().chain(&table.rows) {
        for (i, field) in record.iter().enumerate() {
            if i > 0 {
                out.push(delimiter);
            }
            write_field(&mut out, field, delimiter);
        }
        out.push('\n');
    }
    out
}

/// Formats numeric rows (with an optional header); numbers use Rust's shortest round-trip
/// representation, so parsing the text back recovers them exactly.
pub fn format_csv_f64(header: Option<&[&str]>, rows: &[Vec<f64>], delimiter: char) -> String {
    let table = CsvTable {
        header: header.map(|h| h.iter().map(|s| s.to_string()).collect()),
        rows: rows.iter().map(|r| r.iter().map(|v| if v.is_nan() { String::new() } else { format!("{v:?}") }).collect()).collect(),
    };
    format_csv(&table, delimiter)
}

/// Reads a delimited text file.
pub fn read_csv_file(path: impl AsRef<Path>, delimiter: char, has_header: bool) -> Result<CsvTable, ErrorsJSL> {
    let text = std::fs::read_to_string(path).map_err(|_| ErrorsJSL::RuntimeError("Could not read the CSV file."))?;
    parse_csv(text.strip_prefix('\u{feff}').unwrap_or(&text), delimiter, has_header)
}

/// Writes a delimited text file.
pub fn write_csv_file(path: impl AsRef<Path>, table: &CsvTable, delimiter: char) -> Result<(), ErrorsJSL> {
    std::fs::write(path, format_csv(table, delimiter)).map_err(|_| ErrorsJSL::RuntimeError("Could not write the CSV file."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_quoting_line_endings_and_headers() {
        let text = "name,value,note\r\n\"Smith, J\",3.5,\"said \"\"hi\"\"\"\nplain,-1e3,\"two\nlines\"\nempty,,\n";
        let t = parse_csv(text, ',', true).unwrap();
        assert_eq!(t.header.as_deref(), Some(&["name".to_string(), "value".into(), "note".into()][..]));
        assert_eq!(t.rows[0], vec!["Smith, J", "3.5", "said \"hi\""]);
        assert_eq!(t.rows[1][2], "two\nlines");
        assert_eq!(t.rows[2], vec!["empty", "", ""]);
        let values = t.column_f64(t.column_index("value").unwrap()).unwrap();
        assert_eq!(&values[..2], &[3.5, -1000.0]);
        assert!(values[2].is_nan());
        // No trailing newline, other delimiters, and a quoted empty field.
        let t = parse_csv("1;2\n\"\";4", ';', false).unwrap();
        assert_eq!(t.rows, vec![vec!["1", "2"], vec!["", "4"]]);
        assert!(parse_csv("a,\"open", ',', false).is_err() && parse_csv("a,b\"c", ',', false).is_err());
        assert!(t.to_f64_rows().is_ok() && parse_csv("1,2\n3", ',', false).unwrap().to_f64_rows().is_err());
    }

    #[test]
    fn writing_round_trips_strings_and_numbers_exactly() {
        let table = CsvTable {
            header: Some(vec!["a".into(), "b c".into()]),
            rows: vec![vec!["x,y".into(), "q\"uote".into()], vec![" padded ".into(), "multi\nline".into()]],
        };
        assert_eq!(parse_csv(&format_csv(&table, ','), ',', true).unwrap(), table);
        let rows = vec![vec![0.1, -2.5e-300, f64::MAX], vec![1.0 / 3.0, f64::NAN, -0.0]];
        let parsed = parse_csv(&format_csv_f64(Some(&["p", "q", "r"]), &rows, '\t'), '\t', true).unwrap().to_f64_rows().unwrap();
        for (a, b) in parsed.iter().flatten().zip(rows.iter().flatten()) {
            assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()));
        }
    }
}
