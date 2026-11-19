//! Attribute-Relation File Format (ARFF) files from Weka (SciPy's `io.arff.loadarff`).
//!
//! An ARFF file is text: `@relation name`, one `@attribute name type` line per column (`numeric`,
//! `real`, `integer`, a nominal set `{a, b, c}`, `string` or `date`), then `@data` and one
//! comma-separated row per instance, `?` marking a missing value. Comments start with `%`.
//!
//! See the book entry [Scientific data files](https://jwkunz.github.io/lib_jsl/book/foundations/scientific-files.html#fortran-records-and-arff).

use lib_jsl_core::ErrorsJSL;
use std::path::Path;

/// The type of an ARFF attribute.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ArffType {
    /// `numeric`, `real` or `integer`.
    Numeric,
    /// A nominal attribute with its allowed values.
    Nominal(Vec<String>),
    String,
    /// A date, with its format (empty for the ISO-8601 default); values are kept as text.
    Date(String),
}

/// One value of a row.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ArffValue {
    /// A numeric value; NaN when missing (as SciPy reads it).
    Number(f64),
    /// A nominal, string or date value.
    Text(String),
    /// A missing nominal, string or date value (`?`).
    Missing,
}

/// The contents of an ARFF file.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Arff {
    pub relation: String,
    pub attributes: Vec<(String, ArffType)>,
    pub rows: Vec<Vec<ArffValue>>,
}

impl Arff {
    /// A numeric column as `f64` (NaN where missing).
    pub fn column_f64(&self, index: usize) -> Option<Vec<f64>> {
        if self.attributes.get(index)?.1 != ArffType::Numeric {
            return None;
        }
        Some(self.rows.iter().map(|r| if let ArffValue::Number(v) = r[index] { v } else { f64::NAN }).collect())
    }
}

fn bad(msg: &'static str) -> ErrorsJSL {
    ErrorsJSL::InvalidInputRange(msg)
}

/// Splits on commas outside single or double quotes, trimming and unquoting each field.
fn split_fields(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match quote {
            Some(q) if c == '\\' => {
                if let Some(n) = chars.next() {
                    cur.push(n);
                }
                let _ = q;
            }
            Some(q) if c == q => quote = None,
            Some(_) => cur.push(c),
            None if c == '\'' || c == '"' => quote = Some(c),
            None if c == ',' => {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            None => cur.push(c),
        }
    }
    out.push(cur.trim().to_string());
    out
}

/// The next token of a header line: a quoted string or a run of non-blank characters.
fn take_token(s: &str) -> Option<(String, &str)> {
    let s = s.trim_start();
    let first = s.chars().next()?;
    if first == '\'' || first == '"' {
        let end = s[1..].find(first)? + 1;
        Some((s[1..end].to_string(), &s[end + 1..]))
    } else {
        let end = s.find(char::is_whitespace).unwrap_or(s.len());
        Some((s[..end].to_string(), &s[end..]))
    }
}

// ANCHOR: arff
/// Parses an ARFF file (SciPy's `loadarff`, which additionally rejects string attributes).
/// Numeric values are `f64`, with missing values NaN; nominal values are checked against their
/// declared set.
pub fn arff_decode(text: &str) -> Result<Arff, ErrorsJSL> {
    let mut relation = String::new();
    let mut attributes = Vec::new();
    let mut rows = Vec::new();
    let mut in_data = false;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('%') {
            continue;
        }
        if !in_data {
            let lower = line.to_ascii_lowercase();
            if lower.starts_with("@relation") {
                relation = take_token(&line[9..]).map(|t| t.0).unwrap_or_default();
            } else if lower.starts_with("@attribute") {
                let (name, rest) = take_token(&line[10..]).ok_or(bad("ARFF: an attribute has no name"))?;
                let rest = rest.trim();
                let lr = rest.to_ascii_lowercase();
                let ty = if rest.starts_with('{') {
                    let inner = rest.trim_start_matches('{').trim_end_matches('}');
                    ArffType::Nominal(split_fields(inner))
                } else if lr.starts_with("numeric") || lr.starts_with("real") || lr.starts_with("integer") {
                    ArffType::Numeric
                } else if lr.starts_with("string") {
                    ArffType::String
                } else if lr.starts_with("date") {
                    ArffType::Date(take_token(&rest[4..]).map(|t| t.0).unwrap_or_default())
                } else {
                    return Err(bad("ARFF: unsupported attribute type (relational attributes are not supported)"));
                };
                attributes.push((name, ty));
            } else if lower.starts_with("@data") {
                in_data = true;
            } else {
                return Err(bad("ARFF: unexpected line in the header"));
            }
            continue;
        }
        if line.starts_with('{') {
            return Err(bad("ARFF: sparse data rows are not supported"));
        }
        let fields = split_fields(line);
        if fields.len() != attributes.len() {
            return Err(bad("ARFF: a data row has the wrong number of values"));
        }
        let mut row = Vec::with_capacity(fields.len());
        for (f, (_, ty)) in fields.into_iter().zip(&attributes) {
            let missing = f == "?";
            row.push(match ty {
                ArffType::Numeric => ArffValue::Number(if missing { f64::NAN } else { f.parse().map_err(|_| bad("ARFF: a numeric value is not a number"))? }),
                _ if missing => ArffValue::Missing,
                ArffType::Nominal(set) => {
                    if !set.contains(&f) {
                        return Err(bad("ARFF: a nominal value is not in its declared set"));
                    }
                    ArffValue::Text(f)
                }
                _ => ArffValue::Text(f),
            });
        }
        rows.push(row);
    }
    if !in_data {
        return Err(bad("ARFF: no @data section"));
    }
    Ok(Arff { relation, attributes, rows })
}
// ANCHOR_END: arff

fn quote(s: &str) -> String {
    if s.is_empty() || s.contains([' ', ',', '{', '}', '%', '\'', '"']) {
        format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
    } else {
        s.to_string()
    }
}

/// Writes an ARFF file (a dense data section; SciPy has no writer).
pub fn arff_encode(a: &Arff) -> String {
    let mut out = format!("@relation {}\n\n", quote(&a.relation));
    for (name, ty) in &a.attributes {
        let t = match ty {
            ArffType::Numeric => "numeric".to_string(),
            ArffType::Nominal(v) => format!("{{{}}}", v.iter().map(|s| quote(s)).collect::<Vec<_>>().join(", ")),
            ArffType::String => "string".into(),
            ArffType::Date(f) if f.is_empty() => "date".into(),
            ArffType::Date(f) => format!("date {}", quote(f)),
        };
        out.push_str(&format!("@attribute {} {t}\n", quote(name)));
    }
    out.push_str("\n@data\n");
    for r in &a.rows {
        let fields: Vec<String> = r
            .iter()
            .map(|v| match v {
                ArffValue::Number(x) if x.is_nan() => "?".into(),
                ArffValue::Number(x) => format!("{x}"),
                ArffValue::Text(s) => quote(s),
                ArffValue::Missing => "?".into(),
            })
            .collect();
        out.push_str(&fields.join(","));
        out.push('\n');
    }
    out
}

/// Reads an ARFF file.
pub fn read_arff_file(path: impl AsRef<Path>) -> Result<Arff, ErrorsJSL> {
    arff_decode(&std::fs::read_to_string(path).map_err(|_| ErrorsJSL::RuntimeError("could not read the ARFF file"))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_round_trips() {
        let text = "% iris excerpt\n@RELATION iris\n@ATTRIBUTE sepallength NUMERIC\n@ATTRIBUTE 'petal width' REAL\n@ATTRIBUTE class {Iris-setosa,Iris-versicolor}\n@DATA\n5.1,0.2,Iris-setosa\n?,1.3,Iris-versicolor\n4.9,0.2,?\n";
        let a = arff_decode(text).unwrap();
        assert_eq!(a.relation, "iris");
        assert_eq!(a.attributes[1].0, "petal width");
        let c = a.column_f64(0).unwrap();
        assert!(c[1].is_nan() && c[2] == 4.9);
        assert_eq!(a.rows[2][2], ArffValue::Missing);
        let back = arff_decode(&arff_encode(&a)).unwrap();
        assert_eq!(back.rows.len(), 3);
        assert_eq!(back.attributes, a.attributes);
        assert!(arff_decode("@relation r\n@attribute c {a,b}\n@data\nz\n").is_err());
    }
}
