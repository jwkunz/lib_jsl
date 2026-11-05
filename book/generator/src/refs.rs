//! Reading the `case|values` reference files written by the `data/make_*_refs.py` scripts:
//! one case per line, values comma-separated, several lists separated by `;`.

use std::collections::HashMap;

pub struct Refs(HashMap<String, Vec<Vec<f64>>>);

impl Refs {
    pub fn parse(text: &str) -> Self {
        let parse = |v: &str| match v {
            "inf" => f64::INFINITY,
            "-inf" => f64::NEG_INFINITY,
            "nan" => f64::NAN,
            _ => v.parse::<f64>().unwrap_or_else(|_| panic!("not a number: {v}")),
        };
        Refs(
            text.lines()
                .filter(|l| !l.trim().is_empty())
                .map(|line| {
                    let (key, rest) = line.split_once('|').unwrap();
                    let lists = rest.split(';').map(|l| if l.is_empty() { Vec::new() } else { l.split(',').map(parse).collect() }).collect();
                    (key.to_string(), lists)
                })
                .collect(),
        )
    }

    /// The lists of a case.
    pub fn get(&self, key: &str) -> &Vec<Vec<f64>> {
        self.0.get(key).unwrap_or_else(|| panic!("no reference case {key:?}"))
    }

    /// The first list of a case.
    pub fn values(&self, key: &str) -> &[f64] {
        &self.get(key)[0]
    }
}

/// The largest absolute difference between two lists of equal length.
pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "lists of different lengths");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// The largest difference relative to the largest reference magnitude.
pub fn max_rel_diff(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(f64::MIN_POSITIVE);
    max_abs_diff(got, want) / scale
}

/// A difference formatted for a table: `0` for exact agreement, otherwise one significant digit.
pub fn fmt_diff(d: f64) -> String {
    if d == 0.0 { "0".to_string() } else { format!("{d:.0e}") }
}
