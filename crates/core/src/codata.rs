//! The CODATA 2022 recommended values of the fundamental physical constants, looked up by name
//! with their units and standard uncertainties (SciPy's `scipy.constants.physical_constants`,
//! `value`, `unit`, `precision` and `find`).
//!
//! The table holds SciPy's 445 entries: the 355 constants of CODATA 2022 and 90 names kept from
//! earlier adjustments (with their older values), which are flagged
//! [`PhysicalConstant::obsolete`] and left out of [`find`].
//!
//! ```
//! use lib_jsl_core::codata::{find, physical_constant, value};
//!
//! let me = physical_constant("electron mass").unwrap();
//! assert_eq!(me.value, 9.1093837139e-31);
//! assert_eq!(me.unit, "kg");
//! assert_eq!(me.uncertainty, 2.8e-40);
//! assert_eq!(value("speed of light in vacuum"), Some(299792458.0));
//! assert!(find("Bohr radius").contains(&"Bohr radius"));
//! ```
//!
//! See the book entry [Shared types, constants and utilities](https://jwkunz.github.io/lib_jsl/book/foundations/core-utils.html#codata-constants-by-name).

#[path = "codata_table.rs"]
mod codata_table;
use codata_table::TABLE;

// ANCHOR: codata
/// One entry of the CODATA table: SciPy's `physical_constants[name]` as a record.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct PhysicalConstant {
    /// The name as CODATA spells it, e.g. `"electron mass"`.
    pub name: &'static str,
    /// The recommended value, in `unit`.
    pub value: f64,
    /// The SI unit, e.g. `"kg"` or `"J T^-1"`; empty for dimensionless constants.
    pub unit: &'static str,
    /// The standard (one-sigma) absolute uncertainty; 0 for exact constants.
    pub uncertainty: f64,
    /// Whether the name comes from an earlier CODATA adjustment and is no longer current.
    pub obsolete: bool,
}

impl PhysicalConstant {
    /// The relative uncertainty `uncertainty / |value|` (SciPy's `precision`, which divides by
    /// the signed value).
    pub fn precision(&self) -> f64 {
        self.uncertainty / self.value.abs()
    }

    /// Whether the value is exact by definition (zero uncertainty).
    pub fn is_exact(&self) -> bool {
        self.uncertainty == 0.0
    }
}

fn entry(row: &'static (&'static str, f64, &'static str, f64, bool)) -> PhysicalConstant {
    PhysicalConstant { name: row.0, value: row.1, unit: row.2, uncertainty: row.3, obsolete: row.4 }
}

/// The constant called `name` (exact, case-sensitive match), or `None`.
pub fn physical_constant(name: &str) -> Option<PhysicalConstant> {
    TABLE.iter().find(|r| r.0 == name).map(entry)
}

/// The value of the constant called `name` (SciPy's `value`).
pub fn value(name: &str) -> Option<f64> {
    physical_constant(name).map(|c| c.value)
}

/// The unit of the constant called `name` (SciPy's `unit`).
pub fn unit(name: &str) -> Option<&'static str> {
    physical_constant(name).map(|c| c.unit)
}

/// The relative uncertainty of the constant called `name`.
pub fn precision(name: &str) -> Option<f64> {
    physical_constant(name).map(|c| c.precision())
}

/// The names of the current (CODATA 2022) constants that contain `sub`, ignoring case, in
/// sorted order (SciPy's `find`). An empty `sub` lists them all.
pub fn find(sub: &str) -> Vec<&'static str> {
    let sub = sub.to_lowercase();
    let mut out: Vec<&'static str> = TABLE.iter().filter(|r| !r.4 && r.0.to_lowercase().contains(&sub)).map(|r| r.0).collect();
    out.sort_unstable();
    out
}

/// Every entry of the table, current and obsolete, in SciPy's order.
pub fn all() -> impl Iterator<Item = PhysicalConstant> {
    TABLE.iter().map(entry)
}
// ANCHOR_END: codata

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::physics;

    #[test]
    fn table_and_lookups() {
        assert_eq!(all().count(), 445);
        assert_eq!(all().filter(|c| !c.obsolete).count(), 355);
        assert_eq!(find("").len(), 355);
        assert_eq!(find("radius"), vec!["Bohr radius", "alpha particle rms charge radius", "classical electron radius", "deuteron rms charge radius", "proton rms charge radius"]);
        assert!(physical_constant("Planck constant").unwrap().is_exact());
        assert_eq!(value("Planck constant"), Some(physics::H));
        assert_eq!(unit("Boltzmann constant"), Some("J K^-1"));
        assert!((precision("proton mass").unwrap() - 3.1e-10).abs() < 1e-11);
        assert!(physical_constant("mag. constant").unwrap().obsolete);
        assert_eq!(value("no such constant"), None);
    }
}
