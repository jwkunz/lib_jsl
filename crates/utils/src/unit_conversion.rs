//! Conversion between common units of measure.
//!
//! Units are looked up by symbol or name, ignoring spaces. An exact, case-sensitive match is tried
//! first (`mW` is a milliwatt and `MW` a megawatt, `MB` a megabyte and `Mb` a megabit); otherwise
//! case is ignored if that leaves only one unit (`hz`, `pa` and `kwh` work), and a spelling that
//! fits several units, such as `mw`, is an error rather than a guess. Conversion is allowed only
//! within one physical dimension; temperatures use affine conversions.

use lib_jsl_core::ErrorsJSL;

/// Physical dimension of a unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    Length,
    Mass,
    Time,
    Temperature,
    Angle,
    Frequency,
    Area,
    Volume,
    Speed,
    Pressure,
    Energy,
    Power,
    Force,
    Data,
}

struct Unit {
    names: &'static [&'static str],
    dim: Dimension,
    /// Value in the dimension's base unit of one of this unit (`base = value * scale + offset`).
    scale: f64,
    offset: f64,
}

macro_rules! unit {
    ($dim:ident, $scale:expr, [$($n:expr),+]) => {
        Unit { names: &[$($n),+], dim: Dimension::$dim, scale: $scale, offset: 0.0 }
    };
}

// Base units: m, kg, s, K, rad, Hz, m², m³, m/s, Pa, J, W, N, bit.
const UNITS: &[Unit] = &[
    unit!(Length, 1.0, ["m", "meter", "meters", "metre", "metres"]),
    unit!(Length, 1e-3, ["mm", "millimeter", "millimeters"]),
    unit!(Length, 1e-2, ["cm", "centimeter", "centimeters"]),
    unit!(Length, 1e3, ["km", "kilometer", "kilometers"]),
    unit!(Length, 1e-6, ["um", "µm", "micrometer", "micron"]),
    unit!(Length, 1e-9, ["nm", "nanometer"]),
    unit!(Length, 0.0254, ["in", "inch", "inches"]),
    unit!(Length, 0.3048, ["ft", "foot", "feet"]),
    unit!(Length, 0.9144, ["yd", "yard", "yards"]),
    unit!(Length, 1609.344, ["mi", "mile", "miles"]),
    unit!(Length, 1852.0, ["nmi", "nauticalmile", "nauticalmiles"]),
    unit!(Length, 149_597_870_700.0, ["au", "astronomicalunit"]),
    unit!(Length, 9_460_730_472_580_800.0, ["ly", "lightyear", "lightyears"]),
    unit!(Mass, 1.0, ["kg", "kilogram", "kilograms"]),
    unit!(Mass, 1e-3, ["g", "gram", "grams"]),
    unit!(Mass, 1e-6, ["mg", "milligram"]),
    unit!(Mass, 1e3, ["t", "tonne", "tonnes"]),
    unit!(Mass, 0.453_592_37, ["lb", "lbs", "pound", "pounds"]),
    unit!(Mass, 0.028_349_523_125, ["oz", "ounce", "ounces"]),
    unit!(Time, 1.0, ["s", "sec", "second", "seconds"]),
    unit!(Time, 1e-3, ["ms", "millisecond"]),
    unit!(Time, 1e-6, ["us", "µs", "microsecond"]),
    unit!(Time, 1e-9, ["ns", "nanosecond"]),
    unit!(Time, 60.0, ["min", "minute", "minutes"]),
    unit!(Time, 3600.0, ["h", "hr", "hour", "hours"]),
    unit!(Time, 86_400.0, ["d", "day", "days"]),
    unit!(Time, 604_800.0, ["week", "weeks"]),
    Unit { names: &["k", "kelvin"], dim: Dimension::Temperature, scale: 1.0, offset: 0.0 },
    Unit { names: &["c", "degc", "celsius"], dim: Dimension::Temperature, scale: 1.0, offset: 273.15 },
    Unit { names: &["f", "degf", "fahrenheit"], dim: Dimension::Temperature, scale: 5.0 / 9.0, offset: 255.372_222_222_222_2 },
    Unit { names: &["r", "rankine"], dim: Dimension::Temperature, scale: 5.0 / 9.0, offset: 0.0 },
    unit!(Angle, 1.0, ["rad", "radian", "radians"]),
    unit!(Angle, std::f64::consts::PI / 180.0, ["deg", "degree", "degrees"]),
    unit!(Angle, std::f64::consts::PI / 200.0, ["grad", "gradian", "gradians"]),
    unit!(Angle, std::f64::consts::TAU, ["turn", "turns", "rev", "revolution"]),
    unit!(Angle, std::f64::consts::PI / 10_800.0, ["arcmin"]),
    unit!(Angle, std::f64::consts::PI / 648_000.0, ["arcsec"]),
    unit!(Frequency, 1.0, ["hz", "hertz"]),
    unit!(Frequency, 1e3, ["kHz", "kilohertz"]),
    unit!(Frequency, 1e6, ["MHz", "megahertz"]),
    unit!(Frequency, 1e9, ["GHz", "gigahertz"]),
    unit!(Frequency, 1.0 / 60.0, ["rpm"]),
    unit!(Frequency, std::f64::consts::TAU, ["rad/s"]),
    unit!(Area, 1.0, ["m2", "m^2", "sqm"]),
    unit!(Area, 1e-4, ["cm2", "cm^2"]),
    unit!(Area, 1e6, ["km2", "km^2"]),
    unit!(Area, 10_000.0, ["ha", "hectare", "hectares"]),
    unit!(Area, 4_046.856_422_4, ["acre", "acres"]),
    unit!(Area, 0.092_903_04, ["ft2", "ft^2", "sqft"]),
    unit!(Volume, 1.0, ["m3", "m^3"]),
    unit!(Volume, 1e-3, ["l", "liter", "liters", "litre", "litres"]),
    unit!(Volume, 1e-6, ["ml", "milliliter", "cm3", "cc"]),
    unit!(Volume, 3.785_411_784e-3, ["gal", "gallon", "gallons"]),
    unit!(Volume, 2.365_882_365e-4, ["cup", "cups"]),
    unit!(Volume, 0.028_316_846_592, ["ft3", "ft^3"]),
    unit!(Speed, 1.0, ["m/s", "mps"]),
    unit!(Speed, 1.0 / 3.6, ["km/h", "kph", "kmh"]),
    unit!(Speed, 0.447_04, ["mph", "mi/h"]),
    unit!(Speed, 1852.0 / 3600.0, ["kn", "kt", "knot", "knots"]),
    unit!(Speed, 0.3048, ["ft/s", "fps"]),
    unit!(Speed, 299_792_458.0, ["c0", "lightspeed"]),
    unit!(Pressure, 1.0, ["pa", "pascal"]),
    unit!(Pressure, 1e3, ["kpa"]),
    unit!(Pressure, 1e5, ["bar"]),
    unit!(Pressure, 100.0, ["mbar", "hpa"]),
    unit!(Pressure, 101_325.0, ["atm"]),
    unit!(Pressure, 133.322_387_415, ["mmHg"]),
    unit!(Pressure, 101_325.0 / 760.0, ["torr"]),
    unit!(Pressure, 6_894.757_293_168, ["psi"]),
    unit!(Energy, 1.0, ["j", "joule", "joules"]),
    unit!(Energy, 1e3, ["kj"]),
    unit!(Energy, 4.184, ["cal"]),
    unit!(Energy, 4184.0, ["kcal"]),
    unit!(Energy, 3.6e6, ["kwh"]),
    unit!(Energy, 1.602_176_634e-19, ["ev"]),
    unit!(Energy, 1_055.055_852_62, ["btu"]),
    unit!(Power, 1.0, ["w", "watt", "watts"]),
    unit!(Power, 1e3, ["kW", "kilowatt", "kilowatts"]),
    unit!(Power, 1e-3, ["mW", "milliwatt", "milliwatts"]),
    unit!(Power, 1e6, ["MW", "megawatt", "megawatts"]),
    unit!(Power, 745.699_871_582_270_2, ["hp"]),
    unit!(Force, 1.0, ["n", "newton", "newtons"]),
    unit!(Force, 4.448_221_615_260_5, ["lbf"]),
    unit!(Force, 9.806_65, ["kgf"]),
    unit!(Data, 1.0, ["bit", "bits"]),
    unit!(Data, 8.0, ["byte", "bytes"]),
    unit!(Data, 1e3, ["kb", "kbit", "kilobit"]),
    unit!(Data, 1e6, ["Mb", "Mbit", "megabit"]),
    unit!(Data, 1e9, ["Gb", "Gbit", "gigabit"]),
    unit!(Data, 8e3, ["kB", "kilobyte"]),
    unit!(Data, 8e6, ["MB", "megabyte"]),
    unit!(Data, 8e9, ["GB", "gigabyte"]),
    unit!(Data, 8192.0, ["KiB", "kibibyte"]),
    unit!(Data, 8_388_608.0, ["MiB", "mebibyte"]),
    unit!(Data, 8_589_934_592.0, ["GiB", "gibibyte"]),
];

fn lookup(name: &str) -> Result<&'static Unit, ErrorsJSL> {
    let key: String = name.chars().filter(|c| !c.is_whitespace()).collect();
    if let Some(unit) = UNITS.iter().find(|u| u.names.contains(&key.as_str())) {
        return Ok(unit);
    }
    let lower = key.to_lowercase();
    let mut matches = UNITS.iter().filter(|u| u.names.iter().any(|n| n.to_lowercase() == lower));
    match (matches.next(), matches.next()) {
        (Some(unit), None) => Ok(unit),
        (Some(_), Some(_)) => Err(ErrorsJSL::InvalidInputRange("Ambiguous unit: give its case-sensitive symbol, such as mW or MW.")),
        _ => Err(ErrorsJSL::InvalidInputRange("Unknown unit.")),
    }
}

/// The dimension of a named unit.
pub fn unit_dimension(name: &str) -> Result<Dimension, ErrorsJSL> {
    lookup(name).map(|u| u.dim)
}

/// Converts `value` from unit `from` to unit `to`, e.g. `unit_conversion(100.0, "km/h", "mph")`.
///
/// Errors for unknown units or when the units belong to different dimensions.
pub fn unit_conversion(value: f64, from: &str, to: &str) -> Result<f64, ErrorsJSL> {
    let (a, b) = (lookup(from)?, lookup(to)?);
    if a.dim != b.dim {
        return Err(ErrorsJSL::InvalidInputRange("Units have different dimensions."));
    }
    Ok(((value * a.scale + a.offset) - b.offset) / b.scale)
}

/// Converts a power ratio to decibels, `10 log10(ratio)`.
pub fn linear_power_to_db(ratio: f64) -> f64 {
    10.0 * ratio.log10()
}

/// Converts decibels to a power ratio.
pub fn db_to_linear_power(db: f64) -> f64 {
    10f64.powf(db / 10.0)
}

/// Converts an amplitude (voltage) ratio to decibels, `20 log10(ratio)`.
pub fn linear_amplitude_to_db(ratio: f64) -> f64 {
    20.0 * ratio.log10()
}

/// Converts decibels to an amplitude ratio.
pub fn db_to_linear_amplitude(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

/// Power in watts to dBm.
pub fn watts_to_dbm(watts: f64) -> f64 {
    10.0 * (watts * 1e3).log10()
}

/// dBm to power in watts.
pub fn dbm_to_watts(dbm: f64) -> f64 {
    10f64.powf(dbm / 10.0) * 1e-3
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * b.abs().max(1.0)
    }

    #[test]
    fn linear_conversions() {
        assert!(close(unit_conversion(1.0, "mi", "km").unwrap(), 1.609344));
        assert!(close(unit_conversion(100.0, "km/h", "mph").unwrap(), 62.137_119_223_733_4));
        assert!(close(unit_conversion(180.0, "deg", "rad").unwrap(), std::f64::consts::PI));
        assert!(close(unit_conversion(1.0, "atm", "psi").unwrap(), 14.695_948_775_322_4));
        assert!(close(unit_conversion(1.0, "GiB", "MB").unwrap(), 1_073.741_824));
        assert!(close(unit_conversion(60.0, "rpm", "Hz").unwrap(), 1.0));
        assert!(close(unit_conversion(2.0, " Hours ", "min").unwrap(), 120.0));
    }

    #[test]
    fn temperatures_are_affine() {
        assert!(close(unit_conversion(100.0, "C", "F").unwrap(), 212.0));
        assert!(close(unit_conversion(32.0, "F", "C").unwrap(), 0.0));
        assert!(close(unit_conversion(0.0, "K", "C").unwrap(), -273.15));
        assert!(close(unit_conversion(-40.0, "C", "F").unwrap(), -40.0));
    }

    #[test]
    fn round_trips_and_errors() {
        for (a, b) in [("m", "ft"), ("kcal", "kwh"), ("gal", "l"), ("F", "R")] {
            let v = unit_conversion(unit_conversion(3.7, a, b).unwrap(), b, a).unwrap();
            assert!(close(v, 3.7), "{a}->{b}");
        }
        assert!(unit_conversion(1.0, "m", "kg").is_err());
        // Case decides SI prefixes when it must; an ambiguous spelling is refused.
        assert!(close(unit_conversion(1.0, "mW", "W").unwrap(), 1e-3));
        assert!(close(unit_conversion(1.0, "MW", "W").unwrap(), 1e6));
        assert!(unit_conversion(1.0, "mw", "W").is_err());
        assert!(close(unit_conversion(1.0, "MB", "Mb").unwrap(), 8.0));
        assert!(close(unit_conversion(1.0, "mhz", "kHz").unwrap(), 1000.0));
        assert!(close(unit_conversion(760.0, "torr", "atm").unwrap(), 1.0));
        // The conventional millimetre of mercury is 1.4e-7 larger than the torr (101 325/760 Pa).
        let ratio = unit_conversion(1.0, "mmHg", "torr").unwrap();
        assert!(ratio > 1.0 + 1e-7 && ratio < 1.0 + 2e-7);
        assert!(unit_conversion(1.0, "parsec", "m").is_err());
        assert_eq!(unit_dimension("psi").unwrap(), Dimension::Pressure);
    }

    #[test]
    fn decibel_helpers() {
        assert!(close(linear_power_to_db(100.0), 20.0));
        assert!(close(db_to_linear_power(3.0), 1.995_262_314_968_88));
        assert!(close(linear_amplitude_to_db(10.0), 20.0));
        assert!(close(db_to_linear_amplitude(-6.0), 0.501_187_233_627_272_3));
        assert!(close(watts_to_dbm(1.0), 30.0));
        assert!(close(dbm_to_watts(0.0), 1e-3));
    }
}
