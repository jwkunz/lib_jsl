//! Atmospheric attenuation: gaseous absorption, cloud and fog, and rain.
//!
//! * Oxygen and water vapour by the simplified sea-level formulas of ITU-R P.676 (Annex 2, at
//!   15 °C and 1013 hPa): oxygen valid below 57 GHz, water vapour below 350 GHz.
//! * Cloud and fog liquid water by the Rayleigh model of ITU-R P.840 with the double-Debye
//!   permittivity of water.
//! * Rain by the power law `γ_R = k R^α` with coefficients supplied by the caller (for example
//!   from ITU-R P.838 for the frequency and polarization).

use lib_jsl_core::ErrorsJSL;

/// The specific attenuation of dry air (dB/km) at `f < 57` GHz, sea level and 15 °C.
pub fn oxygen_specific_attenuation(frequency_ghz: f64) -> Result<f64, ErrorsJSL> {
    if !(frequency_ghz > 0.0 && frequency_ghz < 57.0) {
        return Err(ErrorsJSL::InvalidInputRange("The simplified oxygen model holds below 57 GHz."));
    }
    let f = frequency_ghz;
    Ok((7.19e-3 + 6.09 / (f * f + 0.227) + 4.81 / ((f - 57.0).powi(2) + 1.50)) * f * f * 1e-3)
}

/// The specific attenuation of water vapour (dB/km) at `f < 350` GHz for density `ρ` (g/m³;
/// 7.5 is the standard mid-latitude value), sea level and 15 °C. It peaks at the 22.2, 183.3 and
/// 325.4 GHz absorption lines.
pub fn water_vapour_specific_attenuation(frequency_ghz: f64, density: f64) -> Result<f64, ErrorsJSL> {
    if !(frequency_ghz > 0.0 && frequency_ghz < 350.0) || density < 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("The simplified water vapour model holds below 350 GHz."));
    }
    let f = frequency_ghz;
    let lines = 0.050 + 0.0021 * density + 3.6 / ((f - 22.2).powi(2) + 8.5) + 10.6 / ((f - 183.3).powi(2) + 9.0) + 8.9 / ((f - 325.4).powi(2) + 26.3);
    Ok(lines * f * f * density * 1e-4)
}

/// The total gaseous attenuation (dB) along a slant path at elevation `elevation` (radians,
/// above about 10° for the flat-layer approximation), using equivalent heights of 6 km for
/// oxygen and 2.1 km for water vapour.
pub fn gaseous_slant_attenuation(frequency_ghz: f64, density: f64, elevation: f64) -> Result<f64, ErrorsJSL> {
    if !(elevation > 0.0 && elevation <= std::f64::consts::FRAC_PI_2) {
        return Err(ErrorsJSL::InvalidInputRange("The elevation must be in (0, π/2]."));
    }
    let zenith = 6.0 * oxygen_specific_attenuation(frequency_ghz)? + 2.1 * water_vapour_specific_attenuation(frequency_ghz, density)?;
    Ok(zenith / elevation.sin())
}

/// The complex relative permittivity `ε' − jε''` of liquid water at `temperature` (K) by the
/// double-Debye model of ITU-R P.840.
pub fn water_permittivity(frequency_ghz: f64, temperature: f64) -> (f64, f64) {
    let theta = 300.0 / temperature;
    let e0 = 77.66 + 103.3 * (theta - 1.0);
    let (e1, e2) = (0.0671 * e0, 3.52);
    let fp = 20.20 - 146.0 * (theta - 1.0) + 316.0 * (theta - 1.0).powi(2);
    let fs = 39.8 * fp;
    let f = frequency_ghz;
    let imaginary = f * (e0 - e1) / (fp * (1.0 + (f / fp).powi(2))) + f * (e1 - e2) / (fs * (1.0 + (f / fs).powi(2)));
    let real = (e0 - e1) / (1.0 + (f / fp).powi(2)) + (e1 - e2) / (1.0 + (f / fs).powi(2)) + e2;
    (real, imaginary)
}

/// The cloud and fog specific attenuation coefficient `K_l` ((dB/km)/(g/m³)) in the Rayleigh
/// regime (valid to about 200 GHz); multiply by the liquid water density.
pub fn cloud_specific_attenuation_coefficient(frequency_ghz: f64, temperature: f64) -> Result<f64, ErrorsJSL> {
    if !(frequency_ghz > 0.0 && frequency_ghz <= 1000.0) || !(temperature > 200.0 && temperature < 330.0) {
        return Err(ErrorsJSL::InvalidInputRange("Need 0 < f <= 1000 GHz and a temperature in kelvin."));
    }
    let (real, imaginary) = water_permittivity(frequency_ghz, temperature);
    let eta = (2.0 + real) / imaginary;
    Ok(0.819 * frequency_ghz / (imaginary * (1.0 + eta * eta)))
}

/// The rain specific attenuation `k R^α` (dB/km) for rain rate `R` (mm/h).
pub fn rain_specific_attenuation(rain_rate: f64, k: f64, alpha: f64) -> f64 {
    k * rain_rate.max(0.0).powf(alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaseous_attenuation_has_the_expected_magnitudes_and_line_shapes() {
        // About 0.007 dB/km for oxygen at 10 GHz and 0.18 dB/km of water vapour at 22.2 GHz.
        assert!((oxygen_specific_attenuation(10.0).unwrap() - 0.0070).abs() < 3e-4);
        let peak = water_vapour_specific_attenuation(22.2, 7.5).unwrap();
        assert!((peak - 0.181).abs() < 0.005, "{peak}");
        // The 22 GHz line is a local maximum and oxygen rises towards 60 GHz.
        assert!(peak > water_vapour_specific_attenuation(18.0, 7.5).unwrap() && peak > water_vapour_specific_attenuation(26.0, 7.5).unwrap());
        assert!(oxygen_specific_attenuation(55.0).unwrap() > 10.0 * oxygen_specific_attenuation(30.0).unwrap());
        assert_eq!(water_vapour_specific_attenuation(10.0, 0.0).unwrap(), 0.0);
        // Slant paths scale as 1 / sin(elevation).
        let zenith = gaseous_slant_attenuation(20.0, 7.5, std::f64::consts::FRAC_PI_2).unwrap();
        assert!((gaseous_slant_attenuation(20.0, 7.5, std::f64::consts::FRAC_PI_6).unwrap() - 2.0 * zenith).abs() < 1e-12);
        assert!(oxygen_specific_attenuation(60.0).is_err() && gaseous_slant_attenuation(10.0, 7.5, 0.0).is_err());
    }

    #[test]
    fn cloud_attenuation_and_water_permittivity() {
        // Static permittivity near 88 at 0 °C, falling with frequency.
        let (e_low, _) = water_permittivity(0.001, 273.15);
        assert!((e_low - 87.9).abs() < 1.0, "{e_low}");
        let k10 = cloud_specific_attenuation_coefficient(10.0, 273.15).unwrap();
        let k30 = cloud_specific_attenuation_coefficient(30.0, 273.15).unwrap();
        // Rayleigh scattering: roughly quadratic in frequency at low frequencies.
        assert!(k30 / k10 > 6.0 && k30 / k10 < 9.5, "{k10} {k30}");
        // Colder droplets absorb more at these frequencies.
        assert!(cloud_specific_attenuation_coefficient(30.0, 263.15).unwrap() > cloud_specific_attenuation_coefficient(30.0, 293.15).unwrap());
        assert!((rain_specific_attenuation(25.0, 0.0188, 1.217) - 0.0188 * 25f64.powf(1.217)).abs() < 1e-15);
    }
}
