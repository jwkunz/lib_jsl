#![allow(clippy::excessive_precision, clippy::approx_constant)]
//! # `constants`
//!
//! A collection of `f64` constants commonly used in mathematics, physics,
//! electromagnetics, signal processing, and machine learning.
//!
//! ## Layout
//!
//! | Module      | Contents                                                          |
//! |-------------|-------------------------------------------------------------------|
//! | [`math`]    | Pi, e, golden ratio, Euler–Mascheroni, Apéry, Catalan, ...        |
//! | [`physics`] | CODATA fundamental constants, thermodynamics, astronomy           |
//! | [`em`]      | Vacuum permeability/permittivity, impedances, quantum EM units    |
//! | [`dsp`]     | dB conversions, window coefficients, audio, noise, quantization   |
//! | [`ml`]      | Activation constants, optimizer defaults, dataset statistics      |
//! | [`units`]   | Angle, length, mass, pressure, energy conversion factors          |
//!
//! ## Sources and conventions
//!
//! * Physical constants follow **CODATA 2022** (SI 2019 redefinition). Values
//!   that are *exact* by definition (`C`, `H`, `K_B`, `N_A`, `E_CHARGE`) are
//!   exact; measured values carry their CODATA uncertainty in the final digits.
//! * Where a constant is algebraically derived from others (e.g. `HBAR`,
//!   `R_GAS`, `MU_B`) it is computed at compile time from its definition so
//!   that the set stays self-consistent.
//! * All values are SI unless the name says otherwise (`_EV`, `_DEG`, `_HZ`...).
//! * Works in `#![no_std]` contexts: only `core` is used.
//!
//! ## Usage
//!
//! ```ignore
//! mod constants;
//! use constants::{em, physics, math};
//!
//! let wavelength = physics::C / 2.4e9;
//! let z0 = em::Z_0;
//! let tau = math::TAU;
//! ```

// =============================================================================
// Mathematics
// =============================================================================
pub mod math {
    //! Mathematical constants.

    // ---- Re-exports of the standard library's constants ---------------------
    pub use core::f64::consts::{
        E, FRAC_1_PI, FRAC_1_SQRT_2, FRAC_2_PI, FRAC_2_SQRT_PI, FRAC_PI_2, FRAC_PI_3, FRAC_PI_4,
        FRAC_PI_6, FRAC_PI_8, LN_2, LN_10, LOG2_10, LOG2_E, LOG10_2, LOG10_E, PI, SQRT_2, TAU,
    };

    // ---- Pi-related ---------------------------------------------------------
    /// 2π/3
    pub const TWO_PI_OVER_3: f64 = 2.0943951023931954;
    /// 3π/2
    pub const THREE_PI_OVER_2: f64 = 4.7123889803846897;
    /// π²
    pub const PI_SQUARED: f64 = 9.8696044010893586;
    /// √π
    pub const SQRT_PI: f64 = 1.7724538509055159;
    /// √(2π) = √τ  (Gaussian normalization)
    pub const SQRT_TAU: f64 = 2.5066282746310002;
    /// 1/√(2π)  (standard normal pdf peak value)
    pub const FRAC_1_SQRT_TAU: f64 = 0.3989422804014327;
    /// 1/√π
    pub const FRAC_1_SQRT_PI: f64 = 0.5641895835477563;
    /// ln(π)
    pub const LN_PI: f64 = 1.1447298858494002;
    /// ln(√(2π)) = ½ ln(2π)  (Gaussian log-likelihood term)
    pub const LN_SQRT_TAU: f64 = 0.9189385332046727;
    /// ln(2π)
    pub const LN_TAU: f64 = 1.8378770664093455;
    /// 4π  (solid angle of a sphere, steradians)
    pub const FOUR_PI: f64 = 12.566370614359172;

    // ---- Algebraic numbers --------------------------------------------------
    /// √3
    pub const SQRT_3: f64 = 1.7320508075688772;
    /// √5
    pub const SQRT_5: f64 = 2.2360679774997897;
    /// √6
    pub const SQRT_6: f64 = 2.4494897427831781;
    /// ∛2
    pub const CBRT_2: f64 = 1.2599210498948732;
    /// ∛3
    pub const CBRT_3: f64 = 1.4422495703074083;
    /// Golden ratio φ = (1 + √5)/2
    pub const PHI: f64 = 1.6180339887498949;
    /// 1/φ = φ − 1
    pub const FRAC_1_PHI: f64 = 0.6180339887498949;
    /// ln(φ)
    pub const LN_PHI: f64 = 0.48121182505960347;
    /// Silver ratio δ_S = 1 + √2
    pub const SILVER_RATIO: f64 = 2.4142135623730951;
    /// Plastic number (real root of x³ = x + 1)
    pub const PLASTIC_NUMBER: f64 = 1.3247179572447460;

    // ---- Special constants --------------------------------------------------
    /// Euler–Mascheroni constant γ
    pub const EULER_GAMMA: f64 = 0.57721566490153287;
    /// Apéry's constant ζ(3)
    pub const APERY: f64 = 1.2020569031595943;
    /// ζ(2) = π²/6
    pub const ZETA_2: f64 = 1.6449340668482264;
    /// ζ(4) = π⁴/90
    pub const ZETA_4: f64 = 1.0823232337111382;
    /// Catalan's constant G
    pub const CATALAN: f64 = 0.91596559417721901;
    /// Glaisher–Kinkelin constant A
    pub const GLAISHER: f64 = 1.2824271291006226;
    /// Khinchin's constant K₀
    pub const KHINCHIN: f64 = 2.6854520010653064;
    /// Omega constant Ω = W(1), solution of Ω·e^Ω = 1
    pub const OMEGA: f64 = 0.56714329040978387;
    /// Twin prime constant C₂
    pub const TWIN_PRIME: f64 = 0.66016181584686957;
    /// Feigenbaum constant δ (period-doubling ratio)
    pub const FEIGENBAUM_DELTA: f64 = 4.6692016091029907;
    /// Feigenbaum constant α (period-doubling scaling)
    pub const FEIGENBAUM_ALPHA: f64 = 2.5029078750958928;
    /// Wilbraham–Gibbs constant (2/π)·Si(π); Gibbs overshoot peak ratio
    pub const GIBBS: f64 = 1.1789797444721675;

    // ---- Statistics ---------------------------------------------------------
    /// 2√(2 ln 2): converts Gaussian σ → FWHM
    pub const SIGMA_TO_FWHM: f64 = 2.3548200450309493;
    /// 1/(2√(2 ln 2)): converts Gaussian FWHM → σ
    pub const FWHM_TO_SIGMA: f64 = 0.42466090014400953;
    /// Standard normal two-sided z-score for 90 % confidence
    pub const Z_90: f64 = 1.6448536269514722;
    /// Standard normal two-sided z-score for 95 % confidence
    pub const Z_95: f64 = 1.9599639845400543;
    /// Standard normal two-sided z-score for 99 % confidence
    pub const Z_99: f64 = 2.5758293035489004;
    /// Standard normal two-sided z-score for 99.9 % confidence
    pub const Z_999: f64 = 3.2905267314919255;
    /// Probability within ±1σ of a normal distribution
    pub const P_1_SIGMA: f64 = 0.6826894921370859;
    /// Probability within ±2σ of a normal distribution
    pub const P_2_SIGMA: f64 = 0.9544997361036416;
    /// Probability within ±3σ of a normal distribution
    pub const P_3_SIGMA: f64 = 0.9973002039367398;

    // ---- Floating-point helpers --------------------------------------------
    /// `f64` machine epsilon (2⁻⁵²)
    pub const EPSILON: f64 = f64::EPSILON;
    /// `f32` machine epsilon (2⁻²³) promoted to `f64`
    pub const EPSILON_F32: f64 = 1.1920928955078125e-7;
    /// Smallest positive normal `f64`
    pub const MIN_POSITIVE: f64 = f64::MIN_POSITIVE;
    /// Largest finite `f64`
    pub const MAX: f64 = f64::MAX;
}

// =============================================================================
// Physics
// =============================================================================
pub mod physics {
    //! Fundamental and derived physical constants (SI units, CODATA 2022).

    use super::math::{PI, TAU};

    // ---- Exact SI-defining constants ---------------------------------------
    /// Speed of light in vacuum, m/s (exact)
    pub const C: f64 = 299_792_458.0;
    /// Planck constant, J·s (exact)
    pub const H: f64 = 6.626_070_15e-34;
    /// Elementary charge, C (exact)
    pub const E_CHARGE: f64 = 1.602_176_634e-19;
    /// Boltzmann constant, J/K (exact)
    pub const K_B: f64 = 1.380_649e-23;
    /// Avogadro constant, 1/mol (exact)
    pub const N_A: f64 = 6.022_140_76e23;
    /// Luminous efficacy of 540 THz radiation, lm/W (exact)
    pub const K_CD: f64 = 683.0;
    /// Cesium hyperfine transition frequency, Hz (exact)
    pub const DELTA_NU_CS: f64 = 9_192_631_770.0;

    // ---- Derived from the above (exact in the SI) --------------------------
    /// Reduced Planck constant ħ = h/2π, J·s
    pub const HBAR: f64 = H / TAU;
    /// Molar gas constant R = N_A·k_B, J/(mol·K)
    pub const R_GAS: f64 = N_A * K_B;
    /// Faraday constant F = N_A·e, C/mol
    pub const FARADAY: f64 = N_A * E_CHARGE;
    /// Stefan–Boltzmann constant σ = 2π⁵k⁴/(15h³c²), W/(m²·K⁴)
    pub const SIGMA_SB: f64 =
        2.0 * PI * PI * PI * PI * PI * K_B * K_B * K_B * K_B / (15.0 * H * H * H * C * C);
    /// First radiation constant c₁ = 2πhc², W·m²
    pub const C1_RADIATION: f64 = TAU * H * C * C;
    /// Second radiation constant c₂ = hc/k, m·K
    pub const C2_RADIATION: f64 = H * C / K_B;
    /// Electron-volt, J
    pub const EV: f64 = E_CHARGE;

    // ---- Measured constants (CODATA 2022) ----------------------------------
    /// Newtonian constant of gravitation G, m³/(kg·s²)
    pub const G: f64 = 6.674_30e-11;
    /// Fine-structure constant α (dimensionless)
    pub const ALPHA: f64 = 7.297_352_564_3e-3;
    /// Inverse fine-structure constant 1/α
    pub const INV_ALPHA: f64 = 137.035_999_177;
    /// Electron mass, kg
    pub const M_E: f64 = 9.109_383_713_9e-31;
    /// Proton mass, kg
    pub const M_P: f64 = 1.672_621_925_95e-27;
    /// Neutron mass, kg
    pub const M_N: f64 = 1.674_927_500_56e-27;
    /// Atomic mass constant (unified atomic mass unit), kg
    pub const AMU: f64 = 1.660_539_068_92e-27;
    /// Proton-to-electron mass ratio
    pub const MP_OVER_ME: f64 = 1_836.152_673_426;
    /// Wien wavelength displacement constant b, m·K
    pub const WIEN_B: f64 = 2.897_771_955_185_173e-3;
    /// Wien frequency displacement constant, Hz/K
    pub const WIEN_B_FREQ: f64 = 5.878_925_757e10;

    // ---- Derived atomic-scale quantities -----------------------------------
    /// Bohr radius a₀ = ħ/(m_e·c·α), m
    pub const BOHR_RADIUS: f64 = HBAR / (M_E * C * ALPHA);
    /// Compton wavelength of the electron λ_C = h/(m_e·c), m
    pub const COMPTON_WAVELENGTH: f64 = H / (M_E * C);
    /// Reduced Compton wavelength ƛ_C = ħ/(m_e·c), m
    pub const COMPTON_WAVELENGTH_REDUCED: f64 = HBAR / (M_E * C);
    /// Classical electron radius r_e = α·ħ/(m_e·c), m
    pub const CLASSICAL_ELECTRON_RADIUS: f64 = ALPHA * HBAR / (M_E * C);
    /// Thomson scattering cross-section σ_T = (8π/3)·r_e², m²
    pub const THOMSON_CROSS_SECTION: f64 =
        (8.0 * PI / 3.0) * CLASSICAL_ELECTRON_RADIUS * CLASSICAL_ELECTRON_RADIUS;
    /// Rydberg constant R∞ = α²·m_e·c/(2h), 1/m
    pub const RYDBERG: f64 = ALPHA * ALPHA * M_E * C / (2.0 * H);
    /// Hartree energy E_h = m_e·c²·α², J
    pub const HARTREE: f64 = M_E * C * C * ALPHA * ALPHA;
    /// Rydberg energy = E_h / 2, J
    pub const RYDBERG_ENERGY: f64 = HARTREE / 2.0;
    /// Electron rest energy m_e·c², J
    pub const ELECTRON_REST_ENERGY: f64 = M_E * C * C;
    /// Proton rest energy m_p·c², J
    pub const PROTON_REST_ENERGY: f64 = M_P * C * C;

    // ---- Planck units (CODATA 2022) ----------------------------------------
    /// Planck length, m
    pub const PLANCK_LENGTH: f64 = 1.616_255e-35;
    /// Planck mass, kg
    pub const PLANCK_MASS: f64 = 2.176_434e-8;
    /// Planck time, s
    pub const PLANCK_TIME: f64 = 5.391_247e-44;
    /// Planck temperature, K
    pub const PLANCK_TEMPERATURE: f64 = 1.416_784e32;

    // ---- Thermodynamics & mechanics ----------------------------------------
    /// 0 °C in kelvin
    pub const ZERO_CELSIUS: f64 = 273.15;
    /// Standard temperature for chemistry (STP, IUPAC), K
    pub const T_STP: f64 = 273.15;
    /// Room temperature (20 °C), K
    pub const T_ROOM: f64 = 293.15;
    /// Triple point of water, K
    pub const T_TRIPLE_WATER: f64 = 273.16;
    /// Standard acceleration of gravity gₙ, m/s² (exact)
    pub const STANDARD_GRAVITY: f64 = 9.806_65;
    /// Standard atmosphere, Pa (exact)
    pub const STANDARD_ATMOSPHERE: f64 = 101_325.0;
    /// Standard-state pressure (IUPAC), Pa (exact)
    pub const STANDARD_PRESSURE: f64 = 100_000.0;
    /// Molar volume of an ideal gas at 273.15 K and 101.325 kPa, m³/mol
    pub const MOLAR_VOLUME_STP: f64 = R_GAS * 273.15 / 101_325.0;
    /// Specific gas constant of dry air, J/(kg·K)
    pub const R_SPECIFIC_AIR: f64 = 287.052_874;
    /// Specific heat of water at 15 °C, J/(kg·K) (approx.)
    pub const CP_WATER: f64 = 4_184.0;
    /// Density of water at 4 °C, kg/m³ (approx.)
    pub const RHO_WATER: f64 = 999.97;
    /// Density of dry air at 15 °C, 101.325 kPa, kg/m³ (ISA)
    pub const RHO_AIR_ISA: f64 = 1.225;
    /// Speed of sound in dry air at 20 °C, m/s
    pub const SPEED_OF_SOUND_AIR_20C: f64 = 343.21;
    /// Speed of sound in dry air at 0 °C, m/s
    pub const SPEED_OF_SOUND_AIR_0C: f64 = 331.3;
    /// Adiabatic index of air (diatomic ideal gas)
    pub const GAMMA_AIR: f64 = 1.4;

    // ---- Astronomy ----------------------------------------------------------
    /// Astronomical unit, m (exact, IAU 2012)
    pub const AU: f64 = 149_597_870_700.0;
    /// Julian year, s (exact)
    pub const JULIAN_YEAR: f64 = 31_557_600.0;
    /// Light-year, m (exact)
    pub const LIGHT_YEAR: f64 = 9_460_730_472_580_800.0;
    /// Parsec, m (exact: 648000/π AU)
    pub const PARSEC: f64 = 648_000.0 / PI * AU;
    /// Nominal solar mass parameter GM☉, m³/s² (IAU 2015 nominal)
    pub const GM_SUN: f64 = 1.327_124_4e20;
    /// Solar mass, kg
    pub const M_SUN: f64 = 1.988_41e30;
    /// Nominal solar radius, m (IAU 2015)
    pub const R_SUN: f64 = 6.957e8;
    /// Nominal solar luminosity, W (IAU 2015)
    pub const L_SUN: f64 = 3.828e26;
    /// Earth mass, kg
    pub const M_EARTH: f64 = 5.972_2e24;
    /// Earth equatorial radius (WGS-84), m
    pub const R_EARTH_EQUATORIAL: f64 = 6_378_137.0;
    /// Earth mean radius, m
    pub const R_EARTH_MEAN: f64 = 6_371_008.8;
    /// Moon mass, kg
    pub const M_MOON: f64 = 7.342e22;
    /// Mean Earth–Moon distance, m
    pub const EARTH_MOON_DISTANCE: f64 = 3.844e8;
    /// Sidereal day, s
    pub const SIDEREAL_DAY: f64 = 86_164.090_5;
    /// Solar constant (total solar irradiance at 1 AU), W/m²
    pub const SOLAR_CONSTANT: f64 = 1_361.0;
    /// Earth's angular rotation rate, rad/s (WGS-84)
    pub const EARTH_ANGULAR_VELOCITY: f64 = 7.292_115e-5;
    /// Hubble constant (≈ 70 km/s/Mpc) expressed in 1/s — a rough reference value
    pub const HUBBLE_APPROX: f64 = 70.0e3 / (1.0e6 * PARSEC);
}

// =============================================================================
// Electromagnetics
// =============================================================================
pub mod em {
    //! Electromagnetic constants and typical material properties.

    use super::math::FOUR_PI;
    use super::physics::{ALPHA, C, E_CHARGE, H, HBAR, M_E, M_P};

    // ---- Free space ---------------------------------------------------------
    /// Speed of light in vacuum, m/s (exact)
    pub const C0: f64 = C;
    /// Vacuum permeability μ₀, H/m (CODATA 2022; ≈ 4π × 10⁻⁷)
    pub const MU_0: f64 = 1.256_637_061_27e-6;
    /// Vacuum permittivity ε₀, F/m (CODATA 2022)
    pub const EPS_0: f64 = 8.854_187_818_8e-12;
    /// Characteristic impedance of free space Z₀ = μ₀·c, Ω
    pub const Z_0: f64 = MU_0 * C;
    /// Admittance of free space Y₀ = 1/Z₀, S
    pub const Y_0: f64 = 1.0 / Z_0;
    /// Coulomb constant kₑ = 1/(4π ε₀), N·m²/C²
    pub const K_E: f64 = 1.0 / (FOUR_PI * EPS_0);
    /// Classical μ₀ = 4π × 10⁻⁷ (pre-2019 SI exact definition), H/m
    pub const MU_0_CLASSIC: f64 = FOUR_PI * 1.0e-7;
    /// Classical ε₀ = 1/(μ₀_classic·c²), F/m
    pub const EPS_0_CLASSIC: f64 = 1.0 / (MU_0_CLASSIC * C * C);
    /// Elementary charge, C
    pub const E: f64 = E_CHARGE;

    // ---- Quantum electrical standards --------------------------------------
    /// Magnetic flux quantum Φ₀ = h/(2e), Wb
    pub const FLUX_QUANTUM: f64 = H / (2.0 * E_CHARGE);
    /// Josephson constant K_J = 2e/h, Hz/V
    pub const JOSEPHSON: f64 = 2.0 * E_CHARGE / H;
    /// von Klitzing constant R_K = h/e², Ω
    pub const VON_KLITZING: f64 = H / (E_CHARGE * E_CHARGE);
    /// Conductance quantum G₀ = 2e²/h, S
    pub const CONDUCTANCE_QUANTUM: f64 = 2.0 * E_CHARGE * E_CHARGE / H;
    /// Bohr magneton μ_B = e·ħ/(2m_e), J/T
    pub const BOHR_MAGNETON: f64 = E_CHARGE * HBAR / (2.0 * M_E);
    /// Nuclear magneton μ_N = e·ħ/(2m_p), J/T
    pub const NUCLEAR_MAGNETON: f64 = E_CHARGE * HBAR / (2.0 * M_P);
    /// Electron charge-to-mass ratio e/m_e, C/kg
    pub const E_OVER_ME: f64 = E_CHARGE / M_E;
    /// Fine-structure constant α = e²/(4π ε₀ ħ c)
    pub const FINE_STRUCTURE: f64 = ALPHA;
    /// Electron g-factor magnitude |g_e|
    pub const G_ELECTRON: f64 = 2.002_319_304_360_9;

    // ---- Standard impedances (ohms) ----------------------------------------
    /// Common RF system impedance, Ω
    pub const Z_50: f64 = 50.0;
    /// Common video/CATV impedance, Ω
    pub const Z_75: f64 = 75.0;
    /// Twisted-pair / balanced line impedance, Ω
    pub const Z_100: f64 = 100.0;
    /// Standard dipole radiation resistance (half-wave, in free space), Ω
    pub const HALF_WAVE_DIPOLE_RESISTANCE: f64 = 73.13;
    /// Half-wave dipole directivity (linear, ≈ 2.15 dBi)
    pub const HALF_WAVE_DIPOLE_DIRECTIVITY: f64 = 1.640;
    /// Short-dipole radiation directivity (linear, 1.5 ≈ 1.76 dBi)
    pub const SHORT_DIPOLE_DIRECTIVITY: f64 = 1.5;

    // ---- Unit conversions ---------------------------------------------------
    /// Gauss → tesla
    pub const GAUSS_TO_TESLA: f64 = 1.0e-4;
    /// Oersted → A/m (1000/4π)
    pub const OERSTED_TO_A_PER_M: f64 = 1000.0 / FOUR_PI;
    /// Maxwell → weber
    pub const MAXWELL_TO_WEBER: f64 = 1.0e-8;
    /// Gaussian-cgs statcoulomb per coulomb (≈ 10c/…): 1 C = this many statC
    pub const COULOMB_TO_STATCOULOMB: f64 = 2.997_924_58e9;
    /// Skin-depth prefactor for non-magnetic conductors: δ = SKIN_DEPTH_K · √(ρ/f),
    /// with ρ in Ω·m and f in Hz (K = 1/√(π·μ₀) ≈ 503.29)
    pub const SKIN_DEPTH_K: f64 = 503.292_121_2;

    // ---- Typical material properties at ≈ 20 °C ----------------------------
    /// Resistivity of copper (annealed), Ω·m
    pub const RHO_COPPER: f64 = 1.68e-8;
    /// Resistivity of silver, Ω·m
    pub const RHO_SILVER: f64 = 1.59e-8;
    /// Resistivity of gold, Ω·m
    pub const RHO_GOLD: f64 = 2.44e-8;
    /// Resistivity of aluminum, Ω·m
    pub const RHO_ALUMINUM: f64 = 2.65e-8;
    /// Resistivity of iron, Ω·m
    pub const RHO_IRON: f64 = 9.71e-8;
    /// Conductivity of the International Annealed Copper Standard, S/m (exact)
    pub const SIGMA_IACS: f64 = 5.8e7;
    /// Copper temperature coefficient of resistance, 1/K
    pub const ALPHA_COPPER: f64 = 3.93e-3;
    /// Relative permittivity of PTFE (Teflon)
    pub const EPS_R_PTFE: f64 = 2.1;
    /// Relative permittivity of polyethylene
    pub const EPS_R_POLYETHYLENE: f64 = 2.25;
    /// Relative permittivity of silicon dioxide
    pub const EPS_R_SIO2: f64 = 3.9;
    /// Typical relative permittivity of FR-4 at ~1 GHz
    pub const EPS_R_FR4: f64 = 4.4;
    /// Relative permittivity of crystalline silicon
    pub const EPS_R_SILICON: f64 = 11.7;
    /// Relative permittivity of water at 20 °C (static)
    pub const EPS_R_WATER: f64 = 80.1;
    /// Relative permittivity of air (≈ 1.000 59 at STP)
    pub const EPS_R_AIR: f64 = 1.000_59;
}

// =============================================================================
// Signal processing
// =============================================================================
pub mod dsp {
    //! Constants for DSP, audio, and RF noise/link-budget calculations.

    use super::math::{LN_10, LOG10_2, LOG10_E};
    use super::physics::K_B;

    // ---- Decibels / nepers --------------------------------------------------
    /// dB per neper (amplitude): 20·log₁₀(e)
    pub const DB_PER_NEPER: f64 = 20.0 * LOG10_E;
    /// nepers per dB (amplitude): ln(10)/20
    pub const NEPER_PER_DB: f64 = LN_10 / 20.0;
    /// Half-power point in dB: 10·log₁₀(½)
    pub const HALF_POWER_DB: f64 = -3.010_299_956_639_812;
    /// Half-power amplitude (−3 dB point of a magnitude response): 1/√2
    pub const HALF_POWER_AMPLITUDE: f64 = core::f64::consts::FRAC_1_SQRT_2;
    /// dB change per doubling of power: 10·log₁₀(2)
    pub const DB_PER_POWER_DOUBLING: f64 = 10.0 * LOG10_2;
    /// dB change per doubling of amplitude: 20·log₁₀(2)
    pub const DB_PER_AMPLITUDE_DOUBLING: f64 = 20.0 * LOG10_2;
    /// Butterworth Q of a 2nd-order section (maximally flat): 1/√2
    pub const Q_BUTTERWORTH: f64 = core::f64::consts::FRAC_1_SQRT_2;
    /// Critical-damping Q: 1/2
    pub const Q_CRITICAL: f64 = 0.5;
    /// Linkwitz–Riley 2nd-order Q (= 0.5)
    pub const Q_LINKWITZ_RILEY: f64 = 0.5;
    /// 2nd-order Bessel Q: 1/√3
    pub const Q_BESSEL: f64 = 0.577_350_269_189_625_8;

    // ---- Quantization -------------------------------------------------------
    /// Ideal quantizer SNR slope: 20·log₁₀(2) dB per bit
    pub const SQNR_DB_PER_BIT: f64 = 6.020_599_913_279_624;
    /// Ideal sine-wave SQNR offset: 10·log₁₀(1.5) dB (SQNR = 6.02·N + 1.76)
    pub const SQNR_OFFSET_DB: f64 = 1.760_912_590_556_812_4;
    /// Q7 fixed-point scale (2⁷)
    pub const Q7_SCALE: f64 = 128.0;
    /// Q15 fixed-point scale (2¹⁵)
    pub const Q15_SCALE: f64 = 32_768.0;
    /// Q31 fixed-point scale (2³¹)
    pub const Q31_SCALE: f64 = 2_147_483_648.0;

    // ---- Standard sample rates (Hz) ----------------------------------------
    pub const FS_8K: f64 = 8_000.0;
    pub const FS_11K025: f64 = 11_025.0;
    pub const FS_16K: f64 = 16_000.0;
    pub const FS_22K05: f64 = 22_050.0;
    pub const FS_32K: f64 = 32_000.0;
    /// CD audio sample rate
    pub const FS_44K1: f64 = 44_100.0;
    /// Professional audio / DVD sample rate
    pub const FS_48K: f64 = 48_000.0;
    pub const FS_88K2: f64 = 88_200.0;
    pub const FS_96K: f64 = 96_000.0;
    pub const FS_176K4: f64 = 176_400.0;
    pub const FS_192K: f64 = 192_000.0;

    // ---- Audio / music ------------------------------------------------------
    /// Concert pitch A4, Hz
    pub const A4_HZ: f64 = 440.0;
    /// MIDI note number for A4
    pub const MIDI_A4: f64 = 69.0;
    /// Equal-temperament semitone frequency ratio: 2^(1/12)
    pub const SEMITONE_RATIO: f64 = 1.059_463_094_359_295_3;
    /// Equal-temperament cent frequency ratio: 2^(1/1200)
    pub const CENT_RATIO: f64 = 1.000_577_789_506_554_8;
    /// Lower limit of nominal human hearing, Hz
    pub const HEARING_MIN_HZ: f64 = 20.0;
    /// Upper limit of nominal human hearing, Hz
    pub const HEARING_MAX_HZ: f64 = 20_000.0;
    /// Reference sound pressure in air (0 dB SPL), Pa
    pub const P_REF_AIR: f64 = 20.0e-6;
    /// Reference sound intensity (0 dB SPL), W/m²
    pub const I_REF_AIR: f64 = 1.0e-12;
    /// Reference sound power, W
    pub const W_REF_AIR: f64 = 1.0e-12;
    /// Reference sound pressure in water, Pa
    pub const P_REF_WATER: f64 = 1.0e-6;

    // ---- RF noise -----------------------------------------------------------
    /// IEEE standard reference noise temperature T₀, K
    pub const T0_NOISE: f64 = 290.0;
    /// Thermal noise power density at T₀: k·T₀, W/Hz (≈ −174 dBm/Hz)
    pub const KT0: f64 = K_B * T0_NOISE;
    /// Thermal noise floor at T₀ in dBm/Hz
    pub const KT0_DBM_PER_HZ: f64 = -173.975;
    /// Thermal noise floor at 300 K in dBm/Hz (≈ −173.83)
    pub const KT300_DBM_PER_HZ: f64 = -173.83;
    /// Watts in 0 dBm
    pub const DBM_REF_WATTS: f64 = 1.0e-3;

    // ---- Window functions: cosine-sum coefficients -------------------------
    // w[n] = a0 − a1·cos(2πn/N) + a2·cos(4πn/N) − a3·cos(6πn/N) + a4·cos(8πn/N)

    /// Hann window a₀
    pub const HANN_A0: f64 = 0.5;
    /// Hann window a₁
    pub const HANN_A1: f64 = 0.5;

    /// Hamming window a₀ (classic)
    pub const HAMMING_A0: f64 = 0.54;
    /// Hamming window a₁ (classic)
    pub const HAMMING_A1: f64 = 0.46;
    /// Hamming window a₀ (exact, zeroes the first side-lobe): 25/46
    pub const HAMMING_A0_EXACT: f64 = 25.0 / 46.0;
    /// Hamming window a₁ (exact): 21/46
    pub const HAMMING_A1_EXACT: f64 = 21.0 / 46.0;

    /// Blackman window a₀ (classic)
    pub const BLACKMAN_A0: f64 = 0.42;
    /// Blackman window a₁ (classic)
    pub const BLACKMAN_A1: f64 = 0.5;
    /// Blackman window a₂ (classic)
    pub const BLACKMAN_A2: f64 = 0.08;
    /// Blackman window a₀ (exact): 7938/18608
    pub const BLACKMAN_A0_EXACT: f64 = 7_938.0 / 18_608.0;
    /// Blackman window a₁ (exact): 9240/18608
    pub const BLACKMAN_A1_EXACT: f64 = 9_240.0 / 18_608.0;
    /// Blackman window a₂ (exact): 1430/18608
    pub const BLACKMAN_A2_EXACT: f64 = 1_430.0 / 18_608.0;

    /// 4-term Blackman–Harris window coefficients (−92 dB sidelobes)
    pub const BLACKMAN_HARRIS_A: [f64; 4] = [0.35875, 0.48829, 0.14128, 0.01168];
    /// 4-term Nuttall window coefficients (continuous first derivative)
    pub const NUTTALL_A: [f64; 4] = [0.3635819, 0.4891775, 0.1365995, 0.0106411];
    /// 5-term flat-top window coefficients (SciPy / MATLAB convention)
    pub const FLATTOP_A: [f64; 5] = [
        0.21557895,
        0.41663158,
        0.277263158,
        0.083578947,
        0.006947368,
    ];

    // ---- Window figures of merit -------------------------------------------
    /// Coherent gain (mean of window) — rectangular
    pub const CG_RECT: f64 = 1.0;
    /// Coherent gain — Hann
    pub const CG_HANN: f64 = 0.5;
    /// Coherent gain — Hamming
    pub const CG_HAMMING: f64 = 0.54;
    /// Coherent gain — Blackman
    pub const CG_BLACKMAN: f64 = 0.42;
    /// Noise-equivalent bandwidth in bins — rectangular
    pub const ENBW_RECT: f64 = 1.0;
    /// Noise-equivalent bandwidth in bins — Hann
    pub const ENBW_HANN: f64 = 1.5;
    /// Noise-equivalent bandwidth in bins — Hamming
    pub const ENBW_HAMMING: f64 = 1.3628;
    /// Noise-equivalent bandwidth in bins — Blackman
    pub const ENBW_BLACKMAN: f64 = 1.7268;
    /// Noise-equivalent bandwidth in bins — 4-term Blackman–Harris
    pub const ENBW_BLACKMAN_HARRIS: f64 = 2.0044;
    /// Kaiser window β giving ≈ 60 dB stop-band attenuation (≈ 5.65)
    pub const KAISER_BETA_60DB: f64 = 5.653;
    /// Kaiser window β giving ≈ 80 dB stop-band attenuation (≈ 7.86)
    pub const KAISER_BETA_80DB: f64 = 7.857;
    /// Kaiser window β commonly used for audio resampling
    pub const KAISER_BETA_AUDIO: f64 = 8.6;
}

// =============================================================================
// Machine learning
// =============================================================================
pub mod ml {
    //! Constants and default hyper-parameters common in machine learning.
    //!
    //! Optimizer defaults follow the original papers / PyTorch conventions.

    // ---- Activation functions ----------------------------------------------
    /// √(2/π): GELU tanh-approximation prefactor
    pub const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
    /// GELU tanh-approximation cubic coefficient: 0.044715
    pub const GELU_TANH_COEFF: f64 = 0.044_715;
    /// 1/√2 for exact GELU via erf: x·½·(1 + erf(x/√2))
    pub const FRAC_1_SQRT_2: f64 = core::f64::consts::FRAC_1_SQRT_2;
    /// SELU α (Klambauer et al. 2017)
    pub const SELU_ALPHA: f64 = 1.673_263_242_354_377_3;
    /// SELU scale λ (Klambauer et al. 2017)
    pub const SELU_SCALE: f64 = 1.050_700_987_355_480_5;
    /// ELU default α
    pub const ELU_ALPHA: f64 = 1.0;
    /// Leaky-ReLU default negative slope (PyTorch)
    pub const LEAKY_RELU_SLOPE: f64 = 0.01;
    /// Leaky-ReLU slope commonly used in GANs
    pub const LEAKY_RELU_SLOPE_GAN: f64 = 0.2;
    /// Maximum of the sigmoid derivative σ'(0) = 1/4
    pub const SIGMOID_DERIV_MAX: f64 = 0.25;
    /// Softplus linearisation threshold (PyTorch default)
    pub const SOFTPLUS_THRESHOLD: f64 = 20.0;

    // ---- Weight-init gains --------------------------------------------------
    /// He/Kaiming gain for ReLU: √2
    pub const GAIN_RELU: f64 = core::f64::consts::SQRT_2;
    /// Recommended gain for tanh: 5/3
    pub const GAIN_TANH: f64 = 5.0 / 3.0;
    /// Gain for linear / sigmoid layers
    pub const GAIN_LINEAR: f64 = 1.0;
    /// Gain for SELU (LeCun normal init): 3/4
    pub const GAIN_SELU: f64 = 0.75;
    /// Truncated-normal init standard deviation used by BERT / GPT-2
    pub const INIT_STD_TRANSFORMER: f64 = 0.02;

    // ---- Optimizer defaults -------------------------------------------------
    /// Adam β₁
    pub const ADAM_BETA1: f64 = 0.9;
    /// Adam β₂
    pub const ADAM_BETA2: f64 = 0.999;
    /// Adam ε (original paper)
    pub const ADAM_EPS: f64 = 1.0e-8;
    /// Adam default learning rate
    pub const ADAM_LR: f64 = 1.0e-3;
    /// "Karpathy constant": commonly good Adam learning rate
    pub const ADAM_LR_KARPATHY: f64 = 3.0e-4;
    /// AdamW default weight decay (PyTorch)
    pub const ADAMW_WEIGHT_DECAY: f64 = 1.0e-2;
    /// Adam β₂ for large-LM training (GPT-3 / LLaMA style)
    pub const ADAM_BETA2_LLM: f64 = 0.95;
    /// Adam β₂ and ε used in the original Transformer paper
    pub const ADAM_BETA2_TRANSFORMER: f64 = 0.98;
    pub const ADAM_EPS_TRANSFORMER: f64 = 1.0e-9;
    /// Original Transformer LR warm-up steps
    pub const TRANSFORMER_WARMUP_STEPS: f64 = 4_000.0;
    /// SGD momentum (classic)
    pub const SGD_MOMENTUM: f64 = 0.9;
    /// RMSprop smoothing constant α (PyTorch)
    pub const RMSPROP_ALPHA: f64 = 0.99;
    /// RMSprop ρ (Hinton / Keras)
    pub const RMSPROP_RHO: f64 = 0.9;
    /// RMSprop ε
    pub const RMSPROP_EPS: f64 = 1.0e-8;
    /// Adagrad ε
    pub const ADAGRAD_EPS: f64 = 1.0e-10;
    /// Default gradient-clipping norm
    pub const GRAD_CLIP_NORM: f64 = 1.0;

    // ---- Normalization / regularization ------------------------------------
    /// BatchNorm ε (PyTorch / Keras ≈ 1e-5 / 1e-3)
    pub const BATCHNORM_EPS: f64 = 1.0e-5;
    /// BatchNorm running-stat momentum (PyTorch convention)
    pub const BATCHNORM_MOMENTUM: f64 = 0.1;
    /// LayerNorm ε (PyTorch)
    pub const LAYERNORM_EPS: f64 = 1.0e-5;
    /// LayerNorm ε (original Transformer / BERT)
    pub const LAYERNORM_EPS_TRANSFORMER: f64 = 1.0e-12;
    /// RMSNorm ε (LLaMA)
    pub const RMSNORM_EPS: f64 = 1.0e-6;
    /// Typical dropout rate for transformers
    pub const DROPOUT_TRANSFORMER: f64 = 0.1;
    /// Typical dropout rate for fully-connected layers (Srivastava et al.)
    pub const DROPOUT_FC: f64 = 0.5;
    /// Typical label-smoothing factor
    pub const LABEL_SMOOTHING: f64 = 0.1;

    // ---- Numerical safety ---------------------------------------------------
    /// Keras backend epsilon for clipping probabilities
    pub const PROB_CLIP_EPS: f64 = 1.0e-7;
    /// ε for cosine similarity / L2 normalization (PyTorch)
    pub const NORMALIZE_EPS: f64 = 1.0e-12;
    /// log(ε) floor used to avoid −∞ in log-probabilities
    pub const LOG_FLOOR: f64 = -36.841_361_487_904_73; // ln(1e-16)
    /// Large negative number used for attention masking
    pub const MASK_NEG_INF: f64 = -1.0e9;

    // ---- Transformers / embeddings -----------------------------------------
    /// Sinusoidal positional-encoding base (Vaswani et al.)
    pub const POS_ENC_BASE: f64 = 10_000.0;
    /// RoPE base θ (original RoFormer / LLaMA 1-2)
    pub const ROPE_BASE: f64 = 10_000.0;
    /// RoPE base θ (LLaMA 3)
    pub const ROPE_BASE_LLAMA3: f64 = 500_000.0;

    // ---- Information theory -------------------------------------------------
    /// Convert nats → bits: log₂(e)
    pub const NATS_TO_BITS: f64 = core::f64::consts::LOG2_E;
    /// Convert bits → nats: ln(2)
    pub const BITS_TO_NATS: f64 = core::f64::consts::LN_2;
    /// Convert nats → hartleys (bans): log₁₀(e)
    pub const NATS_TO_HARTLEYS: f64 = core::f64::consts::LOG10_E;
    /// Differential entropy of N(0,1) in nats: ½·ln(2πe)
    pub const STD_NORMAL_ENTROPY: f64 = 1.418_938_533_204_672_7;

    // ---- Dataset statistics (per channel, inputs scaled to [0, 1]) ---------
    /// ImageNet RGB mean
    pub const IMAGENET_MEAN: [f64; 3] = [0.485, 0.456, 0.406];
    /// ImageNet RGB standard deviation
    pub const IMAGENET_STD: [f64; 3] = [0.229, 0.224, 0.225];
    /// CIFAR-10 RGB mean
    pub const CIFAR10_MEAN: [f64; 3] = [0.4914, 0.4822, 0.4465];
    /// CIFAR-10 RGB standard deviation
    pub const CIFAR10_STD: [f64; 3] = [0.2470, 0.2435, 0.2616];
    /// CIFAR-100 RGB mean
    pub const CIFAR100_MEAN: [f64; 3] = [0.5071, 0.4865, 0.4409];
    /// CIFAR-100 RGB standard deviation
    pub const CIFAR100_STD: [f64; 3] = [0.2673, 0.2564, 0.2762];
    /// MNIST mean
    pub const MNIST_MEAN: f64 = 0.1307;
    /// MNIST standard deviation
    pub const MNIST_STD: f64 = 0.3081;

    // ---- Color → grayscale luma weights ------------------------------------
    /// ITU-R BT.601 luma weights (R, G, B)
    pub const LUMA_BT601: [f64; 3] = [0.299, 0.587, 0.114];
    /// ITU-R BT.709 luma weights (R, G, B)
    pub const LUMA_BT709: [f64; 3] = [0.2126, 0.7152, 0.0722];
    /// ITU-R BT.2020 luma weights (R, G, B)
    pub const LUMA_BT2020: [f64; 3] = [0.2627, 0.6780, 0.0593];
}

// =============================================================================
// Unit conversions
// =============================================================================
pub mod units {
    //! Conversion factors to SI. Multiply a value in the named unit by the
    //! constant to obtain the SI value (e.g. `5.0 * INCH` → metres).

    use super::math::PI;

    // ---- Angle --------------------------------------------------------------
    /// Degrees → radians
    pub const DEG_TO_RAD: f64 = PI / 180.0;
    /// Radians → degrees
    pub const RAD_TO_DEG: f64 = 180.0 / PI;
    /// Arcminutes → radians
    pub const ARCMIN_TO_RAD: f64 = PI / 10_800.0;
    /// Arcseconds → radians
    pub const ARCSEC_TO_RAD: f64 = PI / 648_000.0;
    /// Gradians → radians
    pub const GRAD_TO_RAD: f64 = PI / 200.0;
    /// Turns (revolutions) → radians
    pub const TURN_TO_RAD: f64 = 2.0 * PI;
    /// Hz → rad/s
    pub const HZ_TO_RAD_S: f64 = 2.0 * PI;
    /// RPM → rad/s
    pub const RPM_TO_RAD_S: f64 = 2.0 * PI / 60.0;
    /// Steradians in a full sphere
    pub const SPHERE_SR: f64 = 4.0 * PI;

    // ---- Length (m) ---------------------------------------------------------
    pub const INCH: f64 = 0.0254;
    pub const FOOT: f64 = 0.3048;
    pub const YARD: f64 = 0.9144;
    pub const MILE: f64 = 1_609.344;
    pub const NAUTICAL_MILE: f64 = 1_852.0;
    pub const MIL: f64 = 2.54e-5;
    pub const ANGSTROM: f64 = 1.0e-10;

    // ---- Mass (kg) ----------------------------------------------------------
    pub const POUND: f64 = 0.453_592_37;
    pub const OUNCE: f64 = 0.028_349_523_125;
    pub const TONNE: f64 = 1_000.0;

    // ---- Time (s) -----------------------------------------------------------
    pub const MINUTE: f64 = 60.0;
    pub const HOUR: f64 = 3_600.0;
    pub const DAY: f64 = 86_400.0;
    pub const WEEK: f64 = 604_800.0;

    // ---- Pressure (Pa) ------------------------------------------------------
    pub const ATM: f64 = 101_325.0;
    pub const BAR: f64 = 100_000.0;
    pub const MMHG: f64 = 133.322_387_415;
    pub const TORR: f64 = 101_325.0 / 760.0;
    pub const PSI: f64 = 6_894.757_293_168_361;

    // ---- Energy / power -----------------------------------------------------
    /// Thermochemical calorie, J
    pub const CALORIE: f64 = 4.184;
    /// International Table BTU, J
    pub const BTU: f64 = 1_055.055_852_62;
    /// Kilowatt-hour, J
    pub const KWH: f64 = 3.6e6;
    /// Electron-volt, J
    pub const ELECTRON_VOLT: f64 = 1.602_176_634e-19;
    /// Mechanical horsepower, W
    pub const HORSEPOWER: f64 = 745.699_871_582_270_2;

    // ---- Volume (m³) --------------------------------------------------------
    pub const LITER: f64 = 1.0e-3;
    pub const GALLON_US: f64 = 3.785_411_784e-3;

    // ---- Force --------------------------------------------------------------
    /// Pound-force, N
    pub const LBF: f64 = 4.448_221_615_260_5;
    /// Kilogram-force, N
    pub const KGF: f64 = 9.806_65;

    // ---- Temperature offsets -----------------------------------------------
    /// Celsius → kelvin offset
    pub const CELSIUS_OFFSET: f64 = 273.15;
    /// Fahrenheit → Rankine offset
    pub const FAHRENHEIT_RANKINE_OFFSET: f64 = 459.67;
}
