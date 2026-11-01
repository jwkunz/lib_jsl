# Electromagnetics crate design

This document records the agreed structure of `lib_jsl_electromagnetics`. **Status: implemented**
(all steps of §11; see the demonstrations below).
The field containers are the backbone: antennas produce them, and the overlap integral, array
signal models and beamformer solvers consume them.

## 1. Conventions

| Item | Choice |
|------|--------|
| Units | SI throughout |
| Numerics | `f64` and `num_complex::Complex<f64>` |
| Time convention | Configurable per harmonic field: `TimeConvention::PlusJ` (`e^{+jωt}`, default) or `TimeConvention::MinusI` (`e^{-iωt}`). Converting between them is complex conjugation. Operations on two harmonic fields require the same convention (or convert explicitly). |
| Consequences of the default | Outgoing waves `e^{-jkr}`; far-field steering phase `e^{+jk r̂·p}` |
| Phasor amplitude | Peak (not RMS); time-averaged quantities carry the factor ½ (e.g. `S = ½ E × H*`) |
| Spherical angles | Physics convention: θ from +z, φ azimuth from +x |
| Far fields | Stored with the `e^{-jkr}/r` factor removed (radiation-intensity normalisation), no radial component |
| Gain, directivity | Linear internally; dBi helpers for display |

Physical constants (μ₀, ε₀, η₀, c, k_B) come from `lib_jsl_core::constants`, adding any that are
missing there.

## 2. Coordinates and polarization bases (`coordinates/`)

- **Coordinate systems:** Cartesian `(x, y, z)`, spherical `(r, θ, φ)` and cylindrical `(ρ, φ, z)`.
  Each provides point conversion, scale factors `(h₁, h₂, h₃)` and the orthonormal local frame
  `(û₁, û₂, û₃)` at a point.
- **Polarization bases**, available in every coordinate system:
  - *Linear*: the system's own unit vectors (x̂ŷẑ, r̂θ̂φ̂, ρ̂φ̂ẑ).
  - *Circular*: right/left-hand circular components formed from the two unit vectors transverse
    to the system's reference direction (ẑ for Cartesian, r̂ for spherical, ρ̂ for cylindrical),
    plus the longitudinal component. Handedness follows the IEEE convention and is consistent
    with the field's time convention.
- Every basis is a position-dependent (possibly complex) 3×3 map to Cartesian components, so a
  basis change is always "to Cartesian, then to the target basis" at the sample's position.
- **Rigid transforms:** translation and rotation (reusing `lib_jsl_geometry`'s `f64`
  `Rotation3`/`Quaternion`). A rotation acts on both sample positions and field vectors.

## 3. Domains and quadrature (`fields/domain`)

Structured grids in their native coordinates; uniform or non-uniform axes.

| Kind | Domains |
|------|---------|
| Surface | plane (x, y at z₀), sphere (θ, φ at radius R, including the far-field sphere at ∞), cylinder (φ, z at ρ₀) |
| Volume | Cartesian box, spherical shell (r, θ, φ), cylindrical (ρ, φ, z) |

Each domain owns:
- sample positions,
- **quadrature weights with the Jacobian included** (`sin θ dθ dφ`, `ρ dρ dφ dz`, ...), so no
  integral in the crate hand-rolls a Jacobian;
- periodic axes (φ wraps) and pole handling on spheres;
- the neighbourhood structure used by interpolation and finite differences.

## 4. Field containers (`fields/field_container`)

A field is `{ domain, basis, quantity, values }`:
- **Value kinds:** harmonic (complex phasors at a frequency, with a time convention) and
  time-varying (real samples on a uniform time axis, i.e. a space × time grid).
- **Rank:** scalar (potentials, temperature or intensity maps, divergences) or vector (three
  components in the field's polarization basis).
- **Quantity tag:** `E`, `H`, `D`, `B`, `J`, `M`, vector potential `A`, scalar potential `Φ`,
  brightness/noise temperature, or generic. The tag carries units and lets physics operations
  check their inputs.

Operations:
- **Basis and coordinate conversion:** `to_basis`, conversion of positions and vectors between the
  coordinate systems.
- **Sampling:** `sample_at(point)` and `resample_to(domain)` by nearest, linear or cubic
  interpolation. Vector fields are interpolated component-wise *in the Cartesian basis* (θ̂, φ̂
  components are not smooth across the poles), with periodic and pole-aware stencils.
- **Rigid transforms:** `translate`, `rotate`. Translating a far field applies the phase ramp
  `e^{jk r̂·d}` (sign per time convention).
- **Integration:** `integrate`, and inner products over the domain using its quadrature weights.
- **Vector calculus** on volume domains in native curvilinear coordinates (via scale factors):
  gradient, divergence, curl, scalar and vector Laplacian; `∂/∂t` for time-varying fields.
- **Physics operations (quantity-checked):** Poynting vector (`½ E × H*` harmonic, `E × H`
  instantaneous), radiated power through a surface, wave impedance, fields from potentials
  (`E = -∇Φ - jωA`, `B = ∇ × A`, sign per time convention).

## 5. Antenna models (`antenna_models/`)

A common trait: analytic far field `(E_θ, E_φ)` normalised per unit input (convention to be fixed
at implementation: per unit terminal current, with power-normalised helpers), plus derived
radiated power, directivity, gain, beamwidth and pattern cuts, and the ability to fill a far-field
container on any sphere domain. Optional exact near fields and circuit properties.

| Model | Content |
|-------|---------|
| Hertzian dipole | exact near + far field; D = 1.5, R_rad = 80π²(l/λ)² |
| Magnetic (small) loop | exact near + far field; R_rad ∝ (ka)⁴ |
| Half-wave dipole | far field; D ≈ 1.64, R_rad ≈ 73.1 Ω |
| Dish | circular-aperture integral with optional taper (Bessel functions); D = η(πD/λ)² |
| Patch | transmission-line / cavity model: resonant frequency, edge impedance, two-slot far field |

Analytic models, not full-wave solvers.

## 6. Arrays and signal models (`arrays/`, `signal_models/`) — beamformer part A

- **`SensorArray`:** elements with position, orientation and their own far-field container
  (embedded element patterns; mutual coupling enters only through the supplied patterns).
- **Response:** the N×2 matrix `A(Ω)` of θ̂/φ̂ element responses including the geometric phase.
- **Scene:**
  - discrete point sources: direction, power, polarization (Jones vector or 2×2 coherency);
  - distributed sources: a far-field container of brightness / noise temperature with a 2×2
    coherency per direction (unpolarized = ½ I);
  - receiver noise per element.
- **Covariance:** `R = Σ P·a aᴴ + ∫ A(Ω) C(Ω) A(Ω)ᴴ dΩ + σ² I`, the integral using the sphere
  domain's weights and temperature maps scaled by `k_B·B`.

## 7. Algorithms (`algorithms/`) — beamformer part B and integrals

- **Overlap integral:** `|∫ E₁·E₂* dA|² / (∫|E₁|² dA ∫|E₂|² dA)`. Both fields are resampled onto a
  common domain, compared in a common (Cartesian) basis, and integrated with that domain's
  quadrature weights.
- **CFM beamformer:** conjugate field match, the matched filter for a harmonic array excitation,
  `w ∝ a(Ω₀)` for a desired field / direction.
- **Max-SNR beamformer:** the dominant generalised eigenvector of `R_s w = λ R_n w`.
- **LCMV beamformer:** `w = R⁻¹C (Cᴴ R⁻¹ C)⁻¹ f`, with MVDR as the single-constraint case.
- Outputs: weights, the resulting array pattern as a far-field container, SNR and G/T.

## 8. Module layout

```
electromagnetics/src/
  conventions.rs            time convention, units, dB helpers
  coordinates/              systems, polarization bases, rigid transforms
  fields/
    domain/                 surface and volume grids with quadrature weights
    field_container.rs      scalar/vector, harmonic/time-varying, quantity tags
    coordinate_conversion.rs
    interpolation.rs
    vector_calculus.rs
    physics.rs              Poynting, power, impedance, potentials
  antenna_models/
  arrays/                   sensor arrays and responses
  signal_models/            point and distributed scenes, covariance
  algorithms/               overlap integral, CFM, max-SNR, LCMV
```

## 9. Dependencies

`lib_jsl_core`, `lib_jsl_geometry` (rotations), `lib_jsl_special_functions` (Bessel functions),
`num-complex`, `num-traits`, `ndarray`, `ndarray-linalg` (eigen solves); `lib_jsl_utils` as a dev-dependency for demo
plots (a small polar/dB plot addition is planned there). Interpolation is implemented in-crate,
since periodic and pole-aware stencils on these domains do not fit the interpolation crate.

## 10. Validation

- Closed-form antenna values above; reciprocity and power conservation (radiated power from the
  far-field integral equals the Poynting flux through an enclosing sphere).
- Coordinate and basis round trips; interpolation exactness on low-order polynomials; quadrature
  exactness (sphere area, spherical harmonics orthogonality).
- Vector-calculus identities: `∇·∇×F = 0`, `∇×∇f = 0`, analytic test fields in every coordinate
  system.
- Beamformers: CFM equals the max-SNR solution under white noise; LCMV meets its constraints
  exactly; MVDR nulls a point interferer.

## 11. Implementation order

1. Conventions, coordinates, polarization bases, rigid transforms
2. Domains and quadrature
3. Field container and coordinate/basis conversion
4. Interpolation
5. Vector calculus and physics operations
6. Antenna models
7. Overlap integral
8. Arrays and signal models
9. Beamformer solvers (CFM, max-SNR, LCMV)
10. Demonstration examples

## 12. Demonstrations

`cargo run --release -p lib_jsl_electromagnetics --example <name>`:

- `antenna_patterns`: directivity and radiation resistance of every model, patch design and inset
  feed, dish beamwidth, and polar dB pattern cuts.
- `fields_demo`: Poynting-flux conservation of an exact near field, Maxwell's equations checked with
  the containers' vector calculus, overlap integrals across grids and bases, and rigid transforms.
- `beamforming_demo`: a 16-element dipole array with jammers, a warm ground and a cold sky;
  CFM, max-SNR, MVDR and LCMV compared by SINR, jammer suppression and pattern.

## 13. Notes from implementation

- Array elements are either *analytic* (an antenna model plus orientation, evaluated exactly) or
  *sampled* (a pattern container, interpolated in Cartesian components). Idealised isotropic
  elements with a fixed polarization are discontinuous at the poles (hairy-ball theorem) and must
  be analytic.
- LCMV constraints are stated as responses, `wᴴ c_k = f_k` (equivalently `Cᴴ w = f*`).
- Brightness scenes use per-polarization brightness temperature: unpolarized emission gives
  `R = k_B B / (4π) ∫ T A Aᴴ dΩ`, so a lossless antenna in a uniform sky sees `k_B T B`.
- Vector calculus is implemented for volume domains, and surface operators (gradient, divergence,
  curl, Laplace-Beltrami) for plane, sphere and cylinder domains.

## 14. Extensions beyond the original design

- `networks/`: transmission lines (impedance transformation, standing waves, stub matching),
  two-port S/Z/Y/ABCD/T conversions and Smith chart geometry (`e^{+jωt}` only).
- `propagation/`: Friis and empirical path loss, two-ray and multipath channels, atmospheric
  attenuation (simplified ITU-R P.676 and P.840) and the radar equation.
- `fields/spherical_waves.rs`: spherical wave expansions; a far-field pattern's TE/TM amplitudes
  reproduce the field anywhere outside the minimum sphere.
- `algorithms/`: direction-of-arrival estimation (Bartlett, Capon, MUSIC, ESPRIT), planar and
  spherical near-field to far-field transformation, Hallén method of moments for thin wires, and
  1D and 2D TMz FDTD solvers.
