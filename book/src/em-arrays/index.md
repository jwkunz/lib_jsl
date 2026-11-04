# Electromagnetics: fields, antennas, arrays and beamforming

This chapter and the next cover `lib_jsl_electromagnetics`. This one follows a signal from the field it lives in to the antenna that radiates or receives it, to the array that combines many antennas, and to the processing that steers beams and finds directions. The [field containers](fields.md) are the backbone. Antennas produce them, arrays sample them, and beamformers and overlap integrals consume them. All of it rests on one set of
conventions for units, phasors and coordinates, recorded in the crate's `DESIGN.md`.

## The central questions

- **Which convention?** Every sign in electromagnetics depends on whether phasors rotate as $e^{+j\omega t}$ or $e^{-i\omega t}$. The library makes the choice explicit and checks it.
- **Near or far?** Close to an antenna the fields are reactive and three-dimensional; far away they are transverse spherical waves described by a pattern.
- **One antenna or many?** A single antenna's pattern is fixed; an array's beam is chosen in software, and can be optimised against interference.
- **Known or estimated?** Optimal beamformers and direction finders assume the covariance; real systems estimate it from snapshots, at a predictable cost.

## Testing

Everything here is checked against closed forms:
- antenna directivities and radiation resistances;
- Poynting flux conservation of exact near fields;
- the fourth-order convergence of the vector calculus;
- overlap integrals of Gaussian beams;
- the Reed-Mallett-Brennan law for sample-matrix inversion.

Writing this chapter corrected two documentation details: a demonstration compared the dish beamwidth with 1.02 λ/D instead of the exact 1.029 λ/D, and the crate's description omitted most of its contents.

## In the library

All of these are in `lib_jsl_electromagnetics` ([API reference](../../api/lib_jsl_electromagnetics/index.html)). Run the demonstrations with `cargo run --release -p lib_jsl_electromagnetics --example <name>`.

| You need | Use | Entry |
|---|---|---|
| Fields on surfaces and volumes; integrals, interpolation, transforms, derivatives; Poynting flux | `Domain`, `HarmonicField`, `TimeField`, `physics`, `overlap_integral` | [Fields on grids](fields.md) |
| The pattern, directivity or radiation resistance of a classic antenna | `HertzianDipole`, `MagneticLoop`, `Dipole`, `HalfWaveDipole`, `Dish`, `PatchAntenna` | [Antenna models](antennas.md) |
| An array's response to a scene; beam weights against interference | `SensorArray`, `Scene`, `PointSource`, `BrightnessScene`, `mvdr_beamformer`, `lcmv_beamformer`, `max_snr_beamformer` | [Arrays and beamformers](beamforming.md) |
| The directions of arriving signals | `bartlett_spectrum`, `capon_spectrum`, `music_spectrum`, `esprit_ula` | [Direction finding](direction-finding.md) |
