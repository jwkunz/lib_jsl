# Electromagnetics: transmission lines, propagation and computational methods

The [previous chapter](../em-arrays/index.md) dealt with fields, antennas and arrays. This one follows the signal between them: along transmission lines to the antenna, across the propagation path to the receiver, and through the computational methods that find fields when no closed form exists. It closes with three numerical solvers of increasing generality. Spherical wave expansions turn near fields into far fields. The method of moments finds the current on a wire. The finite-difference time-domain method solves Maxwell's equations on a grid.

## The central questions

- **Matched or mismatched?** Every junction of unequal impedances reflects power; matching networks trade bandwidth for a perfect match at one frequency.
- **Free space or real world?** Free-space loss is exact; ground reflection, buildings, multipath and the atmosphere need models, some physical and some empirical.
- **Closed form or numerical?** Analytic models assume currents and fields; full-wave solvers compute them, at the price of grids, meshes and their discretisation errors.
- **Frequency or time?** Moment methods solve one frequency at a time; FDTD gives every frequency from one pulse.

## Testing

Each method is checked against what it must satisfy:
- two-port conversions that invert each other, and cascades that equal longer lines;
- the textbook Hata value;
- near-to-far transforms of an exact dipole-array field, converging spectrally;
- power conservation in the moment method;
- Fresnel reflection and the Yee dispersion relation in the FDTD solvers.

Writing this chapter found no defects in these modules.

## In the library

All of these are in `lib_jsl_electromagnetics` ([API reference](../../api/lib_jsl_electromagnetics/index.html)): `networks`, `propagation`, `fields::spherical_waves`, and `algorithms`.

| You need | Use | Entry |
|---|---|---|
| Input impedance, reflection, matching; two-port parameters; a Smith chart | `TransmissionLine`, `single_stub_match`, `s_parameters`, `smith_chart` | [Transmission lines and networks](transmission-lines.md) |
| A link budget, path loss, multipath, atmospheric loss or radar range | `LinkBudget`, `okumura_hata_db`, `two_ray_factor`, `MultipathChannel`, `atmosphere`, `Radar` | [Propagation](propagation.md) |
| A far field from near-field data; a field's spherical wave content | `SphericalWaveExpansion`, `spherical_near_to_far`, `planar_near_to_far` | [Near-field to far-field](near-to-far.md) |
| The actual current and impedance of a wire dipole | `WireDipole` | [Method of moments](method-of-moments.md) |
| Fields in time, through materials and around objects | `Fdtd1D`, `FdtdTmz` | [FDTD](fdtd.md) |
