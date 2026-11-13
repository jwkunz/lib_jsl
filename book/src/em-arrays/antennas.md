# Antenna models: dipoles, loops, dishes and patches

<div class="entry-summary">

| | |
|---|---|
| **Problem** | Give the far-field pattern, radiated power, directivity and radiation resistance of the classic antennas, from closed-form current or aperture models |
| **Devised by** | Heinrich Hertz (1887), the dipole; Samuel Silver and the MIT Radiation Laboratory (1940s), aperture theory of reflectors; Robert Munson and John Howell (1972–74), microstrip patches; formulas collected by Constantine Balanis (1982) |
| **Accuracy** | Directivities and radiation resistances agree with the closed forms to four digits (table below); the patch model is the standard approximate transmission-line/cavity model |
| **Cost** | A few function evaluations per direction; radiated power by Gauss-Legendre quadrature over the sphere |
| **API** | [`Antenna`](../../api/lib_jsl_electromagnetics/antenna_models/trait.Antenna.html) (`far_field`, `radiated_power`, `directivity`, `peak_directivity`, `radiation_resistance`, `far_field_pattern`, `gain_pattern`), [`HertzianDipole`](../../api/lib_jsl_electromagnetics/antenna_models/hertzian_dipole/struct.HertzianDipole.html), [`MagneticLoop`](../../api/lib_jsl_electromagnetics/antenna_models/magnetic_loop/struct.MagneticLoop.html), [`Dipole`](../../api/lib_jsl_electromagnetics/antenna_models/half_wave_dipole/struct.Dipole.html), [`HalfWaveDipole`](../../api/lib_jsl_electromagnetics/antenna_models/half_wave_dipole/struct.HalfWaveDipole.html), [`Dish`](../../api/lib_jsl_electromagnetics/antenna_models/dish/struct.Dish.html), [`PatchAntenna`](../../api/lib_jsl_electromagnetics/antenna_models/patch_antenna/struct.PatchAntenna.html), [`Isotropic`](../../api/lib_jsl_electromagnetics/antenna_models/struct.Isotropic.html) |

</div>

## Overview

An antenna converts a guided current into a radiated wave. Far from it, the field falls as $1/r$ and depends on direction through the **pattern** $(E_\theta, E_\varphi)$. The pattern determines the **radiation intensity** $U = |\mathbf E|^2/2\eta$ (W/sr), the radiated power $P = \oint U\,d\Omega$, and the **directivity** $D = 4\pi U/P$, which compares the antenna with an isotropic radiator.
For a current-fed antenna, the **radiation resistance** $R = 2P/|I|^2$ is the resistance that would dissipate the radiated power. Each model here gives its pattern in closed form, from an assumed current or aperture field, and implements the [`Antenna`](../../api/lib_jsl_electromagnetics/antenna_models/trait.Antenna.html) trait, from which the rest is computed numerically.

## History

Heinrich Hertz (1887) produced and detected electromagnetic waves with a spark-gap dipole and a loop, confirming Maxwell's theory. Guglielmo Marconi made them practical for telegraphy. The thin-wire dipole with a sinusoidal current, analysed in the 1920s and 30s, gives the half-wave dipole's 73 Ω and 1.64 directivity. Microwave radar in the Second World War brought reflector dishes and
horns, analysed as radiating apertures by Samuel Silver and colleagues (*Microwave Antenna Theory and Design*, 1949). Microstrip patch antennas, proposed by Georges Deschamps (1953), became practical with printed circuits: Robert Munson (1974) and John Howell (1975) built them for missiles. They are now in every phone and GPS receiver.

## The idea

### Electric and magnetic dipoles

A few quantities describe every antenna in this entry.

- The wavelength is $\lambda$, the **wavenumber** is $k = 2\pi/\lambda$ (the phase advance per
  metre), and $\eta \approx 377\ \Omega$ is the impedance of free space, the ratio of the electric
  to the magnetic field in a plane wave.
- Far from an antenna the field is a spherical wave, $e^{-jkr}/r$ times a function of direction
  only, the **far-field pattern**. The formulas below give that function (the factor
  $e^{-jkr}/r$ is understood), in spherical coordinates with $\theta$ measured from the $z$ axis.
- The **directivity** $D$ is the power density the antenna radiates in its best direction,
  divided by the average over all directions. An antenna that radiated equally in all directions
  would have $D = 1$.
- The **radiation resistance** $R$ is the resistance that would dissipate the same power as the
  antenna radiates, for the same feed current: $P = \tfrac12|I|^2R$. A small $R$ means that large
  currents are needed to radiate little power, and that the antenna is hard to feed efficiently.

A **Hertzian dipole**, a uniform current $I$ on a length $l \ll \lambda$ along $z$, radiates $E_\theta = j\eta kIl\sin\theta/(4\pi)$. The factor $\sin\theta$ is the whole pattern: nothing is radiated along the wire, and the most is
radiated broadside to it, a doughnut shape. Its directivity is 1.5, and $R = 80\pi^2(l/\lambda)^2$, tiny for short antennas. The model also gives its exact near field, with the $1/r^2$ and $1/r^3$ terms. A **small loop** of radius $a$ is a magnetic dipole: the same pattern with $\mathbf E$ along $\hat\varphi$, and $R = 20\pi^2(ka)^4$ per turn squared.

### Thin-wire dipoles

A practical dipole is not short, and its current is not uniform: it must fall to zero at the open
ends of the wire. A longer wire can be regarded as a line of Hertzian dipoles, whose fields add
with phases that depend on direction. A centre-fed wire of length $L$ with the standard sinusoidal current $I(z) = I_m\sin\big(k(L/2 - |z|)\big)$ radiates

$$
E_\theta = \frac{j\eta I_m}{2\pi}\,\frac{\cos\big(\tfrac{kL}{2}\cos\theta\big) - \cos\tfrac{kL}{2}}{\sin\theta}.
$$

At $L = \lambda/2$ this gives $D = 1.64$ and $R = 30\,\mathrm{Cin}(2\pi) = 73.1\ \Omega$. Longer dipoles sharpen the broadside beam until, past $1.25\lambda$, lobes split off toward the ends:

![Dipole patterns](../generated/em-arrays/dipole_patterns.svg)

<p class="caption" id="fig-20-2"><strong>Figure 20.2.</strong> Dipole patterns.</p>

The radiation resistance is referred to the current maximum $I_m$, which is the feed current only for odd multiples of a half wavelength.

### Apertures: the dish

A large antenna such as a reflector dish is best described not by its currents but by the field
across its opening, the **aperture**. By Huygens' principle every point of the aperture acts as
a small source, and the far field in a given direction is the sum of their contributions with
the phase differences appropriate to that direction. That sum is a two-dimensional Fourier
transform of the aperture field: a wide aperture gives a narrow beam, just as a long signal has a
narrow spectrum. The dish is modelled as a circular aperture of diameter $D$ (radius $a$) with a
$\hat y$-polarised field $E_0(1 - \rho^2/a^2)^p$, where $\rho$ is the distance from the centre and
the exponent $p$ controls how strongly the field is tapered towards the edge. For a circular
aperture the Fourier transform becomes a Fourier-Bessel transform:

$$
[E_\theta, E_\varphi] \propto 2^pp!\,\frac{J_{p+1}(u)}{u^{p+1}}\,\frac{1 + \cos\theta}{2}\,[\sin\varphi, \cos\varphi], \qquad u = ka\sin\theta.
$$

Uniform illumination ($p = 0$) gives the maximum directivity $(\pi D/\lambda)^2$, the Airy pattern with −17.6 dB sidelobes, and a half-power beamwidth of $1.029\lambda/D$. Tapering the edge lowers the sidelobes at the cost of directivity:

![Dish patterns](../generated/em-arrays/dish_patterns.svg)

<p class="caption" id="fig-20-3"><strong>Figure 20.3.</strong> Dish patterns.</p>

{{#include ../generated/em-arrays/dish_table.md}}

### Microstrip patches

A rectangular patch of length $L$ on a grounded substrate of permittivity $\varepsilon_r$ and height $h$ resonates when $L$ is about half a wavelength in the dielectric. Fringing fields lengthen it electrically by $\Delta L$ and lower the effective permittivity to $\varepsilon_{\text{eff}}$. The **transmission-line model** treats the two radiating edges as slots of width $W$, separated by $L + 2\Delta L$.
[`PatchAntenna::design`](../../api/lib_jsl_electromagnetics/antenna_models/patch_antenna/struct.PatchAntenna.html#method.design) chooses $W$ and $L$ for a frequency. The model gives the edge resistance from the slot conductances and the inset depth that matches it to 50 Ω. The pattern is broad, wider in the E-plane than in the H-plane:

![Patch patterns](../generated/em-arrays/patch_patterns.svg)

<p class="caption" id="fig-20-4"><strong>Figure 20.4.</strong> Patch patterns.</p>

### Checks against closed forms

<p class="caption" id="tab-20-2"><strong>Table 20.2.</strong> Directivity and radiation resistance of antenna models against closed forms.</p>

{{#include ../generated/em-arrays/antenna_table.md}}

## Algorithm

Every antenna model supplies one thing, its far-field pattern, and the remaining quantities
are computed from that in the same way for all of them.

1. **Pattern.** Evaluate the model's closed-form $(E_\theta, E_\varphi)$ in the requested
   direction, for a unit excitation: a unit current for the wire antennas, a unit aperture
   field for the dish, a unit edge voltage for the patch.
2. **Radiated power.** Integrate the power density over the sphere,
   $P = \oint |\mathbf E|^2/2\eta\,d\Omega$, by
   [Gauss-Legendre quadrature](fields.md#domains-and-quadrature) in $\theta$ and uniform
   samples in $\varphi$. Electrically large antennas have many narrow lobes and are given
   more nodes.
3. **Directivity.** In a given direction it is $4\pi U/P$, where $U = |\mathbf E|^2/2\eta$ is the
   radiation intensity there. The peak directivity is found by searching the sphere, finely in
   $\theta$ alone when the pattern is symmetric about the axis.
4. **Radiation resistance**, for the wire antennas, whose excitation is a current: from
   $P = \tfrac12|I|^2R$ with a unit current, $R = 2P$.
5. **Gain pattern.** Rescale the pattern so that $|\mathbf E|^2$ equals the directivity times
   the efficiency in every direction. This is the normalisation the
   [array models](beamforming.md) expect.

## Implementation

```rust,ignore
{{#include ../../../crates/electromagnetics/src/antenna_models/half_wave_dipole.rs:dipole_far_field}}
```

- Every model gives its far field in either time convention (they differ by conjugation).
- `far_field_pattern` fills a container on a far-field sphere; `HarmonicField::transformed` rotates or moves it.
- These are analytic models, not full-wave solutions; for a wire antenna's actual current, see the [method of moments](../em-waves/method-of-moments.md) in the next chapter.

## In practice

```rust,ignore
{{#include ../../../crates/electromagnetics/examples/antenna_patterns.rs}}
```

```text
== antenna models at 10 GHz (λ = 30.0 mm) ==
  model                             D (lin)  D (dBi)        R_rad (Ω)
  Hertzian dipole (λ/50)               1.50     1.76           0.3156
  small loop (a = λ/30)                1.50     1.76           0.3795
  half-wave dipole                     1.64     2.15          73.0790
  full-wave dipole                     2.41     3.82         198.9500
  microstrip patch (ε_r 2.2)           5.44     7.35                -
  dish, D = 10λ, uniform             986.96    29.94                -
  dish, D = 10λ, tapered (p=2)       548.31    27.39                -

  patch: W = 11.85 mm, L = 9.05 mm, edge resistance 228 Ω, 50 Ω inset 3.12 mm
  dish: half-power beamwidth 5.89° (uniform circular aperture: 1.029 λ/D = 5.90°)
```

The example then draws four polar pattern cuts in the terminal.

**Strengths**

- Exact closed forms for the idealised currents and apertures; consistent power, directivity and resistance from one trait.
- Patterns in either time convention, as containers ready for arrays and overlap integrals.

**Weaknesses**

- Assumed currents: no input reactance, no feed or mutual-coupling effects, and the sinusoidal current is only approximate for thick or long wires.
- The patch model ignores substrate surface waves and finite ground planes.

## Exercises

1. Integrate $\sin^2\theta$ over the sphere to obtain the Hertzian dipole's directivity of 1.5.
2. Show that the dipole formula reduces to the Hertzian dipole's for $kL \ll 1$ when $I_m$ is replaced by the average current.
3. Show that a uniform circular aperture's directivity is $(\pi D/\lambda)^2$, and find the efficiency of the taper $(1 - \rho^2/a^2)^p$.
4. Find the length at which a dipole's broadside lobe vanishes, and explain why the radiation resistance referred to the feed diverges for a full-wave dipole.
5. Design a patch for 2.45 GHz on FR-4 ($\varepsilon_r = 4.4$, $h = 1.6$ mm) and find its 50 Ω inset.

<details>
<summary>Hints and answers</summary>

1. $D = 4\pi U_{\max}/P_{\mathrm{rad}}$ with $U \propto \sin^2\theta$: $\int\sin^2\theta\,\mathrm d\Omega = 2\pi\int_0^\pi\sin^3\theta\,\mathrm d\theta = 8\pi/3$, so $D = 4\pi/(8\pi/3) = 3/2$.
2. For $kL \ll 1$ the current is triangular with average $I_m/2$ (since $\sin(k(L/2 - |z|)) \approx k(L/2 - |z|)$), and the radiation field is that of an element carrying the average current over length $L$.
3. For a uniform aperture, the effective area equals the physical area $\pi D^2/4$, so $D = 4\pi A_e/\lambda^2 = (\pi D/\lambda)^2$. A taper reduces the aperture efficiency to $\dfrac{|\int f\,\mathrm dA|^2}{A\int|f|^2\mathrm dA}$.
4. The pattern $[\cos(\tfrac{kL}2\cos\theta) - \cos\tfrac{kL}2]/\sin\theta$ at broadside is $1 - \cos\tfrac{kL}2$, which vanishes for $kL/2 = 2\pi$, i.e. $L = 2\lambda$. A full-wave dipole has $\sin(kL/2) = 0$, so the feed current is zero and $R_{\mathrm{feed}} = P_{\mathrm{rad}}/(I_{\mathrm{feed}}^2/2)$ diverges.
5. Use the transmission-line model of the patch: $W = \tfrac{c}{2f}\sqrt{2/(\varepsilon_r + 1)} \approx 37$ mm and $\varepsilon_{\mathrm{eff}} \approx 4.1$, giving $L \approx 29$ mm after the fringing extension; the 50 Ω inset follows from $R_{\mathrm{in}}(y_0) = R_{\mathrm{edge}}\cos^2(\pi y_0/L)$.

</details>

## Further reading

- C. A. Balanis, *Antenna Theory: Analysis and Design*, 4th ed., Wiley, 2016, chapters 4, 5, 12 and 14.
- J. D. Kraus and R. J. Marhefka, *Antennas for All Applications*, 3rd ed., McGraw-Hill, 2002.
- S. Silver (ed.), *Microwave Antenna Theory and Design*, MIT Radiation Laboratory Series 12, McGraw-Hill, 1949.
- R. E. Munson, "Conformal microstrip antennas and microstrip phased arrays", *IEEE Transactions on Antennas and Propagation* 22 (1974) 74–78.
