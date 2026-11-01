//! Contour integrals `∫ f(z) dz` of complex functions along paths in the complex plane.
//!
//! A path is a [`Contour`]: a chain of [`PathSegment`]s (straight lines, circular arcs, or any
//! smooth parametrised curve). Each segment is integrated as `∫ f(z(t)) z'(t) dt` with an adaptive
//! 15-point Gauss-Kronrod rule applied to the complex integrand, with the Gauss 7-point rule giving
//! the error estimate.
//!
//! On top of that are the standard tools of complex analysis: [`residue`], the Cauchy formula for
//! derivatives [`cauchy_derivative`], and the [`winding_number`] of a closed path around a point.
//!
//! See the book entry [Complex contour integration](https://jwkunz.github.io/lib_jsl/book/integration/contour-integration.html).

use crate::adaptive_gl7k15_integration_engine::{GAUSS_LEGENDRE_WEIGHTS_7, KRONROD_NODES_15, KRONROD_WEIGHTS_15};
use lib_jsl_core::ErrorsJSL;
use num_complex::Complex;
use std::f64::consts::TAU;
use std::fmt;
use std::rc::Rc;

/// A double-precision complex number.
pub type C64 = Complex<f64>;

/// One smooth piece of a contour.
#[derive(Clone)]
pub enum PathSegment {
    /// The straight segment from `from` to `to`.
    Line { from: C64, to: C64 },
    /// The circular arc `centre + radius·e^{iθ}` for `θ` from `start_angle` to `end_angle`
    /// (radians; counter-clockwise when `end_angle > start_angle`).
    Arc { centre: C64, radius: f64, start_angle: f64, end_angle: f64 },
    /// A parametrised curve `z(t)` for `t` in `[t0, t1]` with derivative `dz(t)`.
    Curve { z: Rc<dyn Fn(f64) -> C64>, dz: Rc<dyn Fn(f64) -> C64>, t0: f64, t1: f64 },
}

impl fmt::Debug for PathSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PathSegment::Line { from, to } => write!(f, "Line({from} -> {to})"),
            PathSegment::Arc { centre, radius, start_angle, end_angle } => write!(f, "Arc(centre {centre}, r {radius}, {start_angle}..{end_angle})"),
            PathSegment::Curve { t0, t1, .. } => write!(f, "Curve(t {t0}..{t1})"),
        }
    }
}

impl PathSegment {
    /// The straight segment from `from` to `to`.
    pub fn line(from: C64, to: C64) -> Self {
        PathSegment::Line { from, to }
    }

    /// The full counter-clockwise circle, starting and ending at angle `0`.
    pub fn circle(centre: C64, radius: f64) -> Self {
        PathSegment::Arc { centre, radius, start_angle: 0.0, end_angle: TAU }
    }

    /// The circular arc `centre + radius·e^{iθ}`, `θ` from `start_angle` to `end_angle` (radians).
    pub fn arc(centre: C64, radius: f64, start_angle: f64, end_angle: f64) -> Self {
        PathSegment::Arc { centre, radius, start_angle, end_angle }
    }

    /// The curve `z(t)`, `t` from `t0` to `t1`, with its derivative `dz(t) = z'(t)`.
    pub fn curve(z: impl Fn(f64) -> C64 + 'static, dz: impl Fn(f64) -> C64 + 'static, t0: f64, t1: f64) -> Self {
        PathSegment::Curve { z: Rc::new(z), dz: Rc::new(dz), t0, t1 }
    }

    /// The segment's parameter interval.
    fn interval(&self) -> (f64, f64) {
        match self {
            PathSegment::Line { .. } => (0.0, 1.0),
            PathSegment::Arc { start_angle, end_angle, .. } => (*start_angle, *end_angle),
            PathSegment::Curve { t0, t1, .. } => (*t0, *t1),
        }
    }

    /// The point `z(t)` and velocity `z'(t)` at parameter `t`.
    fn eval(&self, t: f64) -> (C64, C64) {
        match self {
            PathSegment::Line { from, to } => (from + (to - from) * t, to - from),
            PathSegment::Arc { centre, radius, .. } => {
                let e = C64::from_polar(1.0, t);
                (centre + e * *radius, C64::new(0.0, 1.0) * e * *radius)
            }
            PathSegment::Curve { z, dz, .. } => (z(t), dz(t)),
        }
    }

    /// The first point of the segment.
    pub fn start(&self) -> C64 {
        self.eval(self.interval().0).0
    }

    /// The last point of the segment.
    pub fn end(&self) -> C64 {
        self.eval(self.interval().1).0
    }

    /// The point at fraction `s` in `[0, 1]` of the parameter range.
    pub fn point_at(&self, s: f64) -> C64 {
        let (a, b) = self.interval();
        self.eval(a + s * (b - a)).0
    }
}

/// A path made of consecutive segments.
#[derive(Debug, Clone, Default)]
pub struct Contour {
    pub segments: Vec<PathSegment>,
}

impl Contour {
    /// The contour through the given segments, in order.
    pub fn new(segments: Vec<PathSegment>) -> Self {
        Contour { segments }
    }

    /// The polyline through `vertices` (closed back to the first vertex when `closed`).
    pub fn polyline(vertices: &[C64], closed: bool) -> Self {
        let n = vertices.len();
        let count = if closed { n } else { n.saturating_sub(1) };
        Contour { segments: (0..count).map(|i| PathSegment::line(vertices[i], vertices[(i + 1) % n])).collect() }
    }

    /// The counter-clockwise circle about `centre`.
    pub fn circle(centre: C64, radius: f64) -> Self {
        Contour { segments: vec![PathSegment::circle(centre, radius)] }
    }

    /// The axis-aligned rectangle with opposite corners `lower_left` and `upper_right`, traversed
    /// counter-clockwise.
    pub fn rectangle(lower_left: C64, upper_right: C64) -> Self {
        let (a, c) = (lower_left, upper_right);
        Self::polyline(&[a, C64::new(c.re, a.im), c, C64::new(a.re, c.im)], true)
    }

    /// Appends another contour's segments.
    pub fn then(mut self, other: Contour) -> Self {
        self.segments.extend(other.segments);
        self
    }

    /// Whether each segment starts where the previous one ends (within `tol`).
    pub fn is_connected(&self, tol: f64) -> bool {
        self.segments.windows(2).all(|w| (w[0].end() - w[1].start()).norm() <= tol)
    }

    /// Whether the contour is connected and returns to its starting point (within `tol`).
    pub fn is_closed(&self, tol: f64) -> bool {
        match (self.segments.first(), self.segments.last()) {
            (Some(first), Some(last)) => self.is_connected(tol) && (last.end() - first.start()).norm() <= tol,
            _ => false,
        }
    }

    /// The arc length (adaptively integrated `|z'(t)|`).
    pub fn length(&self) -> f64 {
        self.segments.iter().map(|s| integrate_segment(&|_, dz| C64::new(dz.norm(), 0.0), s, 1e-12, true).0.re).sum()
    }
}

/// The result of a contour integral.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathIntegral {
    pub value: C64,
    /// An estimate of the absolute error (sum of the Gauss/Kronrod differences).
    pub error: f64,
    /// Number of integrand evaluations.
    pub evaluations: usize,
}

/// The 7-point Gauss and 15-point Kronrod estimates of `∫ₐᵇ f`, and the Kronrod estimate of
/// `∫ₐᵇ |f|`.
fn gk15(f: &mut impl FnMut(f64) -> C64, a: f64, b: f64) -> (C64, C64, f64) {
    let (centre, half) = (0.5 * (a + b), 0.5 * (b - a));
    let mut kronrod = C64::new(0.0, 0.0);
    let mut gauss = C64::new(0.0, 0.0);
    let mut absolute = 0.0;
    for (i, (&x, &wk)) in KRONROD_NODES_15.iter().zip(&KRONROD_WEIGHTS_15).enumerate() {
        let (sample, magnitude) = if i == 0 {
            let v = f(centre);
            (v, v.norm())
        } else {
            let (lo, hi) = (f(centre - half * x), f(centre + half * x));
            (lo + hi, lo.norm() + hi.norm())
        };
        kronrod += sample * wk;
        absolute += magnitude * wk;
        // The even-indexed Kronrod nodes (centre first) coincide with the 7-point Gauss nodes.
        if i % 2 == 0 {
            gauss += sample * GAUSS_LEGENDRE_WEIGHTS_7[i / 2];
        }
    }
    (gauss * half, kronrod * half, absolute * half.abs())
}

/// Integrates `weight(z(t), z'(t))` over the segment's parameter range (`weight = f(z) z'` for a
/// contour integral, `|z'|` for arc length).
/// Returns the value, the error estimate, the number of evaluations, and whether the tolerance
/// was met (panels accepted only because of the depth or evaluation limits, with a total error
/// well above the tolerance, mean it was not).
fn integrate_segment(weight: &dyn Fn(C64, C64) -> C64, seg: &PathSegment, tol: f64, absolute: bool) -> (C64, f64, usize, bool) {
    let (a, b) = seg.interval();
    let evals = std::cell::Cell::new(0usize);
    let mut integrand = |t: f64| {
        evals.set(evals.get() + 1);
        let (z, dz) = seg.eval(t);
        weight(z, dz)
    };
    // Stack-based adaptive bisection: each interval is accepted when its error is within its
    // share of the tolerance (proportional to its width). The segment starts as four panels, so
    // that an integrand oscillating several times along it (e^z/z^13 around a circle, say) is not
    // aliased by a single 15-point rule into a wrong answer that looks converged.
    let total_width = (b - a).abs();
    const INITIAL_PANELS: usize = 4;
    let mut stack: Vec<(f64, f64, usize)> = (0..INITIAL_PANELS)
        .map(|i| (a + (b - a) * i as f64 / INITIAL_PANELS as f64, a + (b - a) * (i + 1) as f64 / INITIAL_PANELS as f64, 0))
        .collect();
    let (mut value, mut error) = (C64::new(0.0, 0.0), 0.0);
    let mut forced = false;
    // A relative tolerance is measured against ∫|f(z) z'| dt, the scale that rounding errors in the
    // sum are proportional to: the integral itself may be far smaller through cancellation.
    let mut magnitude = 0.0;
    for &(lo, hi, _) in &stack {
        let (_, k, m) = gk15(&mut integrand, lo, hi);
        if !k.re.is_finite() || !k.im.is_finite() {
            // The integrand blew up (for example a path through a pole): bisecting cannot help.
            return (C64::new(f64::NAN, f64::NAN), f64::NAN, evals.get(), false);
        }
        magnitude += m;
    }
    let scale = if absolute { 1.0 } else { magnitude.max(1e-300) };
    const MAX_EVALUATIONS: usize = 2_000_000;
    while let Some((lo, hi, depth)) = stack.pop() {
        let (g, k, _) = gk15(&mut integrand, lo, hi);
        let err = (k - g).norm();
        let allowed = tol * scale * if total_width > 0.0 { (hi - lo).abs() / total_width } else { 1.0 };
        // Accept when within tolerance; also when out of budget or refined past useful precision,
        // or when the estimate is not finite (a singular sub-interval), where more splitting is futile.
        if err <= allowed || depth >= 40 || evals.get() > MAX_EVALUATIONS || !err.is_finite() {
            forced |= err > allowed || !err.is_finite();
            value += k;
            error += err;
        } else {
            let mid = 0.5 * (lo + hi);
            stack.push((lo, mid, depth + 1));
            stack.push((mid, hi, depth + 1));
        }
    }
    let converged = !forced || error <= 10.0 * tol * scale;
    (value, error, evals.get(), converged)
}

/// Integrates `f` along the contour to relative tolerance `tol` (for example `1e-10`) per segment,
/// relative to `∫|f(z)| |dz|` along the segment (the smallest error floating point can promise when
/// the integral is small through cancellation).
pub fn contour_integral(f: impl Fn(C64) -> C64, contour: &Contour, tol: f64) -> Result<PathIntegral, ErrorsJSL> {
    if contour.segments.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("The contour has no segments."));
    }
    if tol.is_nan() || tol <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("The tolerance must be positive."));
    }
    let mut total = PathIntegral { value: C64::new(0.0, 0.0), error: 0.0, evaluations: 0 };
    for seg in &contour.segments {
        let (v, e, n, converged) = integrate_segment(&|z, dz| f(z) * dz, seg, tol, false);
        if !converged {
            return Err(ErrorsJSL::RuntimeError("The integral did not converge (the path may pass through or very near a singularity)."));
        }
        total.value += v;
        total.error += e;
        total.evaluations += n;
    }
    if !total.value.re.is_finite() || !total.value.im.is_finite() {
        return Err(ErrorsJSL::RuntimeError("The integral is not finite (the path may pass through a singularity)."));
    }
    Ok(total)
}

/// The integral along the polyline through `vertices` (open).
pub fn polyline_integral(f: impl Fn(C64) -> C64, vertices: &[C64], tol: f64) -> Result<PathIntegral, ErrorsJSL> {
    contour_integral(f, &Contour::polyline(vertices, false), tol)
}

/// The residue of `f` at `z0`: `(1/2πi) ∮ f(z) dz` over a counter-clockwise circle of the given
/// `radius`. The circle must enclose `z0` and no other singularity of `f`.
pub fn residue(f: impl Fn(C64) -> C64, z0: C64, radius: f64, tol: f64) -> Result<C64, ErrorsJSL> {
    if radius.is_nan() || radius <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("The radius must be positive."));
    }
    let r = contour_integral(f, &Contour::circle(z0, radius), tol)?;
    Ok(r.value / C64::new(0.0, TAU))
}

/// The `n`-th derivative `f⁽ⁿ⁾(z0)` of an analytic function by Cauchy's integral formula,
/// `n!/(2πi) ∮ f(z)/(z - z0)^{n+1} dz`, over a circle of the given `radius` inside the region
/// where `f` is analytic.
pub fn cauchy_derivative(f: impl Fn(C64) -> C64, z0: C64, n: u32, radius: f64, tol: f64) -> Result<C64, ErrorsJSL> {
    if radius.is_nan() || radius <= 0.0 {
        return Err(ErrorsJSL::InvalidInputRange("The radius must be positive."));
    }
    let r = contour_integral(|z| f(z) / (z - z0).powi(n as i32 + 1), &Contour::circle(z0, radius), tol)?;
    let factorial: f64 = (1..=n).map(f64::from).product();
    Ok(r.value * factorial / C64::new(0.0, TAU))
}

/// The winding number of a closed polygon (vertices in order, implicitly closed) around `point`:
/// positive for counter-clockwise loops. Computed from the accumulated change of argument, so it
/// is exact for any simple or self-intersecting polygon. `None` if `point` lies on the boundary.
pub fn winding_number(vertices: &[C64], point: C64) -> Option<i32> {
    let n = vertices.len();
    if n < 3 {
        return Some(0);
    }
    let mut total = 0.0;
    for i in 0..n {
        let (a, b) = (vertices[i] - point, vertices[(i + 1) % n] - point);
        if a.norm() == 0.0 || b.norm() == 0.0 || (a.re * b.im - a.im * b.re).abs() <= 1e-14 * a.norm() * b.norm() && a.re * b.re + a.im * b.im < 0.0 {
            return None; // vertex or edge passes through the point
        }
        total += (a.re * b.im - a.im * b.re).atan2(a.re * b.re + a.im * b.im);
    }
    Some((total / TAU).round() as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn close(a: C64, b: C64, tol: f64) -> bool {
        (a - b).norm() < tol
    }

    #[test]
    fn integral_of_one_over_z_around_the_unit_circle() {
        let r = contour_integral(|z| z.inv(), &Contour::circle(C64::new(0.0, 0.0), 1.0), 1e-12).unwrap();
        assert!(close(r.value, C64::new(0.0, TAU), 1e-10), "{:?}", r);
        assert!(r.error < 1e-9);
        // Off-centre circle that still encloses the origin, and one that does not.
        let enclosing = contour_integral(|z| z.inv(), &Contour::circle(C64::new(0.5, 0.0), 1.0), 1e-12).unwrap();
        assert!(close(enclosing.value, C64::new(0.0, TAU), 1e-9));
        let outside = contour_integral(|z| z.inv(), &Contour::circle(C64::new(3.0, 0.0), 1.0), 1e-12).unwrap();
        assert!(outside.value.norm() < 1e-10);
    }

    #[test]
    fn analytic_functions_are_path_independent() {
        let f = |z: C64| z * z + z.exp();
        let antiderivative = |z: C64| z * z * z / 3.0 + z.exp();
        let (a, b) = (C64::new(0.0, 0.0), C64::new(1.0, 2.0));
        let want = antiderivative(b) - antiderivative(a);
        let straight = contour_integral(f, &Contour::new(vec![PathSegment::line(a, b)]), 1e-12).unwrap();
        let dog_leg = polyline_integral(f, &[a, C64::new(2.0, -1.0), b], 1e-12).unwrap();
        let arc = Contour::new(vec![PathSegment::line(a, C64::new(0.0, 2.0)), PathSegment::line(C64::new(0.0, 2.0), b)]);
        let via_arc = contour_integral(f, &arc, 1e-12).unwrap();
        for r in [straight, dog_leg, via_arc] {
            assert!(close(r.value, want, 1e-9), "{:?} vs {want}", r.value);
        }
        // Closed contour integral of an analytic function vanishes.
        let closed = contour_integral(f, &Contour::rectangle(C64::new(-1.0, -1.0), C64::new(2.0, 1.5)), 1e-12).unwrap();
        assert!(closed.value.norm() < 1e-9);
    }

    #[test]
    fn powers_around_a_circle_and_curves() {
        // ∮ z^n dz = 2πi for n = -1 and 0 otherwise.
        for n in [-3, -2, 0, 1, 4] {
            let r = contour_integral(|z| z.powi(n), &Contour::circle(C64::new(0.0, 0.0), 2.0), 1e-12).unwrap();
            assert!(r.value.norm() < 1e-9, "n={n}");
        }
        // The same circle as a generic parametrised curve.
        let curve = PathSegment::curve(|t| C64::from_polar(2.0, t), |t| C64::new(0.0, 1.0) * C64::from_polar(2.0, t), 0.0, TAU);
        let r = contour_integral(|z| z.inv(), &Contour::new(vec![curve]), 1e-12).unwrap();
        assert!(close(r.value, C64::new(0.0, TAU), 1e-10));
        // Length of a circle and of a polyline.
        assert!((Contour::circle(C64::new(1.0, 1.0), 3.0).length() - 6.0 * PI).abs() < 1e-9);
        assert!((Contour::polyline(&[C64::new(0.0, 0.0), C64::new(3.0, 4.0)], false).length() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn residues_and_the_residue_theorem() {
        // 1/(z²+1) has residue 1/(2i) = -i/2 at i and +i/2 at -i.
        let f = |z: C64| (z * z + 1.0).inv();
        let at_i = residue(f, C64::new(0.0, 1.0), 0.5, 1e-12).unwrap();
        assert!(close(at_i, C64::new(0.0, -0.5), 1e-10));
        let at_minus_i = residue(f, C64::new(0.0, -1.0), 0.5, 1e-12).unwrap();
        assert!(close(at_minus_i, C64::new(0.0, 0.5), 1e-10));
        // A contour around both poles gives 2πi times the sum of residues (zero here).
        let both = contour_integral(f, &Contour::rectangle(C64::new(-2.0, -2.0), C64::new(2.0, 2.0)), 1e-12).unwrap();
        assert!(both.value.norm() < 1e-9);
        // A double pole: e^z / z² has residue 1 at 0; sin z / z³ has residue 0 (and e^z / z³ has 1/2).
        assert!(close(residue(|z| z.exp() / (z * z), C64::new(0.0, 0.0), 1.0, 1e-12).unwrap(), C64::new(1.0, 0.0), 1e-10));
        assert!(close(residue(|z| z.exp() / (z * z * z), C64::new(0.0, 0.0), 1.0, 1e-12).unwrap(), C64::new(0.5, 0.0), 1e-10));
        assert!(residue(f, C64::new(0.0, 1.0), -1.0, 1e-9).is_err());
    }

    #[test]
    fn cauchy_formula_for_derivatives() {
        let z0 = C64::new(0.3, -0.2);
        for n in 0..5 {
            let d = cauchy_derivative(|z| z.exp(), z0, n, 1.0, 1e-12).unwrap();
            assert!(close(d, z0.exp(), 1e-9), "n={n}");
        }
        // e^z/z^13 oscillates 13 times around the circle; a single 15-point panel used to alias
        // it into a wrong answer that looked converged (error ≈ 1 at radius 2.5). Radii below about
        // 3 are limited by rounding instead (see the book figure).
        for r in [3.0, 6.0, 10.0] {
            let d = cauchy_derivative(|z| z.exp(), C64::new(0.0, 0.0), 12, r, 1e-14).unwrap();
            assert!(close(d, C64::new(1.0, 0.0), 1e-9), "r={r}: {d}");
        }
        // d²/dz² of z⁵ at 2 is 20·8 = 160.
        assert!(close(cauchy_derivative(|z| z.powi(5), C64::new(2.0, 0.0), 2, 0.5, 1e-12).unwrap(), C64::new(160.0, 0.0), 1e-8));
    }

    #[test]
    fn winding_numbers_and_contour_properties() {
        let square = [C64::new(-1.0, -1.0), C64::new(1.0, -1.0), C64::new(1.0, 1.0), C64::new(-1.0, 1.0)];
        assert_eq!(winding_number(&square, C64::new(0.0, 0.0)), Some(1));
        assert_eq!(winding_number(&square, C64::new(3.0, 0.0)), Some(0));
        let reversed: Vec<C64> = square.iter().rev().copied().collect();
        assert_eq!(winding_number(&reversed, C64::new(0.2, 0.1)), Some(-1));
        assert_eq!(winding_number(&square, C64::new(1.0, 0.0)), None); // on an edge
        assert_eq!(winding_number(&square, C64::new(1.0, 1.0)), None); // on a vertex
        // A figure that loops twice winds twice.
        let twice: Vec<C64> = (0..16).map(|k| C64::from_polar(1.0, TAU * 2.0 * k as f64 / 16.0)).collect();
        assert_eq!(winding_number(&twice, C64::new(0.0, 0.0)), Some(2));
        let c = Contour::rectangle(C64::new(0.0, 0.0), C64::new(1.0, 1.0));
        assert!(c.is_closed(1e-12) && Contour::polyline(&square, false).is_connected(1e-12) && !Contour::polyline(&square, false).is_closed(1e-12));
        assert!(Contour::default().segments.is_empty() && !Contour::default().is_closed(1e-12));
    }

    #[test]
    fn error_handling() {
        assert!(contour_integral(|z| z, &Contour::default(), 1e-9).is_err());
        assert!(contour_integral(|z| z, &Contour::circle(C64::new(0.0, 0.0), 1.0), 0.0).is_err());
        // A path straight through a pole is reported rather than returning garbage.
        let through_pole = Contour::new(vec![PathSegment::line(C64::new(-1.0, 0.0), C64::new(1.0, 0.0))]);
        assert!(contour_integral(|z| z.inv(), &through_pole, 1e-9).is_err());
        let arc = Contour::new(vec![PathSegment::arc(C64::new(0.0, 0.0), 1.0, 0.0, PI)]);
        let half = contour_integral(|z| z.inv(), &arc, 1e-12).unwrap();
        assert!(close(half.value, C64::new(0.0, PI), 1e-10));
    }
}
