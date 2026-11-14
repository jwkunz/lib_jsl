//! Geometric transforms with B-spline interpolation, following `scipy.ndimage`: `spline_filter`,
//! `map_coordinates`, `affine_transform`, `shift`, `zoom` and `rotate`.
//!
//! Every geometric transform has the same core: for each output pixel, compute where it comes from
//! in the input (generally between pixels) and interpolate there. Interpolation of order `k` fits a
//! piecewise polynomial of degree `k` that is `k − 1` times differentiable: order 0 takes the
//! nearest pixel, order 1 is bilinear, and order 3 (the default) is the cubic B-spline. For orders
//! above 1 the spline must *pass through* the pixels, which requires solving for its coefficients
//! first: the **prefilter**, a pair of recursive filters per pole, run along each axis.
//!
//! See the book entry [Rank filters, derivative filters and geometric transforms](https://jwkunz.github.io/lib_jsl/book/image/rank-geometric.html).

use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;

use super::boundary::Mode;

/// The poles of the B-spline prefilter of each order (Unser, Aldroubi and Eden; as in SciPy).
fn poles(order: usize) -> Vec<f64> {
    match order {
        2 => vec![8f64.sqrt() - 3.0],
        3 => vec![3f64.sqrt() - 2.0],
        4 => vec![
            (664.0 - 438976f64.sqrt()).sqrt() + 304f64.sqrt() - 19.0,
            (664.0 + 438976f64.sqrt()).sqrt() - 304f64.sqrt() - 19.0,
        ],
        5 => vec![
            (67.5 - 4436.25f64.sqrt()).sqrt() + 26.25f64.sqrt() - 6.5,
            (67.5 + 4436.25f64.sqrt()).sqrt() - 26.25f64.sqrt() - 6.5,
        ],
        _ => vec![],
    }
}

/// The boundary condition of the prefilter recursion.
#[derive(Clone, Copy, PartialEq)]
enum Ext {
    /// Whole-sample symmetry, `c b | a b c | b a`.
    Mirror,
    /// Half-sample symmetry, `b a | a b c | c b`.
    Reflect,
    /// Periodic.
    Wrap,
}

// ANCHOR: prefilter
/// Converts one line of samples into B-spline coefficients in place: an overall gain, then for
/// each pole `z` a causal recursion `c[i] += z c[i − 1]` and an anti-causal one
/// `c[i] = z (c[i + 1] − c[i])`, each started from the infinite sum that the boundary symmetry
/// implies (SciPy's `spline_filter1d`).
fn prefilter_line(c: &mut [f64], order: usize, ext: Ext) {
    let n = c.len();
    if order < 2 || n < 2 {
        return;
    }
    let zs = poles(order);
    let gain: f64 = zs.iter().map(|&z| (1.0 - z) * (1.0 - 1.0 / z)).product();
    c.iter_mut().for_each(|v| *v *= gain);
    for &z in &zs {
        // Causal initial value.
        match ext {
            Ext::Mirror => {
                let zn1 = z.powi(n as i32 - 1);
                let mut zi = z;
                let mut c0 = c[0] + zn1 * c[n - 1];
                for i in 1..n - 1 {
                    c0 += zi * (c[i] + zn1 * c[n - 1 - i]);
                    zi *= z;
                }
                c[0] = c0 / (1.0 - zn1 * zn1);
            }
            Ext::Reflect => {
                let zn = z.powi(n as i32);
                let first = c[0];
                let mut zi = z;
                let mut c0 = c[0] + zn * c[n - 1];
                for i in 1..n {
                    c0 += zi * (c[i] + zn * c[n - 1 - i]);
                    zi *= z;
                }
                c[0] = c0 * z / (1.0 - zn * zn) + first;
            }
            Ext::Wrap => {
                let mut zi = z;
                let mut c0 = c[0];
                for i in 1..n {
                    c0 += zi * c[n - i];
                    zi *= z;
                }
                c[0] = c0 / (1.0 - zi);
            }
        }
        for i in 1..n {
            c[i] += z * c[i - 1];
        }
        // Anti-causal initial value.
        match ext {
            Ext::Mirror => c[n - 1] = (z * c[n - 2] + c[n - 1]) * z / (z * z - 1.0),
            Ext::Reflect => c[n - 1] *= z / (z - 1.0),
            Ext::Wrap => {
                let mut zi = z;
                let mut cn = c[n - 1];
                for &v in &c[..n - 1] {
                    cn += zi * v;
                    zi *= z;
                }
                c[n - 1] = cn * z / (zi - 1.0);
            }
        }
        for i in (0..n - 1).rev() {
            c[i] = z * (c[i + 1] - c[i]);
        }
    }
}
// ANCHOR_END: prefilter

fn ext_of(mode: Mode) -> Ext {
    match mode {
        Mode::Reflect => Ext::Reflect,
        Mode::Wrap => Ext::Wrap,
        _ => Ext::Mirror,
    }
}

/// Edge padding applied before prefiltering in `Nearest` mode, as SciPy does (its recursion has no
/// exact boundary condition for that mode).
const NEAREST_PAD: usize = 12;

/// B-spline coefficients of `image` for interpolation of the given order (SciPy's `spline_filter`
/// with the boundary condition that `mode` implies). Returns the coefficients and the padding
/// added at each side.
pub fn spline_filter(image: &Array2<f64>, order: usize, mode: Mode) -> Result<(Array2<f64>, usize), ErrorsJSL> {
    if order > 5 {
        return Err(ErrorsJSL::InvalidInputRange("spline order must be at most 5"));
    }
    let pad = if matches!(mode, Mode::Nearest) && order > 1 { NEAREST_PAD } else { 0 };
    let (rows, cols) = image.dim();
    let mut c = Array2::from_shape_fn((rows + 2 * pad, cols + 2 * pad), |(r, k)| {
        let rr = (r as isize - pad as isize).clamp(0, rows as isize - 1) as usize;
        let kk = (k as isize - pad as isize).clamp(0, cols as isize - 1) as usize;
        image[(rr, kk)]
    });
    let ext = ext_of(mode);
    for mut col in c.columns_mut() {
        let mut line = col.to_vec();
        prefilter_line(&mut line, order, ext);
        col.iter_mut().zip(line).for_each(|(d, v)| *d = v);
    }
    for mut row in c.rows_mut() {
        let mut line = row.to_vec();
        prefilter_line(&mut line, order, ext);
        row.iter_mut().zip(line).for_each(|(d, v)| *d = v);
    }
    Ok((c, pad))
}

/// The centred B-spline of degree `k` at `x`, `β_k(x)`.
fn bspline(k: usize, x: f64) -> f64 {
    let ax = x.abs();
    match k {
        0 => {
            if ax < 0.5 { 1.0 } else { 0.0 }
        }
        1 => (1.0 - ax).max(0.0),
        2 => {
            if ax < 0.5 { 0.75 - ax * ax } else if ax < 1.5 { 0.5 * (1.5 - ax).powi(2) } else { 0.0 }
        }
        3 => {
            if ax < 1.0 { 2.0 / 3.0 - ax * ax + 0.5 * ax.powi(3) } else if ax < 2.0 { (2.0 - ax).powi(3) / 6.0 } else { 0.0 }
        }
        4 => {
            if ax < 0.5 {
                let t = ax * ax;
                t * (t * 0.25 - 0.625) + 115.0 / 192.0
            } else if ax < 1.5 {
                ax * (ax * (ax * (5.0 / 6.0 - ax / 6.0) - 1.25) + 5.0 / 24.0) + 55.0 / 96.0
            } else if ax < 2.5 {
                (2.5 - ax).powi(4) / 24.0
            } else {
                0.0
            }
        }
        _ => {
            if ax < 1.0 {
                let t = ax * ax;
                t * (t * (0.25 - ax / 12.0) - 0.5) + 0.55
            } else if ax < 2.0 {
                ax * (ax * (ax * (ax * (ax / 24.0 - 0.375) + 1.25) - 1.75) + 0.625) + 0.425
            } else if ax < 3.0 {
                (3.0 - ax).powi(5) / 120.0
            } else {
                0.0
            }
        }
    }
}

/// The index of a coefficient, extended beyond `0..n` by the mode's symmetry.
fn coeff_index(i: isize, n: usize, mode: Mode) -> usize {
    let ni = n as isize;
    if (0..ni).contains(&i) {
        return i as usize;
    }
    if n == 1 {
        return 0;
    }
    match mode {
        Mode::Wrap => i.rem_euclid(ni) as usize,
        Mode::Reflect => {
            let m = i.rem_euclid(2 * ni);
            (if m < ni { m } else { 2 * ni - 1 - m }) as usize
        }
        Mode::Nearest => i.clamp(0, ni - 1) as usize,
        _ => {
            let p = 2 * ni - 2;
            let m = i.rem_euclid(p);
            (if m < ni { m } else { p - m }) as usize
        }
    }
}

/// Interpolation weights and first index along one axis for a coordinate `x`.
fn weights(x: f64, order: usize) -> (isize, Vec<f64>) {
    if order == 0 {
        // Nearest pixel, with halves rounded up as in SciPy.
        return ((x + 0.5).floor() as isize, vec![1.0]);
    }
    let start = if order % 2 == 1 { x.floor() as isize - (order / 2) as isize } else { (x + 0.5).floor() as isize - (order / 2) as isize };
    let w = (0..=order).map(|j| bspline(order, x - (start + j as isize) as f64)).collect();
    (start, w)
}

// ANCHOR: map_coordinates
/// Interpolates `image` at arbitrary points `(row, column)` (SciPy's `map_coordinates`), with
/// spline order 0 to 5. Coefficients from [`spline_filter`] are combined with B-spline weights over
/// the `(k + 1) × (k + 1)` neighbourhood of each point. In `Constant` mode, points outside the
/// image (beyond `−0.5 … n − 0.5`… precisely, outside `[0, n − 1]`) take the constant, as in SciPy.
pub fn map_coordinates(image: &Array2<f64>, points: &[(f64, f64)], order: usize, mode: Mode) -> Result<Vec<f64>, ErrorsJSL> {
    if image.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("image must be non-empty"));
    }
    let (coeffs, pad) = spline_filter(image, order, mode)?;
    let (rows, cols) = image.dim();
    let (cr, cc) = coeffs.dim();
    Ok(points
        .iter()
        .map(|&(r, c)| {
            if let Mode::Constant(k) = mode {
                // SciPy treats coordinates within a small tolerance of the edge as inside.
                let tol = 1e-12;
                if r < -tol || c < -tol || r > rows as f64 - 1.0 + tol || c > cols as f64 - 1.0 + tol {
                    return k;
                }
            }
            let (r, c) = (r + pad as f64, c + pad as f64);
            let (sr, wr) = weights(r, order);
            let (sc, wc) = weights(c, order);
            let mut v = 0.0;
            for (i, a) in wr.iter().enumerate() {
                if *a == 0.0 {
                    continue;
                }
                let ri = coeff_index(sr + i as isize, cr, mode);
                for (j, b) in wc.iter().enumerate() {
                    v += a * b * coeffs[(ri, coeff_index(sc + j as isize, cc, mode))];
                }
            }
            v
        })
        .collect())
}
// ANCHOR_END: map_coordinates

// ANCHOR: affine_transform
/// The affine transform (SciPy's `affine_transform`): output pixel `o` takes the input value at
/// `matrix · o + offset`, interpolated. Note the direction: the matrix maps *output* coordinates to
/// *input* coordinates, so to rotate an image by `θ` the matrix is the rotation by `−θ`.
pub fn affine_transform(image: &Array2<f64>, matrix: [[f64; 2]; 2], offset: (f64, f64), output_shape: (usize, usize), order: usize, mode: Mode) -> Result<Array2<f64>, ErrorsJSL> {
    let (h, w) = output_shape;
    let points: Vec<(f64, f64)> = (0..h * w)
        .map(|k| {
            let (r, c) = ((k / w) as f64, (k % w) as f64);
            (matrix[0][0] * r + matrix[0][1] * c + offset.0, matrix[1][0] * r + matrix[1][1] * c + offset.1)
        })
        .collect();
    let v = map_coordinates(image, &points, order, mode)?;
    Array2::from_shape_vec((h, w), v).map_err(|_| ErrorsJSL::RuntimeError("shape mismatch"))
}
// ANCHOR_END: affine_transform

/// Shifts the image by `(rows, columns)` pixels: output `o` takes input `o − shift`.
pub fn shift(image: &Array2<f64>, by: (f64, f64), order: usize, mode: Mode) -> Result<Array2<f64>, ErrorsJSL> {
    affine_transform(image, [[1.0, 0.0], [0.0, 1.0]], (-by.0, -by.1), image.dim(), order, mode)
}

/// Resizes the image by factors `(rows, columns)` (SciPy's `zoom` with `grid_mode=False`): the
/// output has `round(n · factor)` pixels along each axis, and its first and last pixels sample the
/// input's first and last pixels exactly.
pub fn zoom(image: &Array2<f64>, factor: (f64, f64), order: usize, mode: Mode) -> Result<Array2<f64>, ErrorsJSL> {
    let (rows, cols) = image.dim();
    let out = ((rows as f64 * factor.0).round() as usize, (cols as f64 * factor.1).round() as usize);
    if out.0 == 0 || out.1 == 0 {
        return Err(ErrorsJSL::InvalidInputRange("the zoomed image would be empty"));
    }
    let step = |n: usize, m: usize| if m > 1 { (n as f64 - 1.0) / (m as f64 - 1.0) } else { 0.0 };
    affine_transform(image, [[step(rows, out.0), 0.0], [0.0, step(cols, out.1)]], (0.0, 0.0), out, order, mode)
}

// ANCHOR: rotate
/// Rotates the image by `angle` degrees about its centre (SciPy's `rotate`, axes `(1, 0)`): a
/// positive angle turns the content anticlockwise on the screen. With `reshape` the output grows
/// to hold the whole rotated image.
pub fn rotate(image: &Array2<f64>, angle: f64, reshape: bool, order: usize, mode: Mode) -> Result<Array2<f64>, ErrorsJSL> {
    let (rows, cols) = image.dim();
    let (s, c) = angle.to_radians().sin_cos();
    let m = [[c, s], [-s, c]];
    let out = if reshape {
        let corners = [(0.0, 0.0), (0.0, cols as f64), (rows as f64, 0.0), (rows as f64, cols as f64)];
        let tr: Vec<(f64, f64)> = corners.iter().map(|&(y, x)| (m[0][0] * y + m[0][1] * x, m[1][0] * y + m[1][1] * x)).collect();
        let span = |f: fn(&(f64, f64)) -> f64| {
            let v: Vec<f64> = tr.iter().map(f).collect();
            v.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - v.iter().cloned().fold(f64::INFINITY, f64::min)
        };
        ((span(|p| p.0) + 0.5) as usize, (span(|p| p.1) + 0.5) as usize)
    } else {
        (rows, cols)
    };
    let oc = ((out.0 as f64 - 1.0) / 2.0, (out.1 as f64 - 1.0) / 2.0);
    let ic = ((rows as f64 - 1.0) / 2.0, (cols as f64 - 1.0) / 2.0);
    let offset = (ic.0 - (m[0][0] * oc.0 + m[0][1] * oc.1), ic.1 - (m[1][0] * oc.0 + m[1][1] * oc.1));
    affine_transform(image, m, offset, out, order, mode)
}
// ANCHOR_END: rotate

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splines_interpolate_the_samples() {
        let img = Array2::from_shape_fn((6, 7), |(r, c)| ((r * 7 + c) as f64).sin());
        for order in 0..=5 {
            for mode in [Mode::Mirror, Mode::Reflect, Mode::Nearest, Mode::Wrap, Mode::Constant(0.0)] {
                let pts: Vec<(f64, f64)> = (0..6).flat_map(|r| (0..7).map(move |c| (r as f64, c as f64))).collect();
                let v = map_coordinates(&img, &pts, order, mode).unwrap();
                for (k, &(r, c)) in pts.iter().enumerate() {
                    assert!((v[k] - img[(r as usize, c as usize)]).abs() < 1e-10, "order {order} {mode:?} at ({r},{c}): {}", v[k]);
                }
            }
        }
    }

    #[test]
    fn cubic_splines_reproduce_quadratics_in_the_interior() {
        let f = |r: f64, c: f64| 0.3 * r * r - 0.2 * r * c + c;
        let img = Array2::from_shape_fn((30, 30), |(r, c)| f(r as f64, c as f64));
        let v = map_coordinates(&img, &[(14.3, 15.7)], 3, Mode::Mirror).unwrap();
        assert!((v[0] - f(14.3, 15.7)).abs() < 1e-6, "{} vs {}", v[0], f(14.3, 15.7));
    }

    #[test]
    fn a_quarter_turn_permutes_the_pixels() {
        let img = Array2::from_shape_fn((4, 6), |(r, c)| (r * 6 + c) as f64);
        let r = rotate(&img, 90.0, true, 1, Mode::Constant(0.0)).unwrap();
        assert_eq!(r.dim(), (6, 4));
        // Anticlockwise: the top-right input pixel becomes the top-left output pixel.
        assert!((r[(0, 0)] - img[(0, 5)]).abs() < 1e-9, "{}", r[(0, 0)]);
    }
}
