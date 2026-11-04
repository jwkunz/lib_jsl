//! Speeded-Up Robust Features (SURF).
//!
//! SURF is a faster relative of SIFT built on the *integral image*, in which the sum of any
//! axis-aligned rectangle costs four lookups regardless of its size. Following Bay et al. (2008):
//!
//! 1. **Detection.** The determinant of the Hessian `Dxx Dyy − (0.9 Dxy)²` is approximated with box
//!    filters of increasing size (9, 15, 21, 27, then 15, 27, 39, 51, … in octaves). Instead of
//!    shrinking the image, SURF *grows the filter*, so each scale costs the same. Local maxima over
//!    position and scale are refined with a quadratic fit.
//! 2. **Orientation.** Haar-wavelet responses in a circle of radius `6s` are weighted by a Gaussian
//!    and a `π/3` window is slid around the circle; the window with the longest summed response
//!    gives the dominant orientation.
//! 3. **Descriptor.** A `20s × 20s` square aligned with that orientation is divided into 4×4
//!    subregions of 5×5 Haar samples; each subregion contributes `(Σdx, Σdy, Σ|dx|, Σ|dy|)`, giving
//!    a 64-vector normalised to unit length. (`upright` skips orientation for speed when images are
//!    not rotated.)
//!
//! The image is normalised to `[0, 1]` first, so the detection threshold is relative to its dynamic
//! range. The sign of the Laplacian (`Dxx + Dyy`) is stored so bright blobs on dark and dark on
//! bright are never matched.
//!
//! See the book entry [Local features](https://jwkunz.github.io/lib_jsl/book/image/features.html#surf-and-the-integral-image).

use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;
use std::f64::consts::{PI, TAU};

/// An integral image (summed-area table): `box_sum` of any rectangle in four lookups.
#[derive(Clone, Debug, PartialEq)]
pub struct IntegralImage {
    /// `(rows + 1) × (cols + 1)`; `table[(r, c)]` is the sum of `image[..r, ..c]`.
    table: Array2<f64>,
}

impl IntegralImage {
    pub fn new(image: &Array2<f64>) -> IntegralImage {
        let (rows, cols) = image.dim();
        let mut table = Array2::<f64>::zeros((rows + 1, cols + 1));
        for r in 0..rows {
            let mut running = 0.0;
            for c in 0..cols {
                running += image[(r, c)];
                table[(r + 1, c + 1)] = table[(r, c + 1)] + running;
            }
        }
        IntegralImage { table }
    }

    pub fn rows(&self) -> usize {
        self.table.nrows() - 1
    }

    pub fn cols(&self) -> usize {
        self.table.ncols() - 1
    }

    /// Sum of the `height × width` rectangle whose top-left pixel is `(row, col)`; the part outside
    /// the image contributes zero (so `row`/`col` may be negative).
    pub fn box_sum(&self, row: isize, col: isize, height: usize, width: usize) -> f64 {
        let (rows, cols) = (self.rows() as isize, self.cols() as isize);
        let r0 = row.clamp(0, rows) as usize;
        let c0 = col.clamp(0, cols) as usize;
        let r1 = (row + height as isize).clamp(0, rows) as usize;
        let c1 = (col + width as isize).clamp(0, cols) as usize;
        if r1 <= r0 || c1 <= c0 {
            return 0.0;
        }
        self.table[(r1, c1)] - self.table[(r0, c1)] - self.table[(r1, c0)] + self.table[(r0, c0)]
    }

    /// Haar wavelet response along x (right half minus left half) of a `size × size` window
    /// centred on `(row, col)`.
    fn haar_x(&self, row: isize, col: isize, size: usize) -> f64 {
        let h = (size / 2) as isize;
        self.box_sum(row - h, col, size, size / 2) - self.box_sum(row - h, col - h, size, size / 2)
    }

    /// Haar wavelet response along y (lower half minus upper half).
    fn haar_y(&self, row: isize, col: isize, size: usize) -> f64 {
        let h = (size / 2) as isize;
        self.box_sum(row, col - h, size / 2, size) - self.box_sum(row - h, col - h, size / 2, size)
    }
}

/// SURF parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfParams {
    /// Number of octaves (default 3).
    pub octaves: usize,
    /// Filter sizes per octave (default 4).
    pub intervals: usize,
    /// Pixel step between response samples in the first octave (2 by default; 1 is denser and slower).
    pub init_sample: usize,
    /// Minimum Hessian-determinant response, for an image normalised to `[0, 1]` (default 0.0004).
    pub threshold: f64,
    /// Skip orientation assignment (U-SURF): faster, valid when the image is not rotated.
    pub upright: bool,
}

impl Default for SurfParams {
    fn default() -> Self {
        SurfParams { octaves: 3, intervals: 4, init_sample: 2, threshold: 4e-4, upright: false }
    }
}

/// A SURF keypoint with its descriptor.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfKeypoint {
    /// Column (sub-pixel).
    pub x: f64,
    /// Row (sub-pixel).
    pub y: f64,
    /// Scale `σ` in pixels (`1.2 · filter size / 9`).
    pub scale: f64,
    /// Orientation in radians, in `[0, 2π)` (0 when `upright`).
    pub orientation: f64,
    /// `+1` if the Laplacian is positive (bright blob on dark), `-1` otherwise.
    pub laplacian_sign: i8,
    /// The Hessian-determinant response.
    pub response: f64,
    /// The 64-element unit descriptor.
    pub descriptor: Vec<f64>,
}

struct ResponseLayer {
    filter: usize,
    step: usize,
    rows: usize,
    cols: usize,
    response: Array2<f64>,
    laplacian: Array2<i8>,
}

impl ResponseLayer {
    fn build(integral: &IntegralImage, filter: usize, step: usize) -> ResponseLayer {
        let (rows, cols) = (integral.rows() / step, integral.cols() / step);
        let (b, l) = ((filter as isize - 1) / 2, (filter / 3) as isize);
        let w = filter;
        let inv_area = 1.0 / (w * w) as f64;
        let mut response = Array2::<f64>::zeros((rows, cols));
        let mut laplacian = Array2::<i8>::zeros((rows, cols));
        for ar in 0..rows {
            for ac in 0..cols {
                let (r, c) = ((ar * step) as isize, (ac * step) as isize);
                let ls = l as usize;
                // ANCHOR: box_hessian
                let dxx = (integral.box_sum(r - l + 1, c - b, 2 * ls - 1, w) - 3.0 * integral.box_sum(r - l + 1, c - l / 2, 2 * ls - 1, ls)) * inv_area;
                let dyy = (integral.box_sum(r - b, c - l + 1, w, 2 * ls - 1) - 3.0 * integral.box_sum(r - l / 2, c - l + 1, ls, 2 * ls - 1)) * inv_area;
                let dxy = (integral.box_sum(r - l, c + 1, ls, ls) + integral.box_sum(r + 1, c - l, ls, ls) - integral.box_sum(r - l, c - l, ls, ls) - integral.box_sum(r + 1, c + 1, ls, ls)) * inv_area;
                response[(ar, ac)] = dxx * dyy - 0.81 * dxy * dxy;
                laplacian[(ar, ac)] = if dxx + dyy >= 0.0 { 1 } else { -1 };
                // ANCHOR_END: box_hessian
            }
        }
        ResponseLayer { filter, step, rows, cols, response, laplacian }
    }
}

fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: [[f64; 3]; 3]| m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let d = det(a);
    if d.abs() < 1e-30 {
        return None;
    }
    let mut x = [0.0; 3];
    for (col, xc) in x.iter_mut().enumerate() {
        let mut m = a;
        for row in 0..3 {
            m[row][col] = b[row];
        }
        *xc = det(m) / d;
    }
    Some(x)
}

/// Interpolated extremum position (in pixels) and scale of a response-layer maximum.
fn interpolate(layers: [&ResponseLayer; 3], r: usize, c: usize) -> Option<(f64, f64, f64)> {
    let [bot, mid, top] = layers;
    let v = mid.response[(r, c)];
    let g = [
        (mid.response[(r, c + 1)] - mid.response[(r, c - 1)]) / 2.0,
        (mid.response[(r + 1, c)] - mid.response[(r - 1, c)]) / 2.0,
        (top.response[(r, c)] - bot.response[(r, c)]) / 2.0,
    ];
    let dxx = mid.response[(r, c + 1)] + mid.response[(r, c - 1)] - 2.0 * v;
    let dyy = mid.response[(r + 1, c)] + mid.response[(r - 1, c)] - 2.0 * v;
    let dss = top.response[(r, c)] + bot.response[(r, c)] - 2.0 * v;
    let dxy = (mid.response[(r + 1, c + 1)] - mid.response[(r + 1, c - 1)] - mid.response[(r - 1, c + 1)] + mid.response[(r - 1, c - 1)]) / 4.0;
    let dxs = (top.response[(r, c + 1)] - top.response[(r, c - 1)] - bot.response[(r, c + 1)] + bot.response[(r, c - 1)]) / 4.0;
    let dys = (top.response[(r + 1, c)] - top.response[(r - 1, c)] - bot.response[(r + 1, c)] + bot.response[(r - 1, c)]) / 4.0;
    let x = solve3([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]], [-g[0], -g[1], -g[2]])?;
    if x.iter().all(|o| o.abs() < 1.0) {
        let step = mid.step as f64;
        let filter = mid.filter as f64 + x[2] * (mid.filter as f64 - bot.filter as f64);
        Some(((c as f64 + x[0]) * step, (r as f64 + x[1]) * step, 1.2 / 9.0 * filter))
    } else {
        None
    }
}

fn dominant_orientation(integral: &IntegralImage, x: f64, y: f64, scale: f64) -> f64 {
    let (xi, yi, s) = (x.round() as isize, y.round() as isize, scale.round().max(1.0) as isize);
    let size = (4.0 * scale).round().max(2.0) as usize;
    let mut samples: Vec<(f64, f64, f64)> = Vec::new(); // (dx, dy, polar angle)
    for i in -6isize..=6 {
        for j in -6isize..=6 {
            if i * i + j * j < 36 {
                let weight = (-((i * i + j * j) as f64) / (2.0 * 2.5 * 2.5)).exp() / (2.0 * PI * 2.5 * 2.5).sqrt();
                let (row, col) = (yi + j * s, xi + i * s);
                let (dx, dy) = (weight * integral.haar_x(row, col, size), weight * integral.haar_y(row, col, size));
                samples.push((dx, dy, dy.atan2(dx)));
            }
        }
    }
    let (mut best_len, mut best_angle) = (-1.0, 0.0);
    let mut theta = 0.0;
    while theta < TAU {
        let (mut sx, mut sy) = (0.0, 0.0);
        for &(dx, dy, ang) in &samples {
            // Angular distance on the circle, window half-width pi/6.
            let diff = (ang - theta + PI).rem_euclid(TAU) - PI;
            if diff.abs() < PI / 6.0 {
                sx += dx;
                sy += dy;
            }
        }
        let len = sx * sx + sy * sy;
        if len > best_len {
            best_len = len;
            best_angle = sy.atan2(sx);
        }
        theta += 0.15;
    }
    best_angle.rem_euclid(TAU)
}

fn compute_descriptor(integral: &IntegralImage, x: f64, y: f64, scale: f64, orientation: f64) -> Vec<f64> {
    let (cos_t, sin_t) = (orientation.cos(), orientation.sin());
    let haar_size = (2.0 * scale).round().max(2.0) as usize;
    let mut desc = Vec::with_capacity(64);
    for sub_v in 0..4 {
        for sub_u in 0..4 {
            let (mut sdu, mut sdv, mut sadu, mut sadv) = (0.0, 0.0, 0.0, 0.0);
            for v_idx in 0..5 {
                for u_idx in 0..5 {
                    // Sample position in the keypoint frame, in units of the scale: -10 .. +10.
                    let u = (sub_u as f64 * 5.0 + u_idx as f64 - 9.5) * scale;
                    let v = (sub_v as f64 * 5.0 + v_idx as f64 - 9.5) * scale;
                    let px = x + u * cos_t - v * sin_t;
                    let py = y + u * sin_t + v * cos_t;
                    let (rx, ry) = (integral.haar_x(py.round() as isize, px.round() as isize, haar_size), integral.haar_y(py.round() as isize, px.round() as isize, haar_size));
                    // Express the response in the keypoint frame and weight by a Gaussian about the centre.
                    let (du, dv) = (rx * cos_t + ry * sin_t, -rx * sin_t + ry * cos_t);
                    let r2 = (u / scale).powi(2) + (v / scale).powi(2);
                    let w = (-r2 / (2.0 * 3.3 * 3.3)).exp();
                    sdu += w * du;
                    sdv += w * dv;
                    sadu += w * du.abs();
                    sadv += w * dv.abs();
                }
            }
            desc.extend([sdu, sdv, sadu, sadv]);
        }
    }
    let norm = desc.iter().map(|d| d * d).sum::<f64>().sqrt();
    if norm > 1e-12 {
        desc.iter_mut().for_each(|d| *d /= norm);
    }
    desc
}

/// Detects SURF keypoints and computes their descriptors.
pub fn surf(image: &Array2<f64>, params: &SurfParams) -> Result<Vec<SurfKeypoint>, ErrorsJSL> {
    if params.octaves < 1 || params.intervals < 3 || params.init_sample < 1 || !(params.threshold >= 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Invalid SURF parameters"));
    }
    let (rows, cols) = image.dim();
    if rows < 32 || cols < 32 {
        return Err(ErrorsJSL::InvalidInputRange("Image must be at least 32x32"));
    }
    if image.iter().any(|v| !v.is_finite()) {
        return Err(ErrorsJSL::InvalidInputRange("Image must be finite"));
    }
    let (lo, hi) = image.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
    if hi - lo <= 0.0 {
        return Ok(Vec::new());
    }
    let integral = IntegralImage::new(&image.mapv(|v| (v - lo) / (hi - lo)));

    let mut keypoints = Vec::new();
    let mut start = 9usize;
    for o in 0..params.octaves {
        let spacing = 6 * (1usize << o);
        let step = params.init_sample * (1usize << o);
        let filters: Vec<usize> = (0..params.intervals).map(|i| start + i * spacing).collect();
        start += spacing;
        if filters.last().is_none_or(|&f| f >= rows.min(cols)) {
            break; // the filter would not fit in the image
        }
        let layers: Vec<ResponseLayer> = filters.iter().map(|&f| ResponseLayer::build(&integral, f, step)).collect();
        for t in 1..layers.len() - 1 {
            let (bot, mid, top) = (&layers[t - 1], &layers[t], &layers[t + 1]);
            let border = top.filter.div_ceil(2) / step + 1;
            for r in border..mid.rows.saturating_sub(border) {
                for c in border..mid.cols.saturating_sub(border) {
                    let v = mid.response[(r, c)];
                    if v <= params.threshold {
                        continue;
                    }
                    let mut is_max = true;
                    'check: for (li, layer) in [bot, mid, top].iter().enumerate() {
                        for dr in -1isize..=1 {
                            for dc in -1isize..=1 {
                                if li == 1 && dr == 0 && dc == 0 {
                                    continue;
                                }
                                if layer.response[((r as isize + dr) as usize, (c as isize + dc) as usize)] >= v {
                                    is_max = false;
                                    break 'check;
                                }
                            }
                        }
                    }
                    if !is_max {
                        continue;
                    }
                    let Some((x, y, scale)) = interpolate([bot, mid, top], r, c) else { continue };
                    let orientation = if params.upright { 0.0 } else { dominant_orientation(&integral, x, y, scale) };
                    keypoints.push(SurfKeypoint {
                        x,
                        y,
                        scale,
                        orientation,
                        laplacian_sign: mid.laplacian[(r, c)],
                        response: v,
                        descriptor: compute_descriptor(&integral, x, y, scale, orientation),
                    });
                }
            }
        }
    }
    Ok(keypoints)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::image::feature_extraction::match_descriptors;
    use crate::discrete::image::filters::gaussian_blur::gaussian_blur;

    fn texture(n: usize, seed0: u64) -> Array2<f64> {
        let mut seed = seed0;
        let noise = Array2::from_shape_fn((n, n), |_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as f64 / (1u64 << 31) as f64
        });
        gaussian_blur(&noise, 2.0).unwrap()
    }

    fn descriptors(k: &[SurfKeypoint]) -> Vec<Vec<f64>> {
        k.iter().map(|kp| kp.descriptor.clone()).collect()
    }

    fn dense() -> SurfParams {
        SurfParams { init_sample: 1, ..SurfParams::default() }
    }

    #[test]
    fn integral_image_box_sums_are_exact_including_clipped_boxes() {
        let img = Array2::from_shape_fn((7, 9), |(r, c)| (r * 9 + c) as f64 + 1.0);
        let ii = IntegralImage::new(&img);
        let naive = |r0: isize, c0: isize, h: usize, w: usize| -> f64 {
            let mut s = 0.0;
            for r in r0..r0 + h as isize {
                for c in c0..c0 + w as isize {
                    if (0..7).contains(&r) && (0..9).contains(&c) {
                        s += img[(r as usize, c as usize)];
                    }
                }
            }
            s
        };
        for (r0, c0, h, w) in [(0, 0, 7, 9), (2, 3, 3, 4), (-2, -3, 5, 6), (5, 7, 10, 10), (3, 3, 0, 4), (-5, -5, 2, 2)] {
            assert!((ii.box_sum(r0, c0, h, w) - naive(r0, c0, h, w)).abs() < 1e-9, "box ({r0},{c0}) {h}x{w}");
        }
        assert_eq!((ii.rows(), ii.cols()), (7, 9));
    }

    #[test]
    fn haar_responses_detect_the_direction_of_a_ramp() {
        let ii = IntegralImage::new(&Array2::from_shape_fn((40, 40), |(r, c)| 2.0 * c as f64 + 1.0 * r as f64));
        let (hx, hy) = (ii.haar_x(20, 20, 8), ii.haar_y(20, 20, 8));
        assert!(hx > 0.0 && hy > 0.0 && hx > hy, "x gradient is twice the y gradient: {hx} {hy}");
        assert!((hx / hy - 2.0).abs() < 1e-9);
    }

    #[test]
    fn a_constant_image_has_no_features() {
        assert!(surf(&Array2::from_elem((64, 64), 0.3), &SurfParams::default()).unwrap().is_empty());
    }

    #[test]
    fn a_gaussian_blob_is_found_at_its_centre_with_a_matching_scale() {
        let s = 6.0;
        let img = Array2::from_shape_fn((100, 100), |(r, c)| (-((r as f64 - 50.0).powi(2) + (c as f64 - 50.0).powi(2)) / (2.0 * s * s)).exp());
        let kps = surf(&img, &dense()).unwrap();
        assert!(!kps.is_empty());
        let best = kps.iter().max_by(|a, b| a.response.partial_cmp(&b.response).unwrap()).unwrap();
        assert!((best.x - 50.0).abs() < 1.5 && (best.y - 50.0).abs() < 1.5, "({}, {})", best.x, best.y);
        assert!((best.scale / s - 1.0).abs() < 0.4, "scale {} vs blob std {s}", best.scale);
        assert_eq!(best.laplacian_sign, -1, "a bright blob has a negative Laplacian of intensity");
        assert!((best.descriptor.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn descriptors_are_64_dimensional_unit_vectors() {
        let kps = surf(&texture(128, 1), &SurfParams::default()).unwrap();
        assert!(kps.len() > 20, "{} keypoints", kps.len());
        for k in &kps {
            assert_eq!(k.descriptor.len(), 64);
            assert!((k.descriptor.iter().map(|v| v * v).sum::<f64>().sqrt() - 1.0).abs() < 1e-9);
            assert!((0.0..TAU).contains(&k.orientation) && k.x >= 0.0 && k.y >= 0.0 && k.x < 128.0 && k.y < 128.0);
            assert!(k.laplacian_sign == 1 || k.laplacian_sign == -1);
        }
        let upright = surf(&texture(128, 1), &SurfParams { upright: true, ..SurfParams::default() }).unwrap();
        assert!(upright.iter().all(|k| k.orientation == 0.0));
    }

    #[test]
    fn invariant_to_translation() {
        let t = texture(176, 2);
        let a = t.slice(ndarray::s![16..144, 16..144]).to_owned();
        let b = t.slice(ndarray::s![23..151, 11..139]).to_owned();
        let (ka, kb) = (surf(&a, &dense()).unwrap(), surf(&b, &dense()).unwrap());
        let matches = match_descriptors(&descriptors(&ka), &descriptors(&kb), 0.8, true).unwrap();
        assert!(matches.len() >= 12, "only {} matches from {} and {} keypoints", matches.len(), ka.len(), kb.len());
        let consistent = matches.iter().filter(|m| (kb[m.index_b].x - ka[m.index_a].x - 5.0).abs() < 2.0 && (kb[m.index_b].y - ka[m.index_a].y + 7.0).abs() < 2.0).count();
        assert!(consistent as f64 >= 0.8 * matches.len() as f64, "{consistent} of {} follow the shift", matches.len());
    }

    #[test]
    fn invariant_to_a_quarter_turn_rotation() {
        let n = 128;
        let a = texture(n, 3);
        let b = Array2::from_shape_fn((n, n), |(r, c)| a[(c, n - 1 - r)]);
        let (ka, kb) = (surf(&a, &dense()).unwrap(), surf(&b, &dense()).unwrap());
        let matches = match_descriptors(&descriptors(&ka), &descriptors(&kb), 0.8, true).unwrap();
        assert!(matches.len() >= 12, "only {} matches from {} and {} keypoints", matches.len(), ka.len(), kb.len());
        let positional = matches.iter().filter(|m| (kb[m.index_b].x - ka[m.index_a].y).abs() < 2.5 && (kb[m.index_b].y - (n as f64 - 1.0 - ka[m.index_a].x)).abs() < 2.5).count();
        assert!(positional as f64 >= 0.8 * matches.len() as f64, "{positional} of {} map correctly", matches.len());
        // Matching keypoints keep their Laplacian sign.
        assert!(matches.iter().filter(|m| ka[m.index_a].laplacian_sign == kb[m.index_b].laplacian_sign).count() as f64 >= 0.9 * matches.len() as f64);
    }

    #[test]
    fn invariant_to_contrast_changes_because_the_image_is_normalised() {
        let a = texture(128, 4);
        let b = a.mapv(|v| 0.4 * v + 0.3);
        let (ka, kb) = (surf(&a, &SurfParams::default()).unwrap(), surf(&b, &SurfParams::default()).unwrap());
        assert_eq!(ka.len(), kb.len());
        let matches = match_descriptors(&descriptors(&ka), &descriptors(&kb), 0.8, true).unwrap();
        assert!(matches.len() as f64 >= 0.9 * ka.len() as f64);
        assert!(matches.iter().all(|m| m.distance < 1e-6));
    }

    #[test]
    fn a_higher_threshold_keeps_fewer_keypoints() {
        let t = texture(128, 5);
        let base = surf(&t, &SurfParams::default()).unwrap().len();
        let strict = surf(&t, &SurfParams { threshold: 4e-3, ..SurfParams::default() }).unwrap().len();
        assert!(strict < base, "{strict} vs {base}");
    }

    #[test]
    fn invalid_inputs() {
        assert!(surf(&Array2::<f64>::zeros((20, 64)), &SurfParams::default()).is_err());
        let mut img = texture(64, 6);
        img[(2, 2)] = f64::INFINITY;
        assert!(surf(&img, &SurfParams::default()).is_err());
        let ok = texture(64, 6);
        assert!(surf(&ok, &SurfParams { octaves: 0, ..SurfParams::default() }).is_err());
        assert!(surf(&ok, &SurfParams { intervals: 2, ..SurfParams::default() }).is_err());
        assert!(surf(&ok, &SurfParams { init_sample: 0, ..SurfParams::default() }).is_err());
        assert!(surf(&ok, &SurfParams { threshold: -1.0, ..SurfParams::default() }).is_err());
    }
}
