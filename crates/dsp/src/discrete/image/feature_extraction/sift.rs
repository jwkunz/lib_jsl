//! Scale-Invariant Feature Transform (SIFT).
//!
//! SIFT finds distinctive local features that can be matched between images of the same scene
//! despite changes in position, scale, rotation and brightness/contrast. Following Lowe (2004):
//!
//! 1. **Scale space.** The image is blurred by a ladder of Gaussians grouped in *octaves*; each octave
//!    covers a doubling of the blur `σ` with `scales_per_octave` steps, and the next octave starts from
//!    a half-size copy.
//! 2. **Detection.** Subtracting adjacent blurred images gives difference-of-Gaussian (DoG) images, an
//!    approximation of the scale-normalised Laplacian. Local extrema over position *and* scale
//!    (3×3×3 neighbourhoods) are candidate keypoints.
//! 3. **Localisation.** A quadratic fit refines each candidate to sub-pixel, sub-scale accuracy;
//!    low-contrast candidates and candidates on edges (where the 2×2 Hessian is badly conditioned)
//!    are discarded as unstable.
//! 4. **Orientation.** A histogram of gradient directions around the keypoint (Gaussian weighted)
//!    gives one or more dominant orientations; a keypoint is created for each. Describing the
//!    neighbourhood relative to this orientation makes the descriptor rotation invariant.
//! 5. **Descriptor.** The orientation-aligned neighbourhood is divided into a 4×4 grid; each cell
//!    holds an 8-bin gradient histogram (trilinearly interpolated), giving a 128-vector, normalised
//!    and clipped at 0.2 to resist contrast and illumination changes.
//!
//! The input image is first normalised to `[0, 1]`, so the thresholds are relative to the image's
//! own dynamic range and a uniform brightness/contrast change does not alter the result. A
//! constant image has no features.
//!
//! Compared with the original paper, the image is not upsampled first (so the finest features, below
//! about 2 pixels, are missed).

use crate::discrete::image::filters::gaussian_blur::gaussian_blur;
use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;
use std::f64::consts::TAU;

const DESCRIPTOR_WIDTH: usize = 4;
const DESCRIPTOR_BINS: usize = 8;
const ORIENTATION_BINS: usize = 36;
const MAX_REFINE_STEPS: usize = 5;

/// SIFT parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct SiftParams {
    /// Scale steps per octave (3 in Lowe's paper).
    pub scales_per_octave: usize,
    /// Base blur `σ₀` of each octave (1.6).
    pub sigma: f64,
    /// Blur assumed already present in the input (0.5).
    pub initial_blur: f64,
    /// Minimum DoG magnitude (after normalising the image to `[0, 1]`) of a refined keypoint,
    /// scaled by `1 / scales_per_octave` (0.04).
    pub contrast_threshold: f64,
    /// Principal-curvature ratio limit for rejecting edge responses (10).
    pub edge_threshold: f64,
    /// Maximum number of octaves; `None` uses as many as fit (down to 16-pixel images).
    pub max_octaves: Option<usize>,
}

impl Default for SiftParams {
    fn default() -> Self {
        SiftParams { scales_per_octave: 3, sigma: 1.6, initial_blur: 0.5, contrast_threshold: 0.04, edge_threshold: 10.0, max_octaves: None }
    }
}

/// A SIFT keypoint with its descriptor.
#[derive(Clone, Debug, PartialEq)]
pub struct SiftKeypoint {
    /// Column in the input image (sub-pixel).
    pub x: f64,
    /// Row in the input image (sub-pixel).
    pub y: f64,
    /// Detection scale in input pixels (the blur `σ` of the matching DoG layer).
    pub sigma: f64,
    /// Dominant gradient orientation in radians, in `[0, 2π)`, measured from the +column axis
    /// toward the +row axis.
    pub orientation: f64,
    /// Absolute DoG value at the refined extremum (a contrast/strength measure).
    pub response: f64,
    pub octave: usize,
    /// The 128-element descriptor (unit length).
    pub descriptor: Vec<f64>,
}

struct Octave {
    gaussians: Vec<Array2<f64>>,
    dogs: Vec<Array2<f64>>,
}

fn downsample(img: &Array2<f64>) -> Array2<f64> {
    let (rows, cols) = img.dim();
    Array2::from_shape_fn((rows / 2, cols / 2), |(r, c)| img[(2 * r, 2 * c)])
}

fn build_octaves(image: &Array2<f64>, p: &SiftParams) -> Result<Vec<Octave>, ErrorsJSL> {
    let s = p.scales_per_octave;
    let k = 2f64.powf(1.0 / s as f64);
    let (rows, cols) = image.dim();
    let min_dim = rows.min(cols);
    let fit = ((min_dim as f64 / 16.0).log2().floor().max(1.0)) as usize;
    let n_octaves = p.max_octaves.map_or(fit, |m| m.min(fit)).max(1);
    let mut base = gaussian_blur(image, (p.sigma * p.sigma - p.initial_blur * p.initial_blur).sqrt())?;
    let mut octaves = Vec::with_capacity(n_octaves);
    for _ in 0..n_octaves {
        let mut gaussians = vec![base.clone()];
        for i in 1..s + 3 {
            let sigma_prev = p.sigma * k.powi(i as i32 - 1);
            let sigma_inc = (sigma_prev * sigma_prev * (k * k - 1.0)).sqrt();
            gaussians.push(gaussian_blur(&gaussians[i - 1], sigma_inc)?);
        }
        let dogs: Vec<Array2<f64>> = (0..s + 2).map(|i| &gaussians[i + 1] - &gaussians[i]).collect();
        base = downsample(&gaussians[s]);
        octaves.push(Octave { gaussians, dogs });
    }
    Ok(octaves)
}

/// Solves the 3×3 system `A x = b` by Cramer's rule, or `None` if singular.
fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: [[f64; 3]; 3]| m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let d = det(a);
    if d.abs() < 1e-12 {
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

struct Candidate {
    layer: usize,
    row: usize,
    col: usize,
    /// Sub-pixel offsets from the integer position (x = column, y = row, s = layer).
    dx: f64,
    dy: f64,
    ds: f64,
    contrast: f64,
}

/// Refines an extremum of the DoG stack and rejects unstable ones.
fn refine(oct: &Octave, p: &SiftParams, mut layer: usize, mut row: usize, mut col: usize) -> Option<Candidate> {
    let s = p.scales_per_octave;
    let (rows, cols) = oct.dogs[0].dim();
    for _ in 0..MAX_REFINE_STEPS {
        if layer < 1 || layer > s || row < 1 || col < 1 || row + 2 > rows || col + 2 > cols {
            return None;
        }
        let (dm, d0, dp) = (&oct.dogs[layer - 1], &oct.dogs[layer], &oct.dogs[layer + 1]);
        let v = d0[(row, col)];
        let g = [(d0[(row, col + 1)] - d0[(row, col - 1)]) / 2.0, (d0[(row + 1, col)] - d0[(row - 1, col)]) / 2.0, (dp[(row, col)] - dm[(row, col)]) / 2.0];
        let dxx = d0[(row, col + 1)] + d0[(row, col - 1)] - 2.0 * v;
        let dyy = d0[(row + 1, col)] + d0[(row - 1, col)] - 2.0 * v;
        let dss = dp[(row, col)] + dm[(row, col)] - 2.0 * v;
        let dxy = (d0[(row + 1, col + 1)] - d0[(row + 1, col - 1)] - d0[(row - 1, col + 1)] + d0[(row - 1, col - 1)]) / 4.0;
        let dxs = (dp[(row, col + 1)] - dp[(row, col - 1)] - dm[(row, col + 1)] + dm[(row, col - 1)]) / 4.0;
        let dys = (dp[(row + 1, col)] - dp[(row - 1, col)] - dm[(row + 1, col)] + dm[(row - 1, col)]) / 4.0;
        let h = [[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]];
        let x = solve3(h, [-g[0], -g[1], -g[2]])?;
        if x.iter().all(|v| v.abs() < 0.5) {
            let contrast = v + 0.5 * (g[0] * x[0] + g[1] * x[1] + g[2] * x[2]);
            if contrast.abs() * (s as f64) < p.contrast_threshold {
                return None;
            }
            let (tr, det) = (dxx + dyy, dxx * dyy - dxy * dxy);
            let r = p.edge_threshold;
            if det <= 0.0 || r * tr * tr >= (r + 1.0) * (r + 1.0) * det {
                return None;
            }
            return Some(Candidate { layer, row, col, dx: x[0], dy: x[1], ds: x[2], contrast });
        }
        // Move to the neighbouring sample the offset points to and try again.
        let step = |v: f64| v.round() as isize;
        let (nl, nr, nc) = (layer as isize + step(x[2]), row as isize + step(x[1]), col as isize + step(x[0]));
        if nl < 0 || nr < 0 || nc < 0 {
            return None;
        }
        layer = nl as usize;
        row = nr as usize;
        col = nc as usize;
    }
    None
}

fn is_extremum(oct: &Octave, layer: usize, r: usize, c: usize) -> bool {
    let v = oct.dogs[layer][(r, c)];
    let (mut is_max, mut is_min) = (true, true);
    for l in layer - 1..=layer + 1 {
        for rr in r - 1..=r + 1 {
            for cc in c - 1..=c + 1 {
                if (l, rr, cc) == (layer, r, c) {
                    continue;
                }
                let n = oct.dogs[l][(rr, cc)];
                if n >= v {
                    is_max = false;
                }
                if n <= v {
                    is_min = false;
                }
            }
        }
    }
    is_max || is_min
}

/// Dominant gradient orientations around a keypoint, as angles in `[0, 2π)`.
fn dominant_orientations(img: &Array2<f64>, row: usize, col: usize, sigma_oct: f64) -> Vec<f64> {
    let (rows, cols) = img.dim();
    let radius = (3.0 * 1.5 * sigma_oct).round() as isize;
    let denom = 2.0 * (1.5 * sigma_oct).powi(2);
    let mut hist = [0.0f64; ORIENTATION_BINS];
    for dy in -radius..=radius {
        let r = row as isize + dy;
        if r < 1 || r >= rows as isize - 1 {
            continue;
        }
        for dx in -radius..=radius {
            let c = col as isize + dx;
            if c < 1 || c >= cols as isize - 1 {
                continue;
            }
            let (r, c) = (r as usize, c as usize);
            let gx = img[(r, c + 1)] - img[(r, c - 1)];
            let gy = img[(r + 1, c)] - img[(r - 1, c)];
            let w = (-((dx * dx + dy * dy) as f64) / denom).exp();
            let angle = gy.atan2(gx).rem_euclid(TAU);
            let bin = (angle / TAU * ORIENTATION_BINS as f64).round() as usize % ORIENTATION_BINS;
            hist[bin] += w * gx.hypot(gy);
        }
    }
    for _ in 0..6 {
        let prev = hist;
        for j in 0..ORIENTATION_BINS {
            hist[j] = (prev[(j + ORIENTATION_BINS - 1) % ORIENTATION_BINS] + prev[j] + prev[(j + 1) % ORIENTATION_BINS]) / 3.0;
        }
    }
    let max = hist.iter().cloned().fold(0.0, f64::max);
    if max <= 0.0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for j in 0..ORIENTATION_BINS {
        let (l, c, r) = (hist[(j + ORIENTATION_BINS - 1) % ORIENTATION_BINS], hist[j], hist[(j + 1) % ORIENTATION_BINS]);
        if c > l && c > r && c >= 0.8 * max {
            let offset = 0.5 * (l - r) / (l - 2.0 * c + r);
            out.push(((j as f64 + offset) / ORIENTATION_BINS as f64 * TAU).rem_euclid(TAU));
        }
    }
    out
}

/// The 128-D descriptor of the neighbourhood of `(row, col)` aligned with `orientation`.
fn compute_descriptor(img: &Array2<f64>, row: usize, col: usize, sigma_oct: f64, orientation: f64) -> Vec<f64> {
    let (rows, cols) = img.dim();
    let (d, n) = (DESCRIPTOR_WIDTH, DESCRIPTOR_BINS);
    let (cos_t, sin_t) = (orientation.cos(), orientation.sin());
    let hist_width = 3.0 * sigma_oct;
    let radius = ((hist_width * std::f64::consts::SQRT_2 * (d as f64 + 1.0) * 0.5).round() as isize).min((rows.min(cols) / 2) as isize);
    let exp_scale = -1.0 / (d as f64 * d as f64 * 0.5);
    // Padded histogram so interpolation can spill one cell past each edge.
    let mut hist = vec![0.0f64; (d + 2) * (d + 2) * n];
    let idx = |r: usize, c: usize, o: usize| (r * (d + 2) + c) * n + o;
    for i in -radius..=radius {
        for j in -radius..=radius {
            let c_rot = (j as f64 * cos_t + i as f64 * sin_t) / hist_width;
            let r_rot = (-(j as f64) * sin_t + i as f64 * cos_t) / hist_width;
            let rbin = r_rot + d as f64 / 2.0 - 0.5;
            let cbin = c_rot + d as f64 / 2.0 - 0.5;
            if !(rbin > -1.0 && rbin < d as f64 && cbin > -1.0 && cbin < d as f64) {
                continue;
            }
            let (rr, cc) = (row as isize + i, col as isize + j);
            if rr < 1 || rr >= rows as isize - 1 || cc < 1 || cc >= cols as isize - 1 {
                continue;
            }
            let (rr, cc) = (rr as usize, cc as usize);
            let gx = img[(rr, cc + 1)] - img[(rr, cc - 1)];
            let gy = img[(rr + 1, cc)] - img[(rr - 1, cc)];
            let mag = gx.hypot(gy) * (exp_scale * (c_rot * c_rot + r_rot * r_rot)).exp();
            let ori = (gy.atan2(gx) - orientation).rem_euclid(TAU);
            let obin = ori / TAU * n as f64;
            let (r0, c0, o0) = (rbin.floor(), cbin.floor(), obin.floor());
            let (fr, fc, fo) = (rbin - r0, cbin - c0, obin - o0);
            let (r0, c0, o0) = ((r0 as isize + 1) as usize, (c0 as isize + 1) as usize, o0 as usize);
            for (dr, wr) in [(0usize, 1.0 - fr), (1, fr)] {
                for (dc, wc) in [(0usize, 1.0 - fc), (1, fc)] {
                    for (dobin, wo) in [(0usize, 1.0 - fo), (1, fo)] {
                        hist[idx(r0 + dr, c0 + dc, (o0 + dobin) % n)] += mag * wr * wc * wo;
                    }
                }
            }
        }
    }
    let mut desc = Vec::with_capacity(d * d * n);
    for r in 1..=d {
        for c in 1..=d {
            for o in 0..n {
                desc.push(hist[idx(r, c, o)]);
            }
        }
    }
    let normalise = |v: &mut Vec<f64>| {
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
    };
    normalise(&mut desc);
    desc.iter_mut().for_each(|x| *x = x.min(0.2));
    normalise(&mut desc);
    desc
}

/// Detects SIFT keypoints and computes their descriptors.
pub fn sift(image: &Array2<f64>, params: &SiftParams) -> Result<Vec<SiftKeypoint>, ErrorsJSL> {
    if params.scales_per_octave < 1 || !(params.sigma > params.initial_blur) || !(params.contrast_threshold >= 0.0) || !(params.edge_threshold > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Invalid SIFT parameters"));
    }
    let (rows, cols) = image.dim();
    if rows < 16 || cols < 16 {
        return Err(ErrorsJSL::InvalidInputRange("Image must be at least 16x16"));
    }
    if image.iter().any(|v| !v.is_finite()) {
        return Err(ErrorsJSL::InvalidInputRange("Image must be finite"));
    }
    let (lo, hi) = image.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
    if hi - lo <= 0.0 {
        return Ok(Vec::new());
    }
    let normalised = image.mapv(|v| (v - lo) / (hi - lo));
    let octaves = build_octaves(&normalised, params)?;
    let s = params.scales_per_octave;
    let mut keypoints = Vec::new();
    for (o, oct) in octaves.iter().enumerate() {
        let (orows, ocols) = oct.dogs[0].dim();
        let pre = 0.5 * params.contrast_threshold / s as f64;
        for layer in 1..=s {
            for r in 1..orows.saturating_sub(1) {
                for c in 1..ocols.saturating_sub(1) {
                    if oct.dogs[layer][(r, c)].abs() <= pre || !is_extremum(oct, layer, r, c) {
                        continue;
                    }
                    let Some(cand) = refine(oct, params, layer, r, c) else { continue };
                    let sigma_oct = params.sigma * 2f64.powf((cand.layer as f64 + cand.ds) / s as f64);
                    let img = &oct.gaussians[cand.layer];
                    let scale = (1usize << o) as f64;
                    for orientation in dominant_orientations(img, cand.row, cand.col, sigma_oct) {
                        keypoints.push(SiftKeypoint {
                            x: (cand.col as f64 + cand.dx) * scale,
                            y: (cand.row as f64 + cand.dy) * scale,
                            sigma: sigma_oct * scale,
                            orientation,
                            response: cand.contrast.abs(),
                            octave: o,
                            descriptor: compute_descriptor(img, cand.row, cand.col, sigma_oct, orientation),
                        });
                    }
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

    /// A deterministic smooth random texture, rich in blobs and corners.
    fn texture(n: usize, seed0: u64) -> Array2<f64> {
        let mut seed = seed0;
        let noise = Array2::from_shape_fn((n, n), |_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as f64 / (1u64 << 31) as f64
        });
        gaussian_blur(&noise, 2.0).unwrap()
    }

    fn descriptors(k: &[SiftKeypoint]) -> Vec<Vec<f64>> {
        k.iter().map(|kp| kp.descriptor.clone()).collect()
    }

    #[test]
    fn a_constant_image_has_no_features() {
        assert!(sift(&Array2::from_elem((64, 64), 0.4), &SiftParams::default()).unwrap().is_empty());
    }

    #[test]
    fn a_single_gaussian_blob_is_found_at_its_centre_and_scale() {
        let s = 4.0;
        let img = Array2::from_shape_fn((96, 96), |(r, c)| (-((r as f64 - 48.0).powi(2) + (c as f64 - 48.0).powi(2)) / (2.0 * s * s)).exp());
        let kps = sift(&img, &SiftParams::default()).unwrap();
        assert!(!kps.is_empty());
        let best = kps.iter().max_by(|a, b| a.response.partial_cmp(&b.response).unwrap()).unwrap();
        assert!((best.x - 48.0).abs() < 1.0 && (best.y - 48.0).abs() < 1.0, "({}, {})", best.x, best.y);
        assert!((best.sigma / s - 1.0).abs() < 0.3, "sigma {} vs blob std {s}", best.sigma);
        // A radially symmetric blob has no preferred direction; whatever orientation is chosen,
        // the descriptor is a valid unit vector.
        assert!((best.descriptor.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn descriptors_are_unit_length_clipped_and_finite() {
        let kps = sift(&texture(128, 1), &SiftParams::default()).unwrap();
        assert!(kps.len() > 40, "{} keypoints", kps.len());
        for k in &kps {
            assert_eq!(k.descriptor.len(), 128);
            assert!(k.descriptor.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0 + 1e-9));
            let norm = k.descriptor.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-9);
            assert!((0.0..TAU).contains(&k.orientation));
            assert!(k.x >= 0.0 && k.x < 128.0 && k.y >= 0.0 && k.y < 128.0);
        }
    }

    #[test]
    fn invariant_to_translation() {
        let t = texture(176, 2);
        let a = t.slice(ndarray::s![16..144, 16..144]).to_owned();
        let b = t.slice(ndarray::s![23..151, 11..139]).to_owned(); // content moves by (-7 rows, +5 columns)
        let (ka, kb) = (sift(&a, &SiftParams::default()).unwrap(), sift(&b, &SiftParams::default()).unwrap());
        let matches = match_descriptors(&descriptors(&ka), &descriptors(&kb), 0.8, true).unwrap();
        assert!(matches.len() >= 25, "only {} matches from {} and {} keypoints", matches.len(), ka.len(), kb.len());
        let consistent = matches.iter().filter(|m| (kb[m.index_b].x - ka[m.index_a].x - 5.0).abs() < 1.5 && (kb[m.index_b].y - ka[m.index_a].y + 7.0).abs() < 1.5).count();
        assert!(consistent as f64 >= 0.85 * matches.len() as f64, "{consistent} of {} matches follow the shift", matches.len());
    }

    #[test]
    fn invariant_to_a_quarter_turn_rotation_and_reports_the_rotation() {
        let n = 128;
        let a = texture(n, 3);
        // Rotate 90 degrees counter-clockwise: b[r][c] = a[c][n-1-r].
        let b = Array2::from_shape_fn((n, n), |(r, c)| a[(c, n - 1 - r)]);
        let (ka, kb) = (sift(&a, &SiftParams::default()).unwrap(), sift(&b, &SiftParams::default()).unwrap());
        let matches = match_descriptors(&descriptors(&ka), &descriptors(&kb), 0.8, true).unwrap();
        assert!(matches.len() >= 25, "only {} matches", matches.len());
        // A point (x = col, y = row) in a lands at (x', y') = (y, n-1-x) in b.
        let positional = matches.iter().filter(|m| (kb[m.index_b].x - ka[m.index_a].y).abs() < 1.5 && (kb[m.index_b].y - (n as f64 - 1.0 - ka[m.index_a].x)).abs() < 1.5).count();
        assert!(positional as f64 >= 0.85 * matches.len() as f64, "{positional} of {}", matches.len());
        // The orientation changes by the same quarter turn for the consistent matches.
        let diffs: Vec<f64> = matches.iter().map(|m| (kb[m.index_b].orientation - ka[m.index_a].orientation).rem_euclid(TAU)).collect();
        let mut hist = [0usize; 4];
        for d in &diffs {
            hist[((d / (TAU / 4.0)).round() as usize) % 4] += 1;
        }
        let dominant = hist.iter().cloned().max().unwrap();
        assert!(dominant as f64 >= 0.8 * diffs.len() as f64, "orientation differences cluster at a quarter-turn multiple: {hist:?}");
    }

    #[test]
    fn invariant_to_brightness_and_contrast_changes() {
        let a = texture(128, 4);
        let b = a.mapv(|v| 0.35 * v + 0.2);
        let (ka, kb) = (sift(&a, &SiftParams::default()).unwrap(), sift(&b, &SiftParams::default()).unwrap());
        assert_eq!(ka.len(), kb.len(), "the image is normalised, so an affine intensity change is invisible");
        let matches = match_descriptors(&descriptors(&ka), &descriptors(&kb), 0.8, true).unwrap();
        assert!(matches.len() as f64 >= 0.9 * ka.len() as f64);
        assert!(matches.iter().all(|m| m.distance < 1e-6 && (ka[m.index_a].x - kb[m.index_b].x).abs() < 1e-6 && (ka[m.index_a].y - kb[m.index_b].y).abs() < 1e-6));
    }

    #[test]
    fn stricter_thresholds_keep_fewer_keypoints() {
        let t = texture(128, 5);
        let base = sift(&t, &SiftParams::default()).unwrap().len();
        let strict = sift(&t, &SiftParams { contrast_threshold: 0.12, ..SiftParams::default() }).unwrap().len();
        let one_octave = sift(&t, &SiftParams { max_octaves: Some(1), ..SiftParams::default() }).unwrap();
        assert!(strict < base, "{strict} vs {base}");
        assert!(one_octave.iter().all(|k| k.octave == 0));
    }

    #[test]
    fn invalid_inputs() {
        assert!(sift(&Array2::<f64>::zeros((8, 64)), &SiftParams::default()).is_err());
        let mut img = texture(32, 6);
        img[(3, 3)] = f64::NAN;
        assert!(sift(&img, &SiftParams::default()).is_err());
        let ok = texture(32, 6);
        assert!(sift(&ok, &SiftParams { scales_per_octave: 0, ..SiftParams::default() }).is_err());
        assert!(sift(&ok, &SiftParams { sigma: 0.4, ..SiftParams::default() }).is_err());
        assert!(sift(&ok, &SiftParams { edge_threshold: 0.0, ..SiftParams::default() }).is_err());
    }
}
