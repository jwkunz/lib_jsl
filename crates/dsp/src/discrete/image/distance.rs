//! Distance transforms: for every foreground pixel, the distance to the nearest background pixel.
//!
//! The exact Euclidean transform uses the separable algorithm of Felzenszwalb and Huttenlocher
//! (2004): the squared distance in two dimensions is a one-dimensional problem along the rows of a
//! one-dimensional answer along the columns, and each one-dimensional problem is the lower envelope
//! of parabolas, found in linear time. The chamfer transforms (city-block and chessboard) need only
//! two raster sweeps.
//!
//! See the book entry [Morphology, distance transforms and labelling](https://jwkunz.github.io/lib_jsl/book/image/morphology.html).

use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;

// ANCHOR: envelope
/// The one-dimensional squared distance transform of `f` with sample spacing `h`:
/// `d[q] = min_p ((q − p)² h² + f[p])`, and the minimising `p`. Each `p` contributes the parabola
/// `(q − p)² h² + f[p]`; the lower envelope of the parabolas is built left to right, discarding any
/// parabola that the newest one undercuts before the previous one takes over.
fn envelope(f: &[f64], h: f64) -> (Vec<f64>, Vec<usize>) {
    let n = f.len();
    let h2 = h * h;
    let mut v = vec![0usize; n]; // positions of the parabolas in the envelope
    let mut z = vec![0.0f64; n + 1]; // boundaries between them
    let mut k = 0usize;
    let finite: Vec<usize> = (0..n).filter(|&p| f[p].is_finite()).collect();
    if finite.is_empty() {
        return (vec![f64::INFINITY; n], vec![usize::MAX; n]);
    }
    v[0] = finite[0];
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for &q in &finite[1..] {
        loop {
            let p = v[k];
            // Where the parabolas from p and q intersect.
            let s = ((f[q] + h2 * (q * q) as f64) - (f[p] + h2 * (p * p) as f64)) / (2.0 * h2 * (q as f64 - p as f64));
            if s <= z[k] {
                if k == 0 {
                    v[0] = q;
                    z[1] = f64::INFINITY;
                    break;
                }
                k -= 1;
            } else {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f64::INFINITY;
                break;
            }
        }
    }
    let mut d = vec![0.0; n];
    let mut idx = vec![0usize; n];
    let mut k = 0;
    for q in 0..n {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let p = v[k];
        d[q] = h2 * ((q as f64 - p as f64).powi(2)) + f[p];
        idx[q] = p;
    }
    (d, idx)
}
// ANCHOR_END: envelope

/// The exact Euclidean distance transform with its feature transform.
#[derive(Clone, Debug, PartialEq)]
pub struct DistanceTransform {
    /// The distance from each pixel to the nearest background pixel (0 on the background).
    pub distances: Array2<f64>,
    /// The `(row, column)` of that nearest background pixel.
    pub nearest: Array2<(usize, usize)>,
}

// ANCHOR: edt
/// The exact Euclidean distance transform of a binary image (SciPy's `distance_transform_edt`):
/// for each foreground (`true`) pixel, the distance to the nearest background (`false`) pixel,
/// with pixel spacing `sampling = (row spacing, column spacing)` (default `(1, 1)`). Background
/// pixels get 0. The image must contain at least one background pixel.
pub fn distance_transform_edt(mask: &Array2<bool>, sampling: Option<(f64, f64)>) -> Result<DistanceTransform, ErrorsJSL> {
    let (rows, cols) = mask.dim();
    let (hr, hc) = sampling.unwrap_or((1.0, 1.0));
    if !(hr > 0.0 && hc > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("sampling must be positive"));
    }
    if mask.iter().all(|&b| b) {
        return Err(ErrorsJSL::InvalidInputRange("the image has no background pixel to measure from"));
    }
    // Pass 1, down each column: squared distance to the nearest background pixel in that column.
    let mut g = Array2::<f64>::zeros((rows, cols));
    let mut grow = Array2::<usize>::zeros((rows, cols));
    for c in 0..cols {
        let f: Vec<f64> = (0..rows).map(|r| if mask[(r, c)] { f64::INFINITY } else { 0.0 }).collect();
        let (d, idx) = envelope(&f, hr);
        for r in 0..rows {
            g[(r, c)] = d[r];
            grow[(r, c)] = idx[r];
        }
    }
    // Pass 2, along each row: combine the column answers.
    let mut distances = Array2::<f64>::zeros((rows, cols));
    let mut nearest = Array2::from_elem((rows, cols), (0usize, 0usize));
    for r in 0..rows {
        let f: Vec<f64> = (0..cols).map(|c| g[(r, c)]).collect();
        let (d, idx) = envelope(&f, hc);
        for c in 0..cols {
            distances[(r, c)] = d[c].sqrt();
            nearest[(r, c)] = (grow[(r, idx[c])], idx[c]);
        }
    }
    Ok(DistanceTransform { distances, nearest })
}
// ANCHOR_END: edt

/// The metric of a chamfer distance transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChamferMetric {
    /// City-block (L1) distance: steps to the four edge neighbours.
    Taxicab,
    /// Chessboard (L∞) distance: steps to all eight neighbours.
    Chessboard,
}

// ANCHOR: cdt
/// The chamfer distance transform (SciPy's `distance_transform_cdt`), in integer steps: a forward
/// raster sweep takes each pixel's distance from its already visited neighbours (up and left), and
/// a backward sweep from the others. Background pixels get 0. Images without background give
/// `None` for every pixel.
pub fn distance_transform_cdt(mask: &Array2<bool>, metric: ChamferMetric) -> Array2<Option<u32>> {
    let (rows, cols) = mask.dim();
    let big = u32::MAX / 2;
    let mut d = Array2::from_shape_fn((rows, cols), |(r, c)| if mask[(r, c)] { big } else { 0 });
    let forward: &[(isize, isize)] = match metric {
        ChamferMetric::Taxicab => &[(-1, 0), (0, -1)],
        ChamferMetric::Chessboard => &[(-1, -1), (-1, 0), (-1, 1), (0, -1)],
    };
    let relax = |d: &mut Array2<u32>, r: usize, c: usize, steps: &[(isize, isize)], sign: isize| {
        for &(dr, dc) in steps {
            let (nr, nc) = (r as isize + sign * dr, c as isize + sign * dc);
            if nr >= 0 && nc >= 0 && (nr as usize) < rows && (nc as usize) < cols {
                let cand = d[(nr as usize, nc as usize)].saturating_add(1);
                if cand < d[(r, c)] {
                    d[(r, c)] = cand;
                }
            }
        }
    };
    for r in 0..rows {
        for c in 0..cols {
            relax(&mut d, r, c, forward, 1);
        }
    }
    for r in (0..rows).rev() {
        for c in (0..cols).rev() {
            relax(&mut d, r, c, forward, -1);
        }
    }
    d.mapv(|v| if v >= big { None } else { Some(v) })
}
// ANCHOR_END: cdt

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distances_from_a_single_background_pixel() {
        let mut mask = Array2::from_elem((7, 9), true);
        mask[(3, 4)] = false;
        let t = distance_transform_edt(&mask, None).unwrap();
        for ((r, c), &d) in t.distances.indexed_iter() {
            let want = ((r as f64 - 3.0).powi(2) + (c as f64 - 4.0).powi(2)).sqrt();
            assert!((d - want).abs() < 1e-12, "({r},{c}): {d} vs {want}");
            assert_eq!(t.nearest[(r, c)], (3, 4));
        }
        let cdt = distance_transform_cdt(&mask, ChamferMetric::Taxicab);
        assert_eq!(cdt[(0, 0)], Some(3 + 4));
        let cdt = distance_transform_cdt(&mask, ChamferMetric::Chessboard);
        assert_eq!(cdt[(0, 0)], Some(4));
    }

    #[test]
    fn anisotropic_sampling_scales_the_axes() {
        let mut mask = Array2::from_elem((5, 5), true);
        mask[(0, 0)] = false;
        let t = distance_transform_edt(&mask, Some((2.0, 0.5))).unwrap();
        assert!((t.distances[(3, 4)] - ((6.0f64).powi(2) + 2.0f64.powi(2)).sqrt()).abs() < 1e-12);
    }
}
