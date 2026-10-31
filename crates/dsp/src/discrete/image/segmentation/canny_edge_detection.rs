//! Canny edge detection.
//!
//! The Canny detector turns an image into thin, connected, one-pixel-wide edges in four stages:
//!
//! 1. **Smooth** with a Gaussian (scale `sigma`) to suppress noise that would create false edges.
//! 2. **Differentiate** with the Sobel operator to get gradient magnitude and direction.
//! 3. **Thin** by *non-maximum suppression*: keep a pixel only if its magnitude is at least as large
//!    as both neighbours along the gradient direction, i.e. it is on the ridge of the edge.
//! 4. **Link** by *hysteresis thresholding*: pixels above the `high` threshold are certain edges;
//!    pixels between `low` and `high` are kept only if they connect (8-neighbourhood) to a certain
//!    edge. This keeps faint parts of real edges while discarding isolated noise.
//!
//! Thresholds are on the gradient magnitude in intensity units per pixel. The outermost pixel
//! border is never marked.

use crate::discrete::image::filters::{gaussian_blur::gaussian_blur, sobel::sobel};
use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;
use std::collections::VecDeque;

/// Detects edges in `image`: smoothing scale `sigma` (pixels), and hysteresis thresholds `low <=
/// high` on the gradient magnitude. Returns a boolean edge map.
pub fn canny_edge_detection(image: &Array2<f64>, sigma: f64, low: f64, high: f64) -> Result<Array2<bool>, ErrorsJSL> {
    if !(low >= 0.0) || !(high >= low) || !high.is_finite() {
        return Err(ErrorsJSL::InvalidInputRange("Thresholds must satisfy 0 <= low <= high"));
    }
    let blurred = gaussian_blur(image, sigma)?;
    let grad = sobel(&blurred, true)?;
    let (rows, cols) = image.dim();
    let mag = &grad.magnitude;

    // Non-maximum suppression along the (quantised) gradient direction.
    let mut thin = Array2::<f64>::zeros((rows, cols));
    for r in 1..rows.saturating_sub(1) {
        for c in 1..cols.saturating_sub(1) {
            let m = mag[(r, c)];
            if m <= 0.0 {
                continue;
            }
            // Angle folded into [0, 180) degrees.
            let mut deg = grad.direction[(r, c)].to_degrees();
            if deg < 0.0 {
                deg += 180.0;
            }
            if deg >= 180.0 {
                deg -= 180.0;
            }
            let (dr, dc): (isize, isize) = if !(22.5..157.5).contains(&deg) {
                (0, 1) // gradient along x: compare left and right
            } else if deg < 67.5 {
                (1, 1)
            } else if deg < 112.5 {
                (1, 0) // gradient along y: compare up and down
            } else {
                (1, -1)
            };
            let before = mag[((r as isize - dr) as usize, (c as isize - dc) as usize)];
            let after = mag[((r as isize + dr) as usize, (c as isize + dc) as usize)];
            // Strict on one side so a plateau of equal values yields a single ridge pixel.
            if m > before && m >= after {
                thin[(r, c)] = m;
            }
        }
    }

    // Hysteresis: grow from strong pixels through weak ones.
    let mut edges = Array2::from_elem((rows, cols), false);
    let mut queue = VecDeque::new();
    for ((r, c), &m) in thin.indexed_iter() {
        if m >= high && m > 0.0 {
            edges[(r, c)] = true;
            queue.push_back((r, c));
        }
    }
    while let Some((r, c)) = queue.pop_front() {
        for dr in -1isize..=1 {
            for dc in -1isize..=1 {
                let (nr, nc) = (r as isize + dr, c as isize + dc);
                if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                    continue;
                }
                let (nr, nc) = (nr as usize, nc as usize);
                if !edges[(nr, nc)] && thin[(nr, nc)] >= low && thin[(nr, nc)] > 0.0 {
                    edges[(nr, nc)] = true;
                    queue.push_back((nr, nc));
                }
            }
        }
    }
    Ok(edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
    }

    fn square(n: usize, lo: usize, hi: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, n), |(r, c)| if (lo..hi).contains(&r) && (lo..hi).contains(&c) { 1.0 } else { 0.0 })
    }

    #[test]
    fn edges_of_a_square_are_thin_and_on_its_boundary() {
        let img = square(60, 20, 40);
        let e = canny_edge_detection(&img, 1.0, 0.05, 0.15).unwrap();
        // Along a horizontal cut through the middle there are exactly two edge pixels, at the sides.
        let row: Vec<usize> = (0..60).filter(|&c| e[(30, c)]).collect();
        assert_eq!(row.len(), 2, "one edge pixel per side, got {row:?}");
        assert!(row[0].abs_diff(20) <= 1 && row[1].abs_diff(39) <= 1, "{row:?}");
        // Same vertically, and nothing inside or far outside.
        let col: Vec<usize> = (0..60).filter(|&r| e[(r, 30)]).collect();
        assert_eq!(col.len(), 2, "{col:?}");
        assert!(!e[(30, 30)] && !e[(5, 5)] && !e[(55, 55)] && !e[(30, 5)]);
        // The detected set is the right ballpark: roughly the 80-pixel perimeter.
        let total = e.iter().filter(|v| **v).count();
        assert!((70..=110).contains(&total), "{total} edge pixels");
    }

    #[test]
    fn hysteresis_keeps_faint_segments_attached_to_strong_edges_and_drops_isolated_ones() {
        // One vertical edge at column 15 whose contrast fades smoothly from 1.0 (rows <= 10) to
        // 0.3 (rows >= 30): strong at the top, faint at the bottom, continuous in between. (An
        // abrupt change of contrast would create a second, horizontal edge and a gap at the corner.)
        // Separately, an isolated faint edge at column 45.
        let amplitude = |r: usize| (1.0 - 0.7 * (r as f64 - 10.0) / 20.0).clamp(0.3, 1.0);
        let img = Array2::from_shape_fn((40, 60), |(r, c)| {
            let fading = if c >= 15 { amplitude(r) } else { 0.0 };
            let isolated = if c >= 45 && (5..15).contains(&r) { 0.3 } else { 0.0 };
            fading + isolated
        });
        let e = canny_edge_detection(&img, 1.0, 0.04, 0.2).unwrap();
        let count = |rows: std::ops::Range<usize>, cols: std::ops::Range<usize>| rows.flat_map(|r| cols.clone().map(move |c| (r, c))).filter(|&p| e[p]).count();
        assert!(count(3..18, 13..18) >= 12, "the strong part is found");
        assert!(count(26..38, 13..18) >= 10, "the faint part survives through hysteresis: {}", count(26..38, 13..18));
        assert_eq!(count(5..15, 43..48), 0, "the isolated faint edge is rejected");
        // Without hysteresis (low == high) the faint part disappears but the strong part remains.
        let strict = canny_edge_detection(&img, 1.0, 0.2, 0.2).unwrap();
        assert!((26..38).flat_map(|r| (13..18).map(move |c| (r, c))).all(|p| !strict[p]));
        assert!((3..15).flat_map(|r| (13..18).map(move |c| (r, c))).filter(|&p| strict[p]).count() >= 10);
    }

    #[test]
    fn noise_alone_produces_few_edges_with_sensible_thresholds_and_the_scene_survives_noise() {
        let mut seed = 5;
        let mut noise = |s: f64| s * (0..12).map(|_| lcg(&mut seed)).sum::<f64>();
        let flat_noise = Array2::from_shape_fn((64, 64), |_| 0.5 + noise(0.03));
        let none = canny_edge_detection(&flat_noise, 1.5, 0.05, 0.12).unwrap();
        assert!(none.iter().filter(|v| **v).count() < 20);
        let noisy_square = &square(64, 20, 44) + &Array2::from_shape_fn((64, 64), |_| noise(0.03));
        let e = canny_edge_detection(&noisy_square, 1.5, 0.05, 0.12).unwrap();
        assert!(e[(32, 20)] || e[(32, 19)] || e[(32, 21)], "left edge found despite noise");
        assert!(e[(32, 43)] || e[(32, 44)] || e[(32, 42)], "right edge found despite noise");
    }

    #[test]
    fn a_disc_gives_a_closed_ring_at_the_right_radius() {
        let img = Array2::from_shape_fn((80, 80), |(r, c)| if ((r as f64 - 40.0).powi(2) + (c as f64 - 40.0).powi(2)).sqrt() < 20.0 { 1.0 } else { 0.0 });
        let e = canny_edge_detection(&img, 1.5, 0.05, 0.15).unwrap();
        let radii: Vec<f64> = e.indexed_iter().filter(|(_, v)| **v).map(|((r, c), _)| ((r as f64 - 40.0).powi(2) + (c as f64 - 40.0).powi(2)).sqrt()).collect();
        assert!(radii.len() > 100);
        assert!(radii.iter().all(|r| (r - 20.0).abs() < 1.5), "all edge pixels on the circle");
        // Every direction has an edge pixel: the ring is closed.
        let mut hit = [false; 36];
        for ((r, c), v) in e.indexed_iter() {
            if *v {
                let a = (r as f64 - 40.0).atan2(c as f64 - 40.0).to_degrees().rem_euclid(360.0);
                hit[(a / 10.0) as usize % 36] = true;
            }
        }
        assert!(hit.iter().all(|h| *h));
    }

    #[test]
    fn flat_images_and_invalid_thresholds() {
        let flat = Array2::from_elem((20, 20), 0.7);
        assert!(canny_edge_detection(&flat, 1.0, 0.1, 0.2).unwrap().iter().all(|v| !*v));
        assert!(canny_edge_detection(&flat, 1.0, 0.3, 0.2).is_err());
        assert!(canny_edge_detection(&flat, 1.0, -0.1, 0.2).is_err());
        assert!(canny_edge_detection(&flat, 0.0, 0.1, 0.2).is_err());
        let tiny = Array2::from_elem((2, 2), 1.0);
        assert!(canny_edge_detection(&tiny, 1.0, 0.1, 0.2).unwrap().iter().all(|v| !*v));
    }
}
