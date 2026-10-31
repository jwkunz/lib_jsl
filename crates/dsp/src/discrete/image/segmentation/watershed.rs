//! Watershed segmentation (marker-controlled flooding).
//!
//! Think of a grayscale image as a landscape in which brightness is altitude. Dropping a marker
//! into each valley and slowly raising the water level makes each marker's lake grow; where lakes
//! from different markers would merge, a dam (the watershed line) is built. Applied to a gradient
//! magnitude image, valleys are homogeneous regions and ridges are object boundaries, so the
//! flooded basins are the segments.
//!
//! The implementation is Meyer's priority-flood algorithm: a priority queue always releases the
//! lowest unprocessed pixel next, and each pixel takes the label of the neighbour that reached it.
//! Ties (equal elevation) are broken first-in-first-out so flooding spreads evenly across plateaus.
//! The result labels every pixel reachable from a marker; use [`label_boundaries`] to extract the
//! lines between regions.

use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(PartialEq)]
struct Entry {
    elevation: f64,
    age: u64,
    row: usize,
    col: usize,
}

impl Eq for Entry {}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Entry {
    /// `BinaryHeap` is a max-heap, so reverse the order to pop the lowest elevation, oldest first.
    fn cmp(&self, other: &Self) -> Ordering {
        other.elevation.partial_cmp(&self.elevation).unwrap_or(Ordering::Equal).then(other.age.cmp(&self.age))
    }
}

/// Floods `elevation` from `markers` (positive integers label the seeds, 0 means unlabelled) and
/// returns the label of every reachable pixel. Pixels connected to no marker stay 0.
pub fn watershed(elevation: &Array2<f64>, markers: &Array2<i32>) -> Result<Array2<i32>, ErrorsJSL> {
    watershed_masked(elevation, markers, None)
}

/// Like [`watershed`] but flooding only where `mask` is true (pixels outside the mask stay 0).
pub fn watershed_masked(elevation: &Array2<f64>, markers: &Array2<i32>, mask: Option<&Array2<bool>>) -> Result<Array2<i32>, ErrorsJSL> {
    let (rows, cols) = elevation.dim();
    if markers.dim() != (rows, cols) || mask.is_some_and(|m| m.dim() != (rows, cols)) {
        return Err(ErrorsJSL::IncompatibleArraySizes((rows * cols, markers.len())));
    }
    if rows == 0 || cols == 0 {
        return Err(ErrorsJSL::InvalidInputRange("Image must be non-empty"));
    }
    if markers.iter().any(|&m| m < 0) {
        return Err(ErrorsJSL::InvalidInputRange("Marker labels must be non-negative"));
    }
    if !markers.iter().any(|&m| m > 0) {
        return Err(ErrorsJSL::InvalidInputRange("At least one marker is required"));
    }
    if elevation.iter().any(|v| v.is_nan()) {
        return Err(ErrorsJSL::InvalidInputRange("Elevation must not contain NaN"));
    }
    let allowed = |r: usize, c: usize| mask.is_none_or(|m| m[(r, c)]);
    let mut labels = Array2::<i32>::zeros((rows, cols));
    let mut heap = BinaryHeap::new();
    let mut age = 0u64;
    for ((r, c), &m) in markers.indexed_iter() {
        if m > 0 && allowed(r, c) {
            labels[(r, c)] = m;
            heap.push(Entry { elevation: elevation[(r, c)], age, row: r, col: c });
            age += 1;
        }
    }
    while let Some(Entry { row, col, .. }) = heap.pop() {
        let label = labels[(row, col)];
        for (dr, dc) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
            let (nr, nc) = (row as isize + dr, col as isize + dc);
            if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                continue;
            }
            let (nr, nc) = (nr as usize, nc as usize);
            if labels[(nr, nc)] == 0 && allowed(nr, nc) {
                labels[(nr, nc)] = label;
                // A pixel is never processed below the level of the one that reached it.
                heap.push(Entry { elevation: elevation[(nr, nc)].max(elevation[(row, col)]), age, row: nr, col: nc });
                age += 1;
            }
        }
    }
    Ok(labels)
}

/// Marks pixels that have a right or lower neighbour with a different (nonzero) label: a
/// one-pixel-wide boundary between regions.
pub fn label_boundaries(labels: &Array2<i32>) -> Array2<bool> {
    let (rows, cols) = labels.dim();
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        let l = labels[(r, c)];
        l != 0
            && ((c + 1 < cols && labels[(r, c + 1)] != l && labels[(r, c + 1)] != 0) || (r + 1 < rows && labels[(r + 1, c)] != l && labels[(r + 1, c)] != 0))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::image::filters::{gaussian_blur::gaussian_blur, sobel::sobel};

    /// Two valleys with bottoms at columns 10 and 30: the elevation is the distance to the
    /// nearest bottom, so a ridge peaks at column 20.
    fn two_valleys() -> Array2<f64> {
        Array2::from_shape_fn((15, 41), |(_, c)| (c as f64 - 10.0).abs().min((c as f64 - 30.0).abs()))
    }

    #[test]
    fn two_basins_meet_at_the_ridge() {
        let elev = two_valleys();
        assert!(elev[(5, 10)] < elev[(5, 15)] && elev[(5, 15)] < elev[(5, 20)], "valley, slope, ridge");
        let mut markers = Array2::<i32>::zeros(elev.dim());
        markers[(7, 10)] = 1;
        markers[(7, 30)] = 2;
        let labels = watershed(&elev, &markers).unwrap();
        for r in 0..15 {
            for c in 0..41 {
                let want = if c < 20 { 1 } else if c > 20 { 2 } else { labels[(r, c)] };
                assert_eq!(labels[(r, c)], want, "({r}, {c})");
            }
        }
        // The ridge pixels are assigned to one side or the other, never left unlabelled.
        assert!((0..15).all(|r| labels[(r, 20)] == 1 || labels[(r, 20)] == 2));
        assert!(labels.iter().all(|&l| l == 1 || l == 2), "everything is reachable");
        // The boundary is a single column wide.
        let b = label_boundaries(&labels);
        for r in 0..15 {
            assert_eq!((0..41).filter(|&c| b[(r, c)]).count(), 1, "row {r}");
        }
    }

    #[test]
    fn markers_keep_their_labels_and_a_single_marker_floods_everything() {
        let elev = Array2::from_shape_fn((9, 9), |(r, c)| ((r * 7 + c * 3) % 5) as f64);
        let mut markers = Array2::<i32>::zeros((9, 9));
        markers[(4, 4)] = 7;
        let labels = watershed(&elev, &markers).unwrap();
        assert!(labels.iter().all(|&l| l == 7));
        markers[(0, 0)] = 3;
        markers[(8, 8)] = 9;
        let labels = watershed(&elev, &markers).unwrap();
        assert_eq!((labels[(4, 4)], labels[(0, 0)], labels[(8, 8)]), (7, 3, 9));
        assert!(labels.iter().all(|&l| [3, 7, 9].contains(&l)));
    }

    #[test]
    fn flat_terrain_splits_roughly_midway_by_distance() {
        let elev = Array2::<f64>::zeros((5, 21));
        let mut markers = Array2::<i32>::zeros((5, 21));
        markers[(2, 2)] = 1;
        markers[(2, 18)] = 2;
        let labels = watershed(&elev, &markers).unwrap();
        // First-in-first-out flooding advances both fronts at the same speed.
        let left = labels.iter().filter(|&&l| l == 1).count();
        let right = labels.iter().filter(|&&l| l == 2).count();
        assert!((left as i32 - right as i32).abs() <= 5, "{left} vs {right}");
        assert_eq!(labels[(2, 5)], 1);
        assert_eq!(labels[(2, 15)], 2);
    }

    #[test]
    fn segments_discs_using_the_gradient_magnitude() {
        let n = 80;
        let img = Array2::from_shape_fn((n, n), |(r, c)| {
            let d1 = ((r as f64 - 25.0).powi(2) + (c as f64 - 25.0).powi(2)).sqrt();
            let d2 = ((r as f64 - 55.0).powi(2) + (c as f64 - 55.0).powi(2)).sqrt();
            if d1 < 14.0 { 1.0 } else if d2 < 10.0 { 0.6 } else { 0.0 }
        });
        let grad = sobel(&gaussian_blur(&img, 1.0).unwrap(), true).unwrap().magnitude;
        let mut markers = Array2::<i32>::zeros((n, n));
        markers[(25, 25)] = 1;
        markers[(55, 55)] = 2;
        markers[(5, 70)] = 3; // background
        let labels = watershed(&grad, &markers).unwrap();
        let area = |l: i32| labels.iter().filter(|&&v| v == l).count() as f64;
        let disc = |r: f64| std::f64::consts::PI * r * r;
        assert!((area(1) / disc(14.0) - 1.0).abs() < 0.12, "disc 1 area {} vs {}", area(1), disc(14.0));
        assert!((area(2) / disc(10.0) - 1.0).abs() < 0.15, "disc 2 area {} vs {}", area(2), disc(10.0));
        assert_eq!(labels[(25, 25)], 1);
        assert_eq!(labels[(40, 40)], 3, "background between the discs");
        assert_eq!(area(1) + area(2) + area(3), (n * n) as f64);
    }

    #[test]
    fn masks_restrict_the_flood() {
        let elev = Array2::<f64>::zeros((6, 6));
        let mut markers = Array2::<i32>::zeros((6, 6));
        markers[(1, 1)] = 1;
        let mask = Array2::from_shape_fn((6, 6), |(r, _)| r < 3);
        let labels = watershed_masked(&elev, &markers, Some(&mask)).unwrap();
        assert!((0..6).all(|c| labels[(2, c)] == 1 && labels[(3, c)] == 0));
    }

    #[test]
    fn invalid_inputs() {
        let elev = Array2::<f64>::zeros((4, 4));
        let none = Array2::<i32>::zeros((4, 4));
        assert!(watershed(&elev, &none).is_err(), "no markers");
        assert!(watershed(&elev, &Array2::<i32>::zeros((3, 4))).is_err(), "shape mismatch");
        let mut neg = Array2::<i32>::zeros((4, 4));
        neg[(0, 0)] = -1;
        assert!(watershed(&elev, &neg).is_err());
        let mut m = Array2::<i32>::zeros((4, 4));
        m[(0, 0)] = 1;
        let mut nan = elev.clone();
        nan[(1, 1)] = f64::NAN;
        assert!(watershed(&nan, &m).is_err());
        assert!(watershed_masked(&elev, &m, Some(&Array2::from_elem((2, 2), true))).is_err());
        assert!(watershed(&Array2::<f64>::zeros((0, 3)), &Array2::<i32>::zeros((0, 3))).is_err());
    }
}
