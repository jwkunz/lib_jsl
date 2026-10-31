//! Hough transform: detecting lines and circles by voting.
//!
//! Every edge pixel *votes* for all the shapes that could pass through it; shapes supported by many
//! edge pixels collect many votes and show up as peaks in an accumulator array. This finds shapes
//! even when the edge map has gaps or clutter.
//!
//! - **Lines** use the normal form `ρ = x cos θ + y sin θ` (with `x` the column, `y` the row),
//!   `θ ∈ [0, π)`: each pixel votes for one `ρ` per `θ`, tracing a sinusoid in `(ρ, θ)` space. All
//!   pixels of a line vote for the same `(ρ, θ)` cell.
//! - **Circles** of a known radius: each pixel votes for every possible centre on the circle of that
//!   radius around it; a true circle's pixels all vote for its centre.

use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;
use std::f64::consts::PI;

/// The line-voting accumulator and its axes.
#[derive(Clone, Debug, PartialEq)]
pub struct HoughLines {
    /// Votes, indexed `(rho bin, theta bin)`.
    pub accumulator: Array2<u32>,
    /// The angle of each theta bin, `j · π / num_angles`.
    pub thetas: Vec<f64>,
    /// The `ρ` value of rho bin 0.
    pub rho_min: f64,
    /// Spacing of the rho bins.
    pub rho_step: f64,
}

impl HoughLines {
    /// The `ρ` at the centre of rho bin `i`.
    pub fn rho(&self, i: usize) -> f64 {
        self.rho_min + i as f64 * self.rho_step
    }
}

/// A detected line: all points with `x cos θ + y sin θ = ρ`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HoughLine {
    pub rho: f64,
    pub theta: f64,
    pub votes: u32,
}

/// Builds the line accumulator for a binary edge map using `num_angles` angles over `[0, π)` and a
/// distance resolution of `rho_resolution` pixels.
pub fn hough_transform(edges: &Array2<bool>, num_angles: usize, rho_resolution: f64) -> Result<HoughLines, ErrorsJSL> {
    if num_angles < 2 || !(rho_resolution > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("Need at least 2 angles and a positive rho resolution"));
    }
    let (rows, cols) = edges.dim();
    if rows == 0 || cols == 0 {
        return Err(ErrorsJSL::InvalidInputRange("Edge map must be non-empty"));
    }
    let diag = ((rows * rows + cols * cols) as f64).sqrt();
    let n_rho = (2.0 * diag / rho_resolution).ceil() as usize + 1;
    let rho_min = -diag;
    let thetas: Vec<f64> = (0..num_angles).map(|j| j as f64 * PI / num_angles as f64).collect();
    let trig: Vec<(f64, f64)> = thetas.iter().map(|t| (t.cos(), t.sin())).collect();
    let mut acc = Array2::<u32>::zeros((n_rho, num_angles));
    for ((r, c), &e) in edges.indexed_iter() {
        if !e {
            continue;
        }
        let (x, y) = (c as f64, r as f64);
        for (j, (cos, sin)) in trig.iter().enumerate() {
            let bin = ((x * cos + y * sin - rho_min) / rho_resolution).round() as usize;
            acc[(bin, j)] += 1;
        }
    }
    Ok(HoughLines { accumulator: acc, thetas, rho_min, rho_step: rho_resolution })
}

/// Extracts up to `max_lines` peaks with at least `min_votes` votes, strongest first. After taking
/// a peak, cells within `±min_rho_bins` and `±min_theta_bins` (wrapping across `θ = 0 ≡ π`, where
/// `ρ` changes sign) are suppressed so one physical line gives one result.
pub fn hough_line_peaks(h: &HoughLines, min_votes: u32, max_lines: usize, min_rho_bins: usize, min_theta_bins: usize) -> Vec<HoughLine> {
    let (n_rho, n_theta) = h.accumulator.dim();
    let mut cells: Vec<(u32, usize, usize)> = h.accumulator.indexed_iter().filter(|(_, v)| **v >= min_votes && **v > 0).map(|((i, j), &v)| (v, i, j)).collect();
    cells.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut suppressed = Array2::from_elem((n_rho, n_theta), false);
    let mut out = Vec::new();
    for (votes, i, j) in cells {
        if out.len() >= max_lines {
            break;
        }
        if suppressed[(i, j)] {
            continue;
        }
        out.push(HoughLine { rho: h.rho(i), theta: h.thetas[j], votes });
        for dj in -(min_theta_bins as isize)..=min_theta_bins as isize {
            let jj = j as isize + dj;
            // Wrapping past either end of the theta range mirrors rho.
            let (jw, flip) = if jj < 0 { ((jj + n_theta as isize) as usize, true) } else if jj >= n_theta as isize { ((jj - n_theta as isize) as usize, true) } else { (jj as usize, false) };
            for di in -(min_rho_bins as isize)..=min_rho_bins as isize {
                let ii = i as isize + di;
                let ii = if flip { n_rho as isize - 1 - ii } else { ii };
                if (0..n_rho as isize).contains(&ii) {
                    suppressed[(ii as usize, jw)] = true;
                }
            }
        }
    }
    out
}

/// A detected circle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HoughCircle {
    pub row: usize,
    pub col: usize,
    pub radius: usize,
    /// Votes received.
    pub votes: u32,
    /// Votes as a fraction of the circle's circumference in pixels (about 1 for a complete circle).
    pub support: f64,
}

/// Finds circles with the given candidate `radii` (pixels). A centre is reported if its votes are
/// at least `min_support` times the circumference `2πr` (try 0.6), strongest first, with centres
/// closer than `min_distance` pixels suppressed.
pub fn hough_circles(edges: &Array2<bool>, radii: &[usize], min_support: f64, min_distance: usize) -> Result<Vec<HoughCircle>, ErrorsJSL> {
    let (rows, cols) = edges.dim();
    if rows == 0 || cols == 0 || radii.is_empty() || radii.contains(&0) {
        return Err(ErrorsJSL::InvalidInputRange("Need a non-empty edge map and positive radii"));
    }
    if !(min_support > 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("min_support must be positive"));
    }
    let mut found: Vec<HoughCircle> = Vec::new();
    for &radius in radii {
        let circumference = 2.0 * PI * radius as f64;
        let steps = (circumference.ceil() as usize * 2).max(16);
        let offsets: Vec<(isize, isize)> = {
            let mut o: Vec<(isize, isize)> = (0..steps)
                .map(|k| {
                    let a = 2.0 * PI * k as f64 / steps as f64;
                    ((radius as f64 * a.sin()).round() as isize, (radius as f64 * a.cos()).round() as isize)
                })
                .collect();
            o.sort_unstable();
            o.dedup(); // each distinct centre candidate votes once per edge pixel
            o
        };
        let mut acc = Array2::<u32>::zeros((rows, cols));
        for ((r, c), &e) in edges.indexed_iter() {
            if !e {
                continue;
            }
            for &(dr, dc) in &offsets {
                let (cr, cc) = (r as isize + dr, c as isize + dc);
                if cr >= 0 && cc >= 0 && cr < rows as isize && cc < cols as isize {
                    acc[(cr as usize, cc as usize)] += 1;
                }
            }
        }
        let need = (min_support * circumference).ceil() as u32;
        for ((r, c), &v) in acc.indexed_iter() {
            if v >= need.max(1) {
                found.push(HoughCircle { row: r, col: c, radius, votes: v, support: v as f64 / circumference });
            }
        }
    }
    found.sort_by(|a, b| b.votes.cmp(&a.votes).then(a.row.cmp(&b.row)).then(a.col.cmp(&b.col)));
    let mut kept: Vec<HoughCircle> = Vec::new();
    for c in found {
        if kept.iter().all(|k| ((k.row as f64 - c.row as f64).powi(2) + (k.col as f64 - c.col as f64).powi(2)).sqrt() >= min_distance as f64) {
            kept.push(c);
        }
    }
    Ok(kept)
}

/// Convenience: the strongest lines in an edge map, with the default resolutions of one degree and
/// one pixel and mild suppression. Returns up to `max_lines` lines with at least `min_votes` votes.
pub fn hough_lines(edges: &Array2<bool>, min_votes: u32, max_lines: usize) -> Result<Vec<HoughLine>, ErrorsJSL> {
    let h = hough_transform(edges, 180, 1.0)?;
    Ok(hough_line_peaks(&h, min_votes, max_lines, 4, 8))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draw_line(a: &mut Array2<bool>, from: (usize, usize), to: (usize, usize)) {
        let n = ((to.0 as f64 - from.0 as f64).abs().max((to.1 as f64 - from.1 as f64).abs())) as usize;
        for k in 0..=n {
            let t = k as f64 / n as f64;
            let r = from.0 as f64 + t * (to.0 as f64 - from.0 as f64);
            let c = from.1 as f64 + t * (to.1 as f64 - from.1 as f64);
            a[(r.round() as usize, c.round() as usize)] = true;
        }
    }

    fn draw_circle(a: &mut Array2<bool>, centre: (f64, f64), radius: f64) {
        for k in 0..720 {
            let t = k as f64 * PI / 360.0;
            let (r, c) = ((centre.0 + radius * t.sin()).round(), (centre.1 + radius * t.cos()).round());
            if r >= 0.0 && c >= 0.0 && (r as usize) < a.nrows() && (c as usize) < a.ncols() {
                a[(r as usize, c as usize)] = true;
            }
        }
    }

    #[test]
    fn axis_aligned_lines_have_the_expected_parameters_and_vote_counts() {
        let mut e = Array2::from_elem((100, 100), false);
        draw_line(&mut e, (40, 0), (40, 99)); // horizontal, row 40
        let lines = hough_lines(&e, 50, 5).unwrap();
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!((lines[0].theta - PI / 2.0).abs() < 1e-9 && (lines[0].rho - 40.0).abs() < 1.0, "{:?}", lines[0]);
        assert_eq!(lines[0].votes, 100, "every pixel of the line voted for the same cell");

        let mut v = Array2::from_elem((100, 100), false);
        draw_line(&mut v, (0, 30), (99, 30)); // vertical, column 30
        let lines = hough_lines(&v, 50, 5).unwrap();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].theta.abs() < 1e-9 && (lines[0].rho - 30.0).abs() < 1.0, "{:?}", lines[0]);
    }

    #[test]
    fn a_diagonal_and_two_crossing_lines_are_all_found() {
        let mut e = Array2::from_elem((100, 100), false);
        draw_line(&mut e, (0, 0), (99, 99)); // y = x : normal at 135 degrees, rho = 0
        let lines = hough_lines(&e, 60, 3).unwrap();
        assert_eq!(lines.len(), 1);
        assert!((lines[0].theta - 135f64.to_radians()).abs() < 1e-9 && lines[0].rho.abs() < 1.0, "{:?}", lines[0]);
        draw_line(&mut e, (20, 90), (80, 10));
        draw_line(&mut e, (60, 0), (60, 99));
        // The 3:4-slope line (normal at 53.13 degrees) does not land on a whole degree, so its 81
        // pixels spread over neighbouring cells and no single cell gets all the votes.
        let lines = hough_lines(&e, 30, 10).unwrap();
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert!(lines.windows(2).all(|w| w[0].votes >= w[1].votes), "strongest first");
        let off_grid = lines.iter().min_by_key(|l| l.votes).unwrap();
        assert!((off_grid.theta.to_degrees() - 53.13).abs() < 1.6 && off_grid.votes < 81, "{off_grid:?}");
    }

    #[test]
    fn broken_lines_and_clutter_do_not_prevent_detection() {
        let mut e = Array2::from_elem((120, 120), false);
        draw_line(&mut e, (50, 0), (50, 119));
        for c in 40..60 {
            e[(50, c)] = false; // a gap
        }
        let mut seed = 5u64;
        for _ in 0..150 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let (r, c) = ((seed >> 33) as usize % 120, (seed >> 13) as usize % 120);
            e[(r, c)] = true;
        }
        let lines = hough_lines(&e, 80, 3).unwrap();
        assert!(!lines.is_empty() && (lines[0].theta - PI / 2.0).abs() < 1e-9 && (lines[0].rho - 50.0).abs() < 1.5, "{lines:?}");
        assert!(lines[0].votes >= 100);
    }

    #[test]
    fn suppression_wraps_across_theta_zero_without_duplicating_a_vertical_line() {
        // A vertical line sits exactly on the seam where theta = 0 and theta = pi coincide.
        let mut e = Array2::from_elem((80, 80), false);
        draw_line(&mut e, (0, 25), (79, 25));
        let h = hough_transform(&e, 180, 1.0).unwrap();
        let lines = hough_line_peaks(&h, 20, 10, 3, 6);
        assert_eq!(lines.len(), 1, "{lines:?}");
    }

    #[test]
    fn circles_are_located_and_their_radius_is_chosen_from_the_candidates() {
        let mut e = Array2::from_elem((120, 120), false);
        draw_circle(&mut e, (50.0, 60.0), 15.0);
        let found = hough_circles(&e, &[10, 15, 20], 0.7, 8).unwrap();
        assert!(!found.is_empty());
        let best = found[0];
        assert_eq!(best.radius, 15);
        assert!((best.row as i32 - 50).abs() <= 1 && (best.col as i32 - 60).abs() <= 1, "{best:?}");
        assert!(best.support > 0.9);
        // Only one distinct centre is reported for the radius-15 circle (min_distance suppression).
        assert_eq!(found.iter().filter(|c| c.radius == 15).count(), 1);

        draw_circle(&mut e, (90.0, 25.0), 12.0);
        let both = hough_circles(&e, &[12, 15], 0.7, 10).unwrap();
        assert!(both.iter().any(|c| c.radius == 15 && c.row.abs_diff(50) <= 1 && c.col.abs_diff(60) <= 1));
        assert!(both.iter().any(|c| c.radius == 12 && c.row.abs_diff(90) <= 1 && c.col.abs_diff(25) <= 1));
    }

    #[test]
    fn an_empty_edge_map_finds_nothing() {
        let e = Array2::from_elem((30, 30), false);
        assert!(hough_lines(&e, 1, 5).unwrap().is_empty());
        assert!(hough_circles(&e, &[5], 0.5, 3).unwrap().is_empty());
    }

    #[test]
    fn invalid_arguments() {
        let e = Array2::from_elem((10, 10), false);
        assert!(hough_transform(&e, 1, 1.0).is_err() && hough_transform(&e, 90, 0.0).is_err());
        assert!(hough_transform(&Array2::from_elem((0, 5), false), 90, 1.0).is_err());
        assert!(hough_circles(&e, &[], 0.5, 3).is_err() && hough_circles(&e, &[0], 0.5, 3).is_err() && hough_circles(&e, &[3], 0.0, 3).is_err());
    }
}
