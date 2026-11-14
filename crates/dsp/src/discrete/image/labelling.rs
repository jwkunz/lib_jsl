//! Connected-component labelling and per-region measurements, following `scipy.ndimage.label`,
//! `find_objects`, `sum`, `mean`, `minimum`, `maximum` and `center_of_mass`.
//!
//! Labelling gives each connected group of foreground pixels its own number, so that objects can
//! be counted and measured one by one. Two pixels are connected if a chain of foreground pixels
//! links them, each step allowed by the structuring element (the cross for edge neighbours only,
//! the 3 × 3 square to include diagonals). The classical two-pass algorithm assigns provisional
//! labels in one raster scan, recording which of them touch in a union-find structure, and then
//! replaces each by its representative.
//!
//! See the book entry [Morphology, distance transforms and labelling](https://jwkunz.github.io/lib_jsl/book/image/morphology.html).

use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

// ANCHOR: label
/// Labels the connected components of a binary image (SciPy's `label`). Returns the label image
/// (0 for background, `1..=n` for the components, numbered in the raster order of their first
/// pixel) and the number of components `n`. `structure` must be symmetric with odd dimensions;
/// [`cross`](super::morphology::cross) gives 4-connectivity and
/// [`square`](super::morphology::square)`(3)` 8-connectivity.
pub fn label(mask: &Array2<bool>, structure: &Array2<bool>) -> Result<(Array2<usize>, usize), ErrorsJSL> {
    let (h, w) = structure.dim();
    if h % 2 == 0 || w % 2 == 0 {
        return Err(ErrorsJSL::InvalidInputRange("the structure must have odd dimensions"));
    }
    let (ch, cw) = ((h / 2) as isize, (w / 2) as isize);
    // Neighbours already visited in a raster scan: earlier rows, or earlier columns of this row.
    let earlier: Vec<(isize, isize)> = structure
        .indexed_iter()
        .filter(|(_, b)| **b)
        .map(|((i, j), _)| (i as isize - ch, j as isize - cw))
        .filter(|&(di, dj)| di < 0 || (di == 0 && dj < 0))
        .collect();
    let (rows, cols) = mask.dim();
    let mut labels = Array2::<usize>::zeros((rows, cols));
    let mut parent = vec![0usize];
    for r in 0..rows {
        for c in 0..cols {
            if !mask[(r, c)] {
                continue;
            }
            let mut current = 0;
            for &(di, dj) in &earlier {
                let (nr, nc) = (r as isize + di, c as isize + dj);
                if nr < 0 || nc < 0 || nc >= cols as isize {
                    continue;
                }
                let l = labels[(nr as usize, nc as usize)];
                if l == 0 {
                    continue;
                }
                if current == 0 {
                    current = l;
                } else {
                    let (a, b) = (find(&mut parent, current), find(&mut parent, l));
                    if a != b {
                        let (lo, hi) = (a.min(b), a.max(b));
                        parent[hi] = lo;
                    }
                }
            }
            if current == 0 {
                current = parent.len();
                parent.push(current);
            }
            labels[(r, c)] = current;
        }
    }
    // Second pass: representatives, renumbered 1..n in order of first appearance.
    let mut final_label = vec![0usize; parent.len()];
    let mut n = 0;
    for v in labels.iter_mut() {
        if *v == 0 {
            continue;
        }
        let root = find(&mut parent, *v);
        if final_label[root] == 0 {
            n += 1;
            final_label[root] = n;
        }
        *v = final_label[root];
    }
    Ok((labels, n))
}
// ANCHOR_END: label

/// The measurements of one labelled region.
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    /// The label, `1..=n`.
    pub label: usize,
    /// The number of pixels.
    pub area: usize,
    /// The bounding box as half-open ranges: rows `top..bottom`, columns `left..right` (SciPy's
    /// `find_objects` slices).
    pub bbox: (usize, usize, usize, usize),
    /// The mean `(row, column)` of the region's pixels.
    pub centroid: (f64, f64),
    /// With an intensity image: the sum, mean, minimum and maximum of its values over the region,
    /// and the intensity-weighted centre (SciPy's `center_of_mass`).
    pub sum: Option<f64>,
    pub mean: Option<f64>,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub center_of_mass: Option<(f64, f64)>,
}

// ANCHOR: regions
/// Measures every region of a label image in one pass: area, bounding box and centroid, and with
/// `values` the intensity sum, mean, extremes and weighted centre.
pub fn regions(labels: &Array2<usize>, n: usize, values: Option<&Array2<f64>>) -> Result<Vec<Region>, ErrorsJSL> {
    if let Some(v) = values
        && v.dim() != labels.dim()
    {
        return Err(ErrorsJSL::IncompatibleArraySizes((v.len(), labels.len())));
    }
    let mut area = vec![0usize; n + 1];
    let mut bbox = vec![(usize::MAX, 0usize, usize::MAX, 0usize); n + 1];
    let mut sr = vec![0.0; n + 1];
    let mut sc = vec![0.0; n + 1];
    let mut sum = vec![0.0; n + 1];
    let mut wr = vec![0.0; n + 1];
    let mut wc = vec![0.0; n + 1];
    let mut lo = vec![f64::INFINITY; n + 1];
    let mut hi = vec![f64::NEG_INFINITY; n + 1];
    for ((r, c), &l) in labels.indexed_iter() {
        if l == 0 {
            continue;
        }
        if l > n {
            return Err(ErrorsJSL::InvalidInputRange("a label exceeds the number of regions"));
        }
        area[l] += 1;
        let b = &mut bbox[l];
        *b = (b.0.min(r), b.1.max(r + 1), b.2.min(c), b.3.max(c + 1));
        sr[l] += r as f64;
        sc[l] += c as f64;
        if let Some(v) = values {
            let x = v[(r, c)];
            sum[l] += x;
            wr[l] += x * r as f64;
            wc[l] += x * c as f64;
            lo[l] = lo[l].min(x);
            hi[l] = hi[l].max(x);
        }
    }
    Ok((1..=n)
        .filter(|&l| area[l] > 0)
        .map(|l| {
            let a = area[l] as f64;
            let with = values.is_some();
            Region {
                label: l,
                area: area[l],
                bbox: bbox[l],
                centroid: (sr[l] / a, sc[l] / a),
                sum: with.then_some(sum[l]),
                mean: with.then_some(sum[l] / a),
                minimum: with.then_some(lo[l]),
                maximum: with.then_some(hi[l]),
                center_of_mass: with.then_some((wr[l] / sum[l], wc[l] / sum[l])),
            }
        })
        .collect())
}
// ANCHOR_END: regions

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::image::morphology::{cross, square};

    fn from_rows(rows: &[&str]) -> Array2<bool> {
        Array2::from_shape_fn((rows.len(), rows[0].len()), |(i, j)| rows[i].as_bytes()[j] == b'#')
    }

    #[test]
    fn diagonal_neighbours_connect_only_with_the_square() {
        let img = from_rows(&["#..#", ".#.#", "..##", "#..."]);
        let (_, n4) = label(&img, &cross()).unwrap();
        let (_, n8) = label(&img, &square(3)).unwrap();
        assert_eq!(n4, 4);
        assert_eq!(n8, 2);
    }

    #[test]
    fn a_u_shape_merges_two_provisional_labels() {
        let img = from_rows(&["#.#", "#.#", "###"]);
        let (labels, n) = label(&img, &cross()).unwrap();
        assert_eq!(n, 1);
        assert!(labels.iter().all(|&l| l <= 1));
        let r = regions(&labels, n, Some(&labels.mapv(|l| l as f64 * 2.0))).unwrap();
        assert_eq!(r[0].area, 7);
        assert_eq!(r[0].bbox, (0, 3, 0, 3));
        assert_eq!(r[0].sum, Some(14.0));
    }
}
