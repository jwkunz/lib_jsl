//! Rank filters: median, percentile, minimum and maximum over a moving window, following
//! `scipy.ndimage`.
//!
//! A rank filter replaces each pixel by an order statistic of its neighbourhood: the median, a
//! percentile, the minimum or the maximum. Unlike a linear filter it does not average outliers into
//! their surroundings, so the median removes salt-and-pepper noise completely while keeping edges
//! sharp. The minimum and maximum are greyscale erosion and dilation.
//!
//! See the book entry [Rank filters, derivative filters and geometric transforms](https://jwkunz.github.io/lib_jsl/book/image/rank-geometric.html).

use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;

use super::boundary::{Mode, sample};

// ANCHOR: rank_filter
/// The rank filter (SciPy's `rank_filter`): each pixel becomes the `rank`-th smallest value
/// (counting from 0) of the pixels under the footprint centred on it; a negative rank counts from
/// the largest. The footprint's origin is its element `(h / 2, w / 2)`, so even sizes are allowed
/// and extend one further before the pixel than after it, as in SciPy.
pub fn rank_filter(image: &Array2<f64>, rank: isize, footprint: &Array2<bool>, mode: Mode) -> Result<Array2<f64>, ErrorsJSL> {
    let (h, w) = footprint.dim();
    let offsets: Vec<(isize, isize)> = footprint
        .indexed_iter()
        .filter(|(_, b)| **b)
        .map(|((i, j), _)| (i as isize - (h / 2) as isize, j as isize - (w / 2) as isize))
        .collect();
    let n = offsets.len() as isize;
    if n == 0 || image.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("the image and the footprint must be non-empty"));
    }
    let k = if rank < 0 { n + rank } else { rank };
    if !(0..n).contains(&k) {
        return Err(ErrorsJSL::InvalidInputRange("rank out of range for the footprint"));
    }
    let mut window = vec![0.0; n as usize];
    let mut out = Array2::zeros(image.dim());
    for ((r, c), o) in out.indexed_iter_mut() {
        for (v, &(di, dj)) in window.iter_mut().zip(&offsets) {
            *v = sample(image, r as isize + di, c as isize + dj, mode);
        }
        let (_, kth, _) = window.select_nth_unstable_by(k as usize, |a, b| a.total_cmp(b));
        *o = *kth;
    }
    Ok(out)
}
// ANCHOR_END: rank_filter

/// The median filter over a `size.0 × size.1` window (SciPy's `median_filter`; with an even
/// number of pixels it takes the upper of the two middle values, `rank = n / 2`).
pub fn median_filter(image: &Array2<f64>, size: (usize, usize), mode: Mode) -> Result<Array2<f64>, ErrorsJSL> {
    let fp = Array2::from_elem(size, true);
    rank_filter(image, (size.0 * size.1 / 2) as isize, &fp, mode)
}

/// The percentile filter (SciPy's `percentile_filter`): the rank is `⌊n p / 100⌋` for `p` in
/// `[0, 100)`, and the largest value for `p = 100`.
pub fn percentile_filter(image: &Array2<f64>, percentile: f64, size: (usize, usize), mode: Mode) -> Result<Array2<f64>, ErrorsJSL> {
    if !(0.0..=100.0).contains(&percentile) {
        return Err(ErrorsJSL::InvalidInputRange("percentile must be in [0, 100]"));
    }
    let n = size.0 * size.1;
    let rank = if percentile == 100.0 { n as isize - 1 } else { (n as f64 * percentile / 100.0) as isize };
    rank_filter(image, rank, &Array2::from_elem(size, true), mode)
}

/// The minimum over a `size` window (SciPy's `minimum_filter`).
pub fn minimum_filter(image: &Array2<f64>, size: (usize, usize), mode: Mode) -> Result<Array2<f64>, ErrorsJSL> {
    rank_filter(image, 0, &Array2::from_elem(size, true), mode)
}

/// The maximum over a `size` window (SciPy's `maximum_filter`).
pub fn maximum_filter(image: &Array2<f64>, size: (usize, usize), mode: Mode) -> Result<Array2<f64>, ErrorsJSL> {
    rank_filter(image, -1, &Array2::from_elem(size, true), mode)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_median_removes_impulses_and_keeps_an_edge() {
        let mut img = Array2::from_shape_fn((9, 9), |(_, c)| if c < 4 { 0.0 } else { 10.0 });
        img[(2, 1)] = 255.0;
        img[(6, 7)] = -255.0;
        let m = median_filter(&img, (3, 3), Mode::Reflect).unwrap();
        let clean = Array2::from_shape_fn((9, 9), |(_, c)| if c < 4 { 0.0 } else { 10.0 });
        assert_eq!(m, clean);
    }

    #[test]
    fn even_windows_extend_further_before_the_pixel() {
        // A row 0..6 with a window of 2: SciPy's origin makes it [i − 1, i].
        let img = Array2::from_shape_fn((1, 6), |(_, c)| c as f64);
        let m = maximum_filter(&img, (1, 2), Mode::Nearest).unwrap();
        assert_eq!(m.row(0).to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let m = minimum_filter(&img, (1, 2), Mode::Nearest).unwrap();
        assert_eq!(m.row(0).to_vec(), vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
    }
}
