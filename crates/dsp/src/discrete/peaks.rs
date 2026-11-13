//! Peak finding in one-dimensional signals: local maxima selected by height, threshold, distance,
//! prominence and width, following SciPy's `find_peaks`, `peak_prominences` and `peak_widths`.
//!
//! A *peak* is a sample (or a flat run of samples, a *plateau*) larger than its neighbours on both
//! sides. Real signals have many such points, most of them noise, so peaks are filtered by
//! properties that describe how much they stand out:
//!
//! - **height**: the value itself;
//! - **threshold**: the drop to the immediate neighbours;
//! - **distance**: the gap to a taller peak (the shorter of two close peaks is removed);
//! - **prominence**: how far the peak rises above the higher of the two lowest points that
//!   separate it from taller terrain on each side (the topographic prominence of a mountain);
//! - **width**: the peak's width at a given fraction of its prominence below the summit.
//!
//! See the book entry [Peak finding](https://jwkunz.github.io/lib_jsl/book/spectral/peaks.html).

use crate::prelude::ErrorsJSL;

/// An inclusive range `[min, max]`; either end may be open.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Bounds {
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl Bounds {
    /// Values at least `min`.
    pub fn at_least(min: f64) -> Self {
        Self { min: Some(min), max: None }
    }
    /// Values in `[min, max]`.
    pub fn between(min: f64, max: f64) -> Self {
        Self { min: Some(min), max: Some(max) }
    }
    fn contains(&self, v: f64) -> bool {
        self.min.is_none_or(|m| m <= v) && self.max.is_none_or(|m| v <= m)
    }
}

/// The selection criteria of [`find_peaks`]; `None` disables a criterion. The defaults select
/// every local maximum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeakOptions {
    /// Required peak value.
    pub height: Option<Bounds>,
    /// Required vertical drop to the neighbouring samples (both sides).
    pub threshold: Option<Bounds>,
    /// Minimum horizontal distance in samples to the nearest kept peak; of two closer peaks the
    /// lower one is removed.
    pub distance: Option<f64>,
    /// Required prominence.
    pub prominence: Option<Bounds>,
    /// Required width in samples, measured at `rel_height` of the prominence.
    pub width: Option<Bounds>,
    /// Window length in samples that limits the search for a peak's bases (prominence).
    pub wlen: Option<f64>,
    /// Where the width is measured, as a fraction of the prominence below the peak (default 0.5).
    pub rel_height: f64,
    /// Required plateau length in samples.
    pub plateau_size: Option<Bounds>,
}

impl Default for PeakOptions {
    fn default() -> Self {
        Self { height: None, threshold: None, distance: None, prominence: None, width: None, wlen: None, rel_height: 0.5, plateau_size: None }
    }
}

/// Prominences of peaks and the bases they were measured from.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Prominences {
    pub prominences: Vec<f64>,
    pub left_bases: Vec<usize>,
    pub right_bases: Vec<usize>,
}

/// Widths of peaks: the width, the height at which it was measured, and the interpolated
/// positions of its left and right ends.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Widths {
    pub widths: Vec<f64>,
    pub width_heights: Vec<f64>,
    pub left_ips: Vec<f64>,
    pub right_ips: Vec<f64>,
}

/// The result of [`find_peaks`]: the indices of the peaks, and the properties computed while
/// selecting them (only those the options required, as in SciPy).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Peaks {
    pub indices: Vec<usize>,
    pub peak_heights: Option<Vec<f64>>,
    pub left_thresholds: Option<Vec<f64>>,
    pub right_thresholds: Option<Vec<f64>>,
    pub plateau_sizes: Option<Vec<usize>>,
    pub left_edges: Option<Vec<usize>>,
    pub right_edges: Option<Vec<usize>>,
    pub prominences: Option<Prominences>,
    pub widths: Option<Widths>,
}

// ANCHOR: local_maxima
/// Local maxima of `x`, with the left and right edges of each plateau. A flat run counts once,
/// at its middle sample (rounded down), when the samples on both sides of it are lower. The end
/// samples are never peaks.
pub fn local_maxima(x: &[f64]) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let (mut peaks, mut left, mut right) = (Vec::new(), Vec::new(), Vec::new());
    let n = x.len();
    let mut i = 1;
    while i + 1 < n {
        if x[i - 1] < x[i] {
            let mut ahead = i + 1;
            while ahead + 1 < n && x[ahead] == x[i] {
                ahead += 1;
            }
            if x[ahead] < x[i] {
                left.push(i);
                right.push(ahead - 1);
                peaks.push((i + ahead - 1) / 2);
                i = ahead;
            }
        }
        i += 1;
    }
    (peaks, left, right)
}
// ANCHOR_END: local_maxima

fn check_signal(x: &[f64]) -> Result<(), ErrorsJSL> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err(ErrorsJSL::InvalidInputRange("the signal must be finite"));
    }
    Ok(())
}

// ANCHOR: peak_prominences
/// The prominence of each peak (SciPy's `peak_prominences`).
///
/// From the peak, walk left while the signal stays at or below the peak, and record the lowest
/// value passed; do the same to the right. The walk stops at a higher sample or at the edge of
/// the window `wlen` centred on the peak. The higher of the two minima is the *key col*, and the
/// prominence is the peak's height above it.
pub fn peak_prominences(x: &[f64], peaks: &[usize], wlen: Option<f64>) -> Result<Prominences, ErrorsJSL> {
    check_signal(x)?;
    if peaks.iter().any(|&p| p >= x.len()) {
        return Err(ErrorsJSL::InvalidInputRange("peak index out of range"));
    }
    let half = match wlen {
        None => None,
        Some(w) if w.ceil() > 1.0 => Some(w.ceil() as usize / 2),
        Some(_) => return Err(ErrorsJSL::InvalidInputRange("wlen must be larger than 1")),
    };
    let mut out = Prominences::default();
    for &peak in peaks {
        let (i_min, i_max) = match half {
            Some(h) => (peak.saturating_sub(h), (peak + h).min(x.len() - 1)),
            None => (0, x.len() - 1),
        };
        let (mut left_base, mut left_min) = (peak, x[peak]);
        let mut i = peak as isize;
        while i >= i_min as isize && x[i as usize] <= x[peak] {
            if x[i as usize] < left_min {
                left_min = x[i as usize];
                left_base = i as usize;
            }
            i -= 1;
        }
        let (mut right_base, mut right_min) = (peak, x[peak]);
        let mut i = peak;
        while i <= i_max && x[i] <= x[peak] {
            if x[i] < right_min {
                right_min = x[i];
                right_base = i;
            }
            i += 1;
        }
        out.prominences.push(x[peak] - left_min.max(right_min));
        out.left_bases.push(left_base);
        out.right_bases.push(right_base);
    }
    Ok(out)
}
// ANCHOR_END: peak_prominences

// ANCHOR: peak_widths
/// The width of each peak at `rel_height` of its prominence below the summit (SciPy's
/// `peak_widths`): from the peak, walk outwards until the signal drops below that height (or a
/// base is reached), and interpolate linearly between the two samples that straddle it.
/// `rel_height = 0.5` gives the full width at half prominence; `1.0` the width at the key col.
pub fn peak_widths(x: &[f64], peaks: &[usize], rel_height: f64, prominence: Option<&Prominences>) -> Result<Widths, ErrorsJSL> {
    if !(rel_height >= 0.0) {
        return Err(ErrorsJSL::InvalidInputRange("rel_height must be at least 0"));
    }
    let computed;
    let p = match prominence {
        Some(p) => p,
        None => {
            computed = peak_prominences(x, peaks, None)?;
            &computed
        }
    };
    if p.prominences.len() != peaks.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((p.prominences.len(), peaks.len())));
    }
    let mut out = Widths::default();
    for (k, &peak) in peaks.iter().enumerate() {
        let (i_min, i_max) = (p.left_bases[k], p.right_bases[k]);
        let height = x[peak] - p.prominences[k] * rel_height;
        let mut i = peak;
        while i_min < i && height < x[i] {
            i -= 1;
        }
        let mut left = i as f64;
        if x[i] < height {
            left += (height - x[i]) / (x[i + 1] - x[i]);
        }
        let mut i = peak;
        while i < i_max && height < x[i] {
            i += 1;
        }
        let mut right = i as f64;
        if x[i] < height {
            right -= (height - x[i]) / (x[i - 1] - x[i]);
        }
        out.widths.push(right - left);
        out.width_heights.push(height);
        out.left_ips.push(left);
        out.right_ips.push(right);
    }
    Ok(out)
}
// ANCHOR_END: peak_widths

/// Keeps the entries of `values` where `keep` is true.
fn retain<T: Copy>(values: &mut Vec<T>, keep: &[bool]) {
    let mut k = keep.iter();
    values.retain(|_| *k.next().unwrap());
}

/// Peaks of `x` at least `distance` samples apart, removing lower peaks first (SciPy's
/// `_select_by_peak_distance`). Peaks of equal height are taken from the right.
fn select_by_distance(peaks: &[usize], x: &[f64], distance: f64) -> Vec<bool> {
    let distance = distance.ceil() as usize;
    let n = peaks.len();
    let mut keep = vec![true; n];
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[peaks[a]].total_cmp(&x[peaks[b]]));
    for &j in order.iter().rev() {
        if !keep[j] {
            continue;
        }
        let mut k = j;
        while k > 0 && peaks[j] - peaks[k - 1] < distance {
            keep[k - 1] = false;
            k -= 1;
        }
        let mut k = j + 1;
        while k < n && peaks[k] - peaks[j] < distance {
            keep[k] = false;
            k += 1;
        }
    }
    keep
}

// ANCHOR: find_peaks
/// The peaks of `x` that satisfy the criteria in `options` (SciPy's `find_peaks`).
///
/// Candidates are the local maxima ([`local_maxima`]). The criteria are applied in SciPy's order,
/// cheapest first: plateau size, height, threshold, distance, prominence, width. The distance
/// criterion removes peaks, so prominences and widths are computed only for the survivors.
pub fn find_peaks(x: &[f64], options: &PeakOptions) -> Result<Peaks, ErrorsJSL> {
    check_signal(x)?;
    if options.distance.is_some_and(|d| !(d >= 1.0)) {
        return Err(ErrorsJSL::InvalidInputRange("distance must be at least 1"));
    }
    let (mut peaks, mut left_edges, mut right_edges) = local_maxima(x);
    let mut result = Peaks::default();

    if let Some(bounds) = options.plateau_size {
        let sizes: Vec<usize> = left_edges.iter().zip(&right_edges).map(|(l, r)| r - l + 1).collect();
        let keep: Vec<bool> = sizes.iter().map(|&s| bounds.contains(s as f64)).collect();
        let mut sizes = sizes;
        for v in [&mut peaks, &mut left_edges, &mut right_edges] {
            retain(v, &keep);
        }
        retain(&mut sizes, &keep);
        result.plateau_sizes = Some(sizes);
    }

    // Each later criterion filters the peaks and every property already computed.
    let mut heights: Option<Vec<f64>> = None;
    let mut thresholds: Option<(Vec<f64>, Vec<f64>)> = None;
    let apply = |keep: &[bool], peaks: &mut Vec<usize>, le: &mut Vec<usize>, re: &mut Vec<usize>, result: &mut Peaks, heights: &mut Option<Vec<f64>>, thresholds: &mut Option<(Vec<f64>, Vec<f64>)>| {
        retain(peaks, keep);
        retain(le, keep);
        retain(re, keep);
        if let Some(s) = result.plateau_sizes.as_mut() {
            retain(s, keep);
        }
        if let Some(h) = heights.as_mut() {
            retain(h, keep);
        }
        if let Some((l, r)) = thresholds.as_mut() {
            retain(l, keep);
            retain(r, keep);
        }
    };

    if let Some(bounds) = options.height {
        let h: Vec<f64> = peaks.iter().map(|&p| x[p]).collect();
        let keep: Vec<bool> = h.iter().map(|&v| bounds.contains(v)).collect();
        heights = Some(h);
        apply(&keep, &mut peaks, &mut left_edges, &mut right_edges, &mut result, &mut heights, &mut thresholds);
    }

    if let Some(bounds) = options.threshold {
        let l: Vec<f64> = peaks.iter().map(|&p| x[p] - x[p - 1]).collect();
        let r: Vec<f64> = peaks.iter().map(|&p| x[p] - x[p + 1]).collect();
        let keep: Vec<bool> = l
            .iter()
            .zip(&r)
            .map(|(&a, &b)| bounds.min.is_none_or(|m| m <= a.min(b)) && bounds.max.is_none_or(|m| a.max(b) <= m))
            .collect();
        thresholds = Some((l, r));
        apply(&keep, &mut peaks, &mut left_edges, &mut right_edges, &mut result, &mut heights, &mut thresholds);
    }

    if let Some(d) = options.distance {
        let keep = select_by_distance(&peaks, x, d);
        apply(&keep, &mut peaks, &mut left_edges, &mut right_edges, &mut result, &mut heights, &mut thresholds);
    }

    let mut prominences = None;
    if options.prominence.is_some() || options.width.is_some() {
        let mut p = peak_prominences(x, &peaks, options.wlen)?;
        if let Some(bounds) = options.prominence {
            let keep: Vec<bool> = p.prominences.iter().map(|&v| bounds.contains(v)).collect();
            apply(&keep, &mut peaks, &mut left_edges, &mut right_edges, &mut result, &mut heights, &mut thresholds);
            retain(&mut p.prominences, &keep);
            retain(&mut p.left_bases, &keep);
            retain(&mut p.right_bases, &keep);
        }
        prominences = Some(p);
    }

    if let Some(bounds) = options.width {
        let p = prominences.as_mut().expect("computed above");
        let mut w = peak_widths(x, &peaks, options.rel_height, Some(p))?;
        let keep: Vec<bool> = w.widths.iter().map(|&v| bounds.contains(v)).collect();
        apply(&keep, &mut peaks, &mut left_edges, &mut right_edges, &mut result, &mut heights, &mut thresholds);
        retain(&mut p.prominences, &keep);
        retain(&mut p.left_bases, &keep);
        retain(&mut p.right_bases, &keep);
        retain(&mut w.widths, &keep);
        retain(&mut w.width_heights, &keep);
        retain(&mut w.left_ips, &keep);
        retain(&mut w.right_ips, &keep);
        result.widths = Some(w);
    }

    if options.plateau_size.is_some() {
        result.left_edges = Some(left_edges);
        result.right_edges = Some(right_edges);
    }
    result.peak_heights = heights;
    if let Some((l, r)) = thresholds {
        result.left_thresholds = Some(l);
        result.right_thresholds = Some(r);
    }
    result.prominences = prominences;
    result.indices = peaks;
    Ok(result)
}
// ANCHOR_END: find_peaks

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plateaus_count_once_at_their_middle() {
        let x = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 0.0, 3.0, 3.0];
        let (p, l, r) = local_maxima(&x);
        assert_eq!(p, vec![2, 6]);
        assert_eq!(l, vec![1, 6]);
        assert_eq!(r, vec![4, 6]);
    }

    #[test]
    fn prominence_is_height_above_the_higher_col() {
        // Two summits: 5 at index 2 and 8 at index 6, separated by a col at 1.
        let x = [0.0, 3.0, 5.0, 2.0, 1.0, 4.0, 8.0, 2.0, 0.0];
        let p = peak_prominences(&x, &[2, 6], None).unwrap();
        assert_eq!(p.prominences, vec![4.0, 8.0]);
        assert_eq!(p.left_bases, vec![0, 0]);
        assert_eq!(p.right_bases, vec![4, 8]);
    }

    #[test]
    fn half_prominence_width_of_a_triangle() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
        let w = peak_widths(&x, &[4], 0.5, None).unwrap();
        assert!((w.widths[0] - 4.0).abs() < 1e-15);
        assert_eq!(w.width_heights, vec![2.0]);
    }

    #[test]
    fn distance_keeps_the_taller_of_close_peaks() {
        let x = [0.0, 2.0, 0.0, 3.0, 0.0, 1.0, 0.0];
        let all = find_peaks(&x, &PeakOptions::default()).unwrap();
        assert_eq!(all.indices, vec![1, 3, 5]);
        let spaced = find_peaks(&x, &PeakOptions { distance: Some(3.0), ..Default::default() }).unwrap();
        assert_eq!(spaced.indices, vec![3]);
        let tall = find_peaks(&x, &PeakOptions { height: Some(Bounds::at_least(1.5)), ..Default::default() }).unwrap();
        assert_eq!(tall.indices, vec![1, 3]);
        assert_eq!(tall.peak_heights, Some(vec![2.0, 3.0]));
    }
}
