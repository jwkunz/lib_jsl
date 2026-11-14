//! Distances between observations, following `scipy.spatial.distance`: the standard metrics, the
//! condensed pairwise distances of one set (`pdist`), the distances between two sets (`cdist`),
//! and conversion between condensed and square forms (`squareform`).
//!
//! See the book entry [More descriptive statistics and robust regression](https://jwkunz.github.io/lib_jsl/book/statistics/robust-regression.html#distances).

use lib_jsl_core::ErrorsJSL;

/// A distance between two vectors `u` and `v`.
#[derive(Clone, Debug, PartialEq)]
pub enum Metric {
    /// `‖u − v‖₂`.
    Euclidean,
    /// `‖u − v‖₂²`.
    SqEuclidean,
    /// `Σ |uᵢ − vᵢ|` (Manhattan).
    Cityblock,
    /// `max |uᵢ − vᵢ|`.
    Chebyshev,
    /// `(Σ |uᵢ − vᵢ|^p)^{1/p}`, `p ≥ 1`.
    Minkowski(f64),
    /// `√(Σ (uᵢ − vᵢ)² / Vᵢ)`, with given component variances `V`.
    SEuclidean(Vec<f64>),
    /// `√((u − v)ᵀ VI (u − v))`, with the inverse covariance matrix `VI` (rows).
    Mahalanobis(Vec<Vec<f64>>),
    /// `1 − u·v / (‖u‖ ‖v‖)`.
    Cosine,
    /// `1 −` the Pearson correlation of `u` and `v`.
    Correlation,
    /// `Σ |uᵢ − vᵢ| / (|uᵢ| + |vᵢ|)`, terms with `0/0` omitted.
    Canberra,
    /// `Σ |uᵢ − vᵢ| / Σ |uᵢ + vᵢ|`.
    BrayCurtis,
    /// The fraction of components that differ.
    Hamming,
    /// Treating the vectors as booleans (non-zero or not), the fraction of components where
    /// exactly one is non-zero among those where either is (SciPy converts to booleans too).
    Jaccard,
}

// ANCHOR: distance
/// The distance between `u` and `v` under `metric` (SciPy's functions in `spatial.distance`).
pub fn distance(u: &[f64], v: &[f64], metric: &Metric) -> f64 {
    let diffs = u.iter().zip(v).map(|(a, b)| a - b);
    match metric {
        Metric::Euclidean => diffs.map(|d| d * d).sum::<f64>().sqrt(),
        Metric::SqEuclidean => diffs.map(|d| d * d).sum(),
        Metric::Cityblock => diffs.map(f64::abs).sum(),
        Metric::Chebyshev => diffs.map(f64::abs).fold(0.0, f64::max),
        Metric::Minkowski(p) => diffs.map(|d| d.abs().powf(*p)).sum::<f64>().powf(1.0 / p),
        Metric::SEuclidean(var) => diffs.zip(var).map(|(d, s)| d * d / s).sum::<f64>().sqrt(),
        Metric::Mahalanobis(vi) => {
            let d: Vec<f64> = diffs.collect();
            vi.iter().zip(&d).map(|(row, di)| di * row.iter().zip(&d).map(|(a, b)| a * b).sum::<f64>()).sum::<f64>().sqrt()
        }
        Metric::Cosine => {
            let (uv, uu, vv) = u.iter().zip(v).fold((0.0, 0.0, 0.0), |(p, q, r), (a, b)| (p + a * b, q + a * a, r + b * b));
            (1.0 - uv / (uu * vv).sqrt()).max(0.0)
        }
        Metric::Correlation => {
            let n = u.len() as f64;
            let (mu, mv) = (u.iter().sum::<f64>() / n, v.iter().sum::<f64>() / n);
            let (uv, uu, vv) = u.iter().zip(v).fold((0.0, 0.0, 0.0), |(p, q, r), (a, b)| {
                let (x, y) = (a - mu, b - mv);
                (p + x * y, q + x * x, r + y * y)
            });
            (1.0 - uv / (uu * vv).sqrt()).max(0.0)
        }
        Metric::Canberra => u.iter().zip(v).map(|(a, b)| {
            let den = a.abs() + b.abs();
            if den == 0.0 { 0.0 } else { (a - b).abs() / den }
        }).sum(),
        Metric::BrayCurtis => {
            let (num, den) = u.iter().zip(v).fold((0.0, 0.0), |(p, q), (a, b)| (p + (a - b).abs(), q + (a + b).abs()));
            num / den
        }
        Metric::Hamming => u.iter().zip(v).filter(|(a, b)| a != b).count() as f64 / u.len() as f64,
        Metric::Jaccard => {
            let (mut differ, mut nonzero) = (0usize, 0usize);
            for (a, b) in u.iter().zip(v) {
                let (p, q) = (*a != 0.0, *b != 0.0);
                if p || q {
                    nonzero += 1;
                    if p != q {
                        differ += 1;
                    }
                }
            }
            if nonzero == 0 { 0.0 } else { differ as f64 / nonzero as f64 }
        }
    }
}
// ANCHOR_END: distance

fn check_rows(rows: &[Vec<f64>]) -> Result<usize, ErrorsJSL> {
    let d = rows.first().map_or(0, |r| r.len());
    if rows.iter().any(|r| r.len() != d) {
        return Err(ErrorsJSL::InvalidInputRange("all observations must have the same length"));
    }
    Ok(d)
}

/// The condensed distance matrix of `m` observations (SciPy's `pdist`): the `m(m − 1)/2`
/// distances `d(i, j)` for `i < j`, in row order.
pub fn pdist(points: &[Vec<f64>], metric: &Metric) -> Result<Vec<f64>, ErrorsJSL> {
    check_rows(points)?;
    let m = points.len();
    let mut out = Vec::with_capacity(m * m.saturating_sub(1) / 2);
    for i in 0..m {
        for j in i + 1..m {
            out.push(distance(&points[i], &points[j], metric));
        }
    }
    Ok(out)
}

/// The `ma × mb` matrix of distances between two sets of observations (SciPy's `cdist`).
pub fn cdist(a: &[Vec<f64>], b: &[Vec<f64>], metric: &Metric) -> Result<Vec<Vec<f64>>, ErrorsJSL> {
    let (da, db) = (check_rows(a)?, check_rows(b)?);
    if !a.is_empty() && !b.is_empty() && da != db {
        return Err(ErrorsJSL::InvalidInputRange("the two sets must have the same dimension"));
    }
    Ok(a.iter().map(|u| b.iter().map(|v| distance(u, v, metric)).collect()).collect())
}

// ANCHOR: squareform
/// Expands a condensed distance vector into the symmetric square matrix with a zero diagonal
/// (SciPy's `squareform`). Its length must be `m(m − 1)/2` for some `m`.
pub fn squareform(condensed: &[f64]) -> Result<Vec<Vec<f64>>, ErrorsJSL> {
    let len = condensed.len();
    let m = ((1.0 + (1.0 + 8.0 * len as f64).sqrt()) / 2.0).round() as usize;
    if m * (m - 1) / 2 != len {
        return Err(ErrorsJSL::InvalidInputRange("the length is not m(m − 1)/2 for any m"));
    }
    // Entry (i, j), i < j, sits at position i·m − i(i + 1)/2 + (j − i − 1) of the condensed form.
    let index = |i: usize, j: usize| i * m - i * (i + 1) / 2 + (j - i - 1);
    Ok((0..m).map(|i| (0..m).map(|j| match i.cmp(&j) {
        std::cmp::Ordering::Less => condensed[index(i, j)],
        std::cmp::Ordering::Greater => condensed[index(j, i)],
        std::cmp::Ordering::Equal => 0.0,
    }).collect()).collect())
}
// ANCHOR_END: squareform

/// The condensed form of a square distance matrix (SciPy's `squareform` in the other direction):
/// its entries above the diagonal, in row order. The matrix must be square and symmetric with a
/// zero diagonal.
pub fn squareform_to_condensed(matrix: &[Vec<f64>]) -> Result<Vec<f64>, ErrorsJSL> {
    let m = matrix.len();
    if matrix.iter().any(|r| r.len() != m) {
        return Err(ErrorsJSL::InvalidInputRange("the matrix must be square"));
    }
    let mut out = Vec::with_capacity(m * m.saturating_sub(1) / 2);
    for (i, row) in matrix.iter().enumerate() {
        if row[i] != 0.0 {
            return Err(ErrorsJSL::InvalidInputRange("the diagonal must be zero"));
        }
        for (j, &value) in row.iter().enumerate().skip(i + 1) {
            if value != matrix[j][i] {
                return Err(ErrorsJSL::InvalidInputRange("the matrix must be symmetric"));
            }
            out.push(value);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_on_a_simple_pair() {
        let (u, v) = ([1.0, 0.0, 2.0], [0.0, 1.0, 2.0]);
        assert!((distance(&u, &v, &Metric::Euclidean) - 2f64.sqrt()).abs() < 1e-15);
        assert_eq!(distance(&u, &v, &Metric::Cityblock), 2.0);
        assert_eq!(distance(&u, &v, &Metric::Chebyshev), 1.0);
        assert!((distance(&u, &v, &Metric::Cosine) - 0.2).abs() < 1e-15);
        assert!((distance(&u, &v, &Metric::Hamming) - 2.0 / 3.0).abs() < 1e-15);
        assert!((distance(&u, &v, &Metric::Jaccard) - 2.0 / 3.0).abs() < 1e-15);
        // Values that differ but are both non-zero count as equal booleans.
        assert_eq!(distance(&[1.0, 2.0, 0.0], &[3.0, 2.0, 0.0], &Metric::Jaccard), 0.0);
        assert_eq!(distance(&u, &v, &Metric::Canberra), 2.0);
        assert!((distance(&u, &v, &Metric::Minkowski(3.0)) - 2f64.powf(1.0 / 3.0)).abs() < 1e-15);
        let id = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        assert!((distance(&u, &v, &Metric::Mahalanobis(id)) - 2f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn condensed_and_square_forms() {
        let pts = vec![vec![0.0, 0.0], vec![3.0, 4.0], vec![6.0, 8.0], vec![0.0, 1.0]];
        let d = pdist(&pts, &Metric::Euclidean).unwrap();
        assert_eq!(d.len(), 6);
        assert_eq!(d[0], 5.0);
        let sq = squareform(&d).unwrap();
        assert_eq!(sq[2][0], 10.0);
        assert_eq!(squareform_to_condensed(&sq).unwrap(), d);
        let c = cdist(&pts, &pts, &Metric::Euclidean).unwrap();
        assert_eq!(c, sq);
        assert!(squareform(&[1.0, 2.0]).is_err());
    }
}
