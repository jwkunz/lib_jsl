//! Evaluation metrics for regression, classification and clustering.

use crate::common::{class_count, squared_distance};
use lib_jsl_core::{ErrorsJSL, VR1D, VR2D};

fn same_len(a: usize, b: usize) -> Result<(), ErrorsJSL> {
    if a != b || a == 0 { Err(ErrorsJSL::IncompatibleArraySizes((a, b))) } else { Ok(()) }
}

/// Mean squared error.
pub fn mean_squared_error(truth: VR1D, predicted: VR1D) -> Result<f64, ErrorsJSL> {
    same_len(truth.len(), predicted.len())?;
    Ok(truth.iter().zip(predicted).map(|(a, b)| (a - b) * (a - b)).sum::<f64>() / truth.len() as f64)
}

pub fn root_mean_squared_error(truth: VR1D, predicted: VR1D) -> Result<f64, ErrorsJSL> {
    mean_squared_error(truth, predicted).map(f64::sqrt)
}

pub fn mean_absolute_error(truth: VR1D, predicted: VR1D) -> Result<f64, ErrorsJSL> {
    same_len(truth.len(), predicted.len())?;
    Ok(truth.iter().zip(predicted).map(|(a, b)| (a - b).abs()).sum::<f64>() / truth.len() as f64)
}

/// The coefficient of determination `1 - SS_res / SS_tot` (1 is perfect; can be negative; a
/// constant target gives 1 for exact predictions and `-inf` otherwise).
pub fn r2_score(truth: VR1D, predicted: VR1D) -> Result<f64, ErrorsJSL> {
    same_len(truth.len(), predicted.len())?;
    let mean = truth.sum() / truth.len() as f64;
    let ss_tot: f64 = truth.iter().map(|v| (v - mean) * (v - mean)).sum();
    let ss_res: f64 = truth.iter().zip(predicted).map(|(a, b)| (a - b) * (a - b)).sum();
    Ok(if ss_tot == 0.0 {
        if ss_res == 0.0 { 1.0 } else { f64::NEG_INFINITY }
    } else {
        1.0 - ss_res / ss_tot
    })
}

/// The fraction of labels predicted correctly.
pub fn accuracy(truth: &[usize], predicted: &[usize]) -> Result<f64, ErrorsJSL> {
    same_len(truth.len(), predicted.len())?;
    Ok(truth.iter().zip(predicted).filter(|(a, b)| a == b).count() as f64 / truth.len() as f64)
}

/// The confusion matrix `m[true][predicted]` over `classes` classes.
pub fn confusion_matrix(truth: &[usize], predicted: &[usize], classes: usize) -> Result<Vec<Vec<usize>>, ErrorsJSL> {
    same_len(truth.len(), predicted.len())?;
    if truth.iter().chain(predicted).any(|&c| c >= classes) {
        return Err(ErrorsJSL::InvalidInputRange("A label is outside the class range."));
    }
    let mut m = vec![vec![0; classes]; classes];
    for (&t, &p) in truth.iter().zip(predicted) {
        m[t][p] += 1;
    }
    Ok(m)
}

/// Per-class `(precision, recall, f1)`; a class with no predictions or no members scores 0.
pub fn precision_recall_f1(truth: &[usize], predicted: &[usize], classes: usize) -> Result<Vec<(f64, f64, f64)>, ErrorsJSL> {
    let m = confusion_matrix(truth, predicted, classes)?;
    Ok((0..classes)
        .map(|c| {
            let tp = m[c][c] as f64;
            let predicted_c: usize = (0..classes).map(|t| m[t][c]).sum();
            let actual_c: usize = m[c].iter().sum();
            let p = if predicted_c == 0 { 0.0 } else { tp / predicted_c as f64 };
            let r = if actual_c == 0 { 0.0 } else { tp / actual_c as f64 };
            (p, r, if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) })
        })
        .collect())
}

/// The mean silhouette coefficient over clustered points (noise, `None`, is ignored). Values near
/// 1 mean tight, well-separated clusters; needs at least two clusters.
pub fn silhouette_score(x: VR2D, assignments: &[Option<usize>]) -> Result<f64, ErrorsJSL> {
    same_len(x.nrows(), assignments.len())?;
    let labels: Vec<usize> = assignments.iter().flatten().copied().collect();
    let k = class_count(&labels);
    let members: Vec<Vec<usize>> = (0..k).map(|c| (0..x.nrows()).filter(|&i| assignments[i] == Some(c)).collect()).collect();
    if members.iter().filter(|m| !m.is_empty()).count() < 2 {
        return Err(ErrorsJSL::InvalidInputRange("A silhouette needs at least two non-empty clusters."));
    }
    let row = |i: usize| x.row(i).to_vec();
    let mut total = 0.0;
    let mut count = 0usize;
    for i in (0..x.nrows()).filter(|&i| assignments[i].is_some()) {
        let c = assignments[i].expect("filtered");
        let ri = row(i);
        let mean_to = |idx: &[usize], skip_self: bool| {
            let others: Vec<&usize> = idx.iter().filter(|&&j| !(skip_self && j == i)).collect();
            others.iter().map(|&&j| squared_distance(&ri, &row(j)).sqrt()).sum::<f64>() / others.len().max(1) as f64
        };
        if members[c].len() < 2 {
            count += 1; // singleton clusters contribute 0
            continue;
        }
        let a = mean_to(&members[c], true);
        let b = (0..k).filter(|&o| o != c && !members[o].is_empty()).map(|o| mean_to(&members[o], false)).fold(f64::INFINITY, f64::min);
        total += (b - a) / a.max(b).max(f64::MIN_POSITIVE);
        count += 1;
    }
    Ok(total / count as f64)
}

/// The adjusted Rand index between two clusterings of the same points (1 = identical up to
/// relabelling, about 0 for random agreement). Noise is treated as one extra group.
pub fn adjusted_rand_index(a: &[Option<usize>], b: &[Option<usize>]) -> Result<f64, ErrorsJSL> {
    same_len(a.len(), b.len())?;
    let norm = |v: &[Option<usize>]| -> Vec<usize> {
        let top = v.iter().flatten().max().map_or(0, |m| m + 1);
        v.iter().map(|x| x.unwrap_or(top)).collect()
    };
    let (la, lb) = (norm(a), norm(b));
    let (ka, kb) = (class_count(&la), class_count(&lb));
    let mut table = vec![vec![0u64; kb]; ka];
    for (&x, &y) in la.iter().zip(&lb) {
        table[x][y] += 1;
    }
    let c2 = |n: u64| (n * n.saturating_sub(1)) as f64 / 2.0;
    let sum_ij: f64 = table.iter().flatten().map(|&n| c2(n)).sum();
    let sum_a: f64 = table.iter().map(|r| c2(r.iter().sum())).sum();
    let sum_b: f64 = (0..kb).map(|j| c2(table.iter().map(|r| r[j]).sum())).sum();
    let total = c2(a.len() as u64);
    let expected = sum_a * sum_b / total.max(1.0);
    let max = 0.5 * (sum_a + sum_b);
    Ok(if (max - expected).abs() < 1e-12 { 1.0 } else { (sum_ij - expected) / (max - expected) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn regression_metrics() {
        let (t, p) = (arr1(&[1.0, 2.0, 3.0, 4.0]), arr1(&[1.0, 2.0, 3.0, 6.0]));
        assert_eq!(mean_squared_error(t.view(), p.view()).unwrap(), 1.0);
        assert_eq!(root_mean_squared_error(t.view(), p.view()).unwrap(), 1.0);
        assert_eq!(mean_absolute_error(t.view(), p.view()).unwrap(), 0.5);
        assert!((r2_score(t.view(), p.view()).unwrap() - (1.0 - 4.0 / 5.0)).abs() < 1e-12);
        assert_eq!(r2_score(t.view(), t.view()).unwrap(), 1.0);
        assert!(mean_squared_error(t.view(), arr1(&[1.0]).view()).is_err());
        assert_eq!(r2_score(arr1(&[2.0, 2.0]).view(), arr1(&[2.0, 3.0]).view()).unwrap(), f64::NEG_INFINITY);
    }

    #[test]
    fn classification_metrics() {
        let (t, p) = ([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0]);
        assert!((accuracy(&t, &p).unwrap() - 4.0 / 6.0).abs() < 1e-12);
        let m = confusion_matrix(&t, &p, 3).unwrap();
        assert_eq!(m, vec![vec![1, 1, 0], vec![0, 2, 0], vec![1, 0, 1]]);
        let prf = precision_recall_f1(&t, &p, 3).unwrap();
        assert!((prf[1].0 - 2.0 / 3.0).abs() < 1e-12 && (prf[1].1 - 1.0).abs() < 1e-12);
        assert!(confusion_matrix(&t, &[0, 0, 0, 0, 0, 5], 3).is_err());
        assert!(precision_recall_f1(&[0, 0], &[0, 0], 2).unwrap()[1] == (0.0, 0.0, 0.0));
    }

    #[test]
    fn clustering_metrics() {
        let x = arr2(&[[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [5.0, 5.0], [5.1, 5.0], [5.0, 5.1]]);
        let good: Vec<Option<usize>> = vec![Some(0), Some(0), Some(0), Some(1), Some(1), Some(1)];
        let bad: Vec<Option<usize>> = vec![Some(0), Some(1), Some(0), Some(1), Some(0), Some(1)];
        assert!(silhouette_score(x.view(), &good).unwrap() > 0.95);
        assert!(silhouette_score(x.view(), &bad).unwrap() < 0.1);
        assert!(silhouette_score(x.view(), &[Some(0); 6]).is_err());
        let relabelled: Vec<Option<usize>> = good.iter().map(|g| g.map(|c| 1 - c)).collect();
        assert!((adjusted_rand_index(&good, &relabelled).unwrap() - 1.0).abs() < 1e-12);
        assert!(adjusted_rand_index(&good, &bad).unwrap() < 0.1);
        assert!(adjusted_rand_index(&good, &good[..3]).is_err());
    }
}
