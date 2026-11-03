//! Model selection: k-fold and stratified cross-validation, and grid search over
//! hyperparameters.
//!
//! See the book entry [Cross-validation and grid search](https://jwkunz.github.io/lib_jsl/book/learning/model-selection.html).

use crate::common::{Classifier, MlRng, Regressor};
use lib_jsl_core::{ErrorsJSL, R1D, R2D, VR1D, VR2D};
use ndarray::Axis;

/// A train/test split as index lists.
pub type Split = (Vec<usize>, Vec<usize>);

/// `folds` train/test splits of `0..samples`: every sample is tested exactly once, and fold sizes
/// differ by at most one. With a seed the samples are shuffled first.
pub fn k_fold(samples: usize, folds: usize, shuffle_seed: Option<u64>) -> Result<Vec<Split>, ErrorsJSL> {
    if folds < 2 || folds > samples {
        return Err(ErrorsJSL::InvalidInputRange("Need 2 <= folds <= samples."));
    }
    let mut order: Vec<usize> = (0..samples).collect();
    if let Some(seed) = shuffle_seed {
        MlRng::new(seed).shuffle(&mut order);
    }
    Ok((0..folds)
        .map(|f| {
            let (start, end) = (f * samples / folds, (f + 1) * samples / folds);
            let test = order[start..end].to_vec();
            let train = order[..start].iter().chain(&order[end..]).copied().collect();
            (train, test)
        })
        .collect())
}

/// Stratified k-fold: each class is dealt round-robin across the folds (after a seeded shuffle
/// within the class), so every fold has nearly the overall class proportions.
// ANCHOR: stratified
pub fn stratified_k_fold(labels: &[usize], folds: usize, seed: u64) -> Result<Vec<Split>, ErrorsJSL> {
    if folds < 2 || folds > labels.len() {
        return Err(ErrorsJSL::InvalidInputRange("Need 2 <= folds <= samples."));
    }
    let classes = labels.iter().max().map_or(0, |m| m + 1);
    let mut rng = MlRng::new(seed);
    let mut assignment = vec![0usize; labels.len()];
    let mut next = 0;
    for c in 0..classes {
        let mut members: Vec<usize> = (0..labels.len()).filter(|&i| labels[i] == c).collect();
        rng.shuffle(&mut members);
        for i in members {
            assignment[i] = next % folds;
            next += 1;
        }
    }
    Ok((0..folds).map(|f| ((0..labels.len()).filter(|&i| assignment[i] != f).collect(), (0..labels.len()).filter(|&i| assignment[i] == f).collect())).collect())
}
// ANCHOR_END: stratified

fn rows(x: VR2D, idx: &[usize]) -> R2D {
    x.select(Axis(0), idx)
}

/// Scores a regressor on each split: `make` builds a fresh model, which is fitted on the training
/// rows and scored on the test rows by `score(truth, predicted)`.
pub fn cross_validate_regressor<M, F, S>(make: F, x: VR2D, y: VR1D, splits: &[Split], score: S) -> Result<Vec<f64>, ErrorsJSL>
where
    M: Regressor,
    F: Fn() -> M,
    S: Fn(VR1D, VR1D) -> Result<f64, ErrorsJSL>,
{
    if x.nrows() != y.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((x.nrows(), y.len())));
    }
    splits
        .iter()
        .map(|(train, test)| {
            let mut model = make();
            let ytrain: R1D = train.iter().map(|&i| y[i]).collect();
            model.fit(rows(x, train).view(), ytrain.view())?;
            let predicted = model.predict(rows(x, test).view())?;
            let truth: R1D = test.iter().map(|&i| y[i]).collect();
            score(truth.view(), predicted.view())
        })
        .collect()
}

/// Scores a classifier on each split (see [`cross_validate_regressor`]).
pub fn cross_validate_classifier<M, F, S>(make: F, x: VR2D, labels: &[usize], splits: &[Split], score: S) -> Result<Vec<f64>, ErrorsJSL>
where
    M: Classifier,
    F: Fn() -> M,
    S: Fn(&[usize], &[usize]) -> Result<f64, ErrorsJSL>,
{
    if x.nrows() != labels.len() {
        return Err(ErrorsJSL::IncompatibleArraySizes((x.nrows(), labels.len())));
    }
    splits
        .iter()
        .map(|(train, test)| {
            let mut model = make();
            let ltrain: Vec<usize> = train.iter().map(|&i| labels[i]).collect();
            model.fit(rows(x, train).view(), &ltrain)?;
            let predicted = model.predict(rows(x, test).view())?;
            let truth: Vec<usize> = test.iter().map(|&i| labels[i]).collect();
            score(&truth, &predicted)
        })
        .collect()
}

/// Every combination of the values on each axis (the Cartesian product), first axis slowest.
pub fn parameter_grid(axes: &[Vec<f64>]) -> Vec<Vec<f64>> {
    axes.iter().fold(vec![Vec::new()], |acc, axis| acc.iter().flat_map(|prefix| axis.iter().map(move |&v| [prefix.as_slice(), &[v]].concat())).collect())
}

/// The outcome of a grid search.
#[derive(Debug, Clone)]
pub struct GridSearchResult<P> {
    pub best: P,
    pub best_score: f64,
    /// Every candidate with its score, in the order given.
    pub scores: Vec<(P, f64)>,
}

/// Evaluates every candidate (typically by the mean of a cross-validation) and keeps the best,
/// by largest score with `greater_is_better` or smallest otherwise. Candidates whose evaluation
/// fails or returns NaN are skipped.
// ANCHOR: grid_search
pub fn grid_search<P: Clone, E: Fn(&P) -> Result<f64, ErrorsJSL>>(candidates: &[P], evaluate: E, greater_is_better: bool) -> Result<GridSearchResult<P>, ErrorsJSL> {
    let scores: Vec<(P, f64)> = candidates.iter().filter_map(|c| evaluate(c).ok().filter(|s| !s.is_nan()).map(|s| (c.clone(), s))).collect();
    let better = |a: f64, b: f64| if greater_is_better { a > b } else { a < b };
    let (best, best_score) = scores.iter().fold(None, |acc: Option<(P, f64)>, (p, s)| match acc {
        Some((_, b)) if !better(*s, b) => acc,
        _ => Some((p.clone(), *s)),
    }).ok_or(ErrorsJSL::RuntimeError("No candidate could be evaluated."))?;
    Ok(GridSearchResult { best, best_score, scores })
}
// ANCHOR_END: grid_search

/// The mean of a slice of scores.
pub fn mean_score(scores: &[f64]) -> f64 {
    scores.iter().sum::<f64>() / scores.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::{make_moons, make_regression};
    use crate::metrics::{accuracy, mean_squared_error};
    use crate::supervised::classification::KNearestNeighbors;
    use crate::supervised::regression::Ridge;

    #[test]
    fn folds_partition_the_samples() {
        let splits = k_fold(23, 5, Some(1)).unwrap();
        let mut tested: Vec<usize> = splits.iter().flat_map(|(_, t)| t.clone()).collect();
        tested.sort_unstable();
        assert_eq!(tested, (0..23).collect::<Vec<_>>());
        assert!(splits.iter().all(|(train, test)| train.len() + test.len() == 23 && (4..=5).contains(&test.len())));
        assert!(splits.iter().all(|(train, test)| test.iter().all(|i| !train.contains(i))));
        // Stratification keeps the 1 : 3 class ratio in every fold.
        let labels: Vec<usize> = (0..40).map(|i| usize::from(i % 4 != 0)).collect();
        for (_, test) in stratified_k_fold(&labels, 5, 2).unwrap() {
            assert_eq!(test.iter().filter(|&&i| labels[i] == 0).count(), 2);
            assert_eq!(test.len(), 8);
        }
        assert!(k_fold(3, 4, None).is_err() && k_fold(3, 1, None).is_err());
        assert_eq!(parameter_grid(&[vec![1.0, 2.0], vec![10.0, 20.0, 30.0]]).len(), 6);
        assert_eq!(parameter_grid(&[vec![1.0, 2.0], vec![10.0]])[1], vec![2.0, 10.0]);
    }

    #[test]
    fn grid_search_with_cross_validation_tunes_hyperparameters() {
        // Ridge on noisy linear data: heavy regularisation clearly hurts.
        let (x, y, _, _) = make_regression(120, 5, 0.5, 3);
        let splits = k_fold(120, 5, Some(4)).unwrap();
        let alphas = vec![1e-3, 1e-1, 10.0, 1e3, 1e5];
        let search = grid_search(&alphas, |&a| Ok(mean_score(&cross_validate_regressor(|| Ridge::new(a), x.view(), y.view(), &splits, mean_squared_error)?)), false).unwrap();
        assert!(search.best <= 10.0, "{:?}", search.scores);
        assert!(search.scores.last().unwrap().1 > 10.0 * search.best_score);
        // k-NN on two moons: k = 1 overfits the noise less than k = 150 underfits.
        let (xm, lm) = make_moons(300, 0.25, 5);
        let folds = stratified_k_fold(&lm, 5, 6).unwrap();
        let ks = vec![1usize, 5, 15, 150];
        let search = grid_search(&ks, |&k| Ok(mean_score(&cross_validate_classifier(|| KNearestNeighbors::new(k), xm.view(), &lm, &folds, accuracy)?)), true).unwrap();
        assert!(search.best == 5 || search.best == 15, "{:?}", search.scores);
        assert!(search.best_score > 0.9 && search.scores[3].1 < search.best_score);
    }
}
