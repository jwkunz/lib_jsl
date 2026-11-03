//! Gaussian mixture models: soft assignments, densities, and choosing the number of components by BIC.

use lib_jsl_machine_learning::common::Clusterer;
use lib_jsl_machine_learning::datasets::make_blobs;
use lib_jsl_machine_learning::unsupervised::clustering::{CovarianceType, GaussianMixture};
use ndarray::arr2;

fn main() {
    let (x, _) = make_blobs(150, &[vec![0.0, 0.0], vec![4.0, 1.0], vec![1.0, 4.0]], 0.9, 2);
    for k in 1..=5 {
        let mut g = GaussianMixture::new(k, CovarianceType::Full, 1);
        g.fit_predict(x.view()).unwrap();
        println!("k = {k}: mean log-likelihood {:.4}, BIC {:7.1}, {} EM iterations", g.mean_log_likelihood(), g.bic(), g.iterations());
    }
    let mut g = GaussianMixture::new(3, CovarianceType::Full, 1);
    g.fit_predict(x.view()).unwrap();
    println!("weights {:.3?}", g.weights());
    let probe = arr2(&[[0.0, 0.0], [2.0, 0.5], [10.0, 10.0]]);
    let p = g.predict_proba(probe.view()).unwrap();
    let density = g.score_samples(probe.view()).unwrap();
    for (i, d) in density.iter().enumerate() {
        println!("point {:?}: memberships {:.3}, log density {d:.2}", probe.row(i).to_vec(), p.row(i));
    }
}
