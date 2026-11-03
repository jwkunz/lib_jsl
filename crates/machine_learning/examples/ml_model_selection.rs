//! Cross-validation and grid search: tuning an RBF support vector machine.

use lib_jsl_machine_learning::datasets::make_moons;
use lib_jsl_machine_learning::kernels::Kernel;
use lib_jsl_machine_learning::metrics::accuracy;
use lib_jsl_machine_learning::model_selection::{cross_validate_classifier, grid_search, mean_score, parameter_grid, stratified_k_fold};
use lib_jsl_machine_learning::supervised::classification::SupportVectorMachine;

fn main() {
    let (x, labels) = make_moons(150, 0.25, 4);
    let folds = stratified_k_fold(&labels, 5, 1).unwrap();
    let grid = parameter_grid(&[vec![0.1, 1.0, 10.0, 100.0], vec![0.1, 1.0, 10.0, 100.0]]);
    let search = grid_search(
        &grid,
        |p| Ok(mean_score(&cross_validate_classifier(|| SupportVectorMachine::new(Kernel::Rbf { gamma: p[1] }, p[0]), x.view(), &labels, &folds, accuracy)?)),
        true,
    )
    .unwrap();
    for (p, score) in &search.scores {
        println!("C = {:>5}, γ = {:>5}: cross-validated accuracy {score:.3}", p[0], p[1]);
    }
    println!("best: C = {}, γ = {} with {:.3}", search.best[0], search.best[1], search.best_score);
}
