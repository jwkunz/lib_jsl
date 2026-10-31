//! Classification demonstration: seven classifiers on three datasets, plus a hidden Markov model
//! and a phylogenetic tree.
//!
//! Run with `cargo run --release -p lib_jsl_machine_learning --example classification_demo`.

mod common;

use common::table;
use lib_jsl_core::{R2D, VR2D};
use lib_jsl_machine_learning::common::{Classifier, Transformer};
use lib_jsl_machine_learning::datasets::{make_circles, make_classification, make_moons};
use lib_jsl_machine_learning::kernels::Kernel;
use lib_jsl_machine_learning::metrics::accuracy;
use lib_jsl_machine_learning::preprocessing::{StandardScaler, select_labels, select_rows, train_test_split};
use lib_jsl_machine_learning::supervised::classification::{GaussianNaiveBayes, HiddenMarkovModel, KNearestNeighbors, LogisticRegression, RandomForest, SupportVectorMachine, neighbor_joining, upgma};
use lib_jsl_machine_learning::unsupervised::reduction::Lda;
use ndarray::arr2;

fn compare(name: &str, x: &R2D, labels: &[usize]) {
    let (train, test) = train_test_split(x.nrows(), 0.3, 11).unwrap();
    let mut scaler = StandardScaler::new();
    let xtr = scaler.fit_transform(select_rows(x.view(), &train).view()).unwrap();
    let xte = scaler.transform(select_rows(x.view(), &test).view()).unwrap();
    let (ltr, lte) = (select_labels(labels, &train), select_labels(labels, &test));
    let models: Vec<(&str, Box<dyn Classifier>)> = vec![
        ("logistic regression", Box::new(LogisticRegression::new(1e-3))),
        ("LDA", Box::new(Lda::new(1.max(lte.iter().max().copied().unwrap_or(1).min(2))))),
        ("Gaussian naive Bayes", Box::new(GaussianNaiveBayes::new())),
        ("k-nearest neighbours (k=7)", Box::new(KNearestNeighbors::new(7))),
        ("SVM (RBF kernel)", Box::new(SupportVectorMachine::new(Kernel::Rbf { gamma: 1.0 }, 5.0))),
        ("random forest (80 trees)", Box::new(RandomForest::new(80, 12, 3))),
    ];
    let rows: Vec<(String, String)> = models
        .into_iter()
        .map(|(n, mut m)| {
            m.fit(xtr.view(), &ltr).unwrap();
            (n.to_string(), format!("test accuracy {:.3}", accuracy(&lte, &m.predict(xte.view() as VR2D).unwrap()).unwrap()))
        })
        .collect();
    table(&format!("== {name} =="), &rows);
}

fn main() {
    let (x, l) = make_classification(120, 5);
    compare("three overlapping Gaussian classes in 4D", &x, &l);
    let (x, l) = make_moons(150, 0.2, 5);
    compare("two noisy moons (not linearly separable)", &x, &l);
    let (x, l) = make_circles(150, 0.08, 0.45, 5);
    compare("concentric circles", &x, &l);
    println!("The linear models (logistic regression, LDA) cannot separate concentric circles at all (61%) and lose");
    println!("ground on the moons, while kernel, neighbour and tree methods adapt to the curved boundaries. (Naive Bayes");
    println!("does well on the circles only because the two rings happen to have different spreads.)\n");

    // A hidden Markov model: infer the weather from what a friend wears.
    let hmm = HiddenMarkovModel::new(vec![0.6, 0.4], arr2(&[[0.7, 0.3], [0.4, 0.6]]), arr2(&[[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])).unwrap();
    let outfit = ["walk", "shop", "clean"];
    let observations = [0usize, 1, 2, 2, 0];
    let (path, log_prob) = hmm.viterbi(&observations).unwrap();
    let states = ["Rainy", "Sunny"];
    println!("== Viterbi decoding (hidden weather from observed activity) ==");
    println!("  observed : {}", observations.iter().map(|&o| outfit[o]).collect::<Vec<_>>().join(" → "));
    println!("  decoded  : {}", path.iter().map(|&s| states[s]).collect::<Vec<_>>().join(" → "));
    println!("  path probability {:.5}, sequence likelihood {:.5}\n", log_prob.exp(), hmm.log_likelihood(&observations).unwrap().exp());

    // Phylogenetic trees from a distance matrix.
    let d = arr2(&[[0.0, 5.0, 9.0, 9.0, 8.0], [5.0, 0.0, 10.0, 10.0, 9.0], [9.0, 10.0, 0.0, 8.0, 7.0], [9.0, 10.0, 8.0, 0.0, 3.0], [8.0, 9.0, 7.0, 3.0, 0.0]]);
    let names = ["human", "chimp", "frog", "newt", "salamander"];
    println!("== phylogenetic trees from a distance matrix ==");
    println!("  UPGMA            {}", upgma(d.view(), &names).unwrap().to_newick());
    println!("  neighbour joining {}", neighbor_joining(d.view(), &names).unwrap().to_newick());
}
