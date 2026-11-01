//! Dynamic programming: sequence problems and a Markov decision process.
//!
//! Run with `cargo run -p lib_jsl_optimization --example optimization_dynamic_programming`.

use lib_jsl_optimization::dynamic_programming::{FiniteMdp, edit_distance, knapsack_01};

fn main() {
    let (value, items) = knapsack_01(&[5, 4, 6, 3], &[10.0, 40.0, 30.0, 50.0], 10).unwrap();
    println!("knapsack: value {value} from items {items:?}");

    let (d, script) = edit_distance(b"intention", b"execution");
    println!("edit distance intention → execution: {d} ({} operations in the script)", script.len());

    // A machine that is good (state 0) or worn (state 1). Running it earns 10 when good and 4
    // when worn, and a good machine wears out with probability 0.3; repairing costs 6 (earning
    // −6 for the period) and restores it. Discount 0.9 per period.
    let mdp = FiniteMdp {
        transitions: vec![
            vec![vec![(0, 0.7), (1, 0.3)], vec![(0, 1.0)]],
            vec![vec![(1, 1.0)], vec![(0, 1.0)]],
        ],
        rewards: vec![vec![10.0, -6.0], vec![4.0, -6.0]],
    };
    let solution = mdp.value_iteration(0.9, 1e-10, 10_000).unwrap();
    let names = ["run", "repair"];
    println!("good: {} (value {:.3}); worn: {} (value {:.3})", names[solution.policy[0]], solution.values[0], names[solution.policy[1]], solution.values[1]);
}
