//! Linear programming by the simplex method.
//!
//! Run with `cargo run -p lib_jsl_optimization --example optimization_linear_programming`.

use lib_jsl_optimization::linear_programming::{LinearProgram, LpStatus};

fn main() {
    // A diet problem: choose servings of bread, milk and cheese, costing 2, 3.5 and 8 each, to
    // provide at least 300 calories (65, 120 and 106 per serving) and 20 g of protein (2, 8 and
    // 7 g per serving) at the least cost.
    let lp = LinearProgram::new(vec![2.0, 3.5, 8.0])
        .with_greater_equal(vec![65.0, 120.0, 106.0], 300.0)
        .with_greater_equal(vec![2.0, 8.0, 7.0], 20.0)
        .with_bounds(1, 0.0, 2.0); // at most two servings of milk
    let s = lp.solve().unwrap();
    assert_eq!(s.status, LpStatus::Optimal);
    println!("servings {:?}, cost {:.4}, {} pivots", s.x, s.objective, s.iterations);

    // Infeasible and unbounded programs are reported through the status.
    let impossible = LinearProgram::new(vec![1.0]).with_greater_equal(vec![1.0], 5.0).with_bounds(0, 0.0, 2.0);
    println!("x ≥ 5 with x ≤ 2: {:?}", impossible.solve().unwrap().status);
}
