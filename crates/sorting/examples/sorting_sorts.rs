//! Sorting with quicksort, heapsort and Shell's method.

use lib_jsl_sorting::{heapsort, quicksort, shell_sort};

fn main() {
    let words = ["pear", "fig", "apple", "kiwi", "banana", "cherry", "date", "apple"];
    let mut a = words.to_vec();
    quicksort(&mut a);
    println!("quicksort:      {a:?}");

    let mut temperatures = vec![21, -4, 15, 37, 0, 15, 8, -12, 29, 3];
    heapsort(&mut temperatures);
    println!("heapsort:       {temperatures:?}");

    // Any type with a total order works; here (surname, first name) pairs.
    let mut people = vec![("Noether", "Emmy"), ("Gauss", "Carl"), ("Euler", "Leonhard"), ("Gauss", "Anna")];
    shell_sort(&mut people);
    println!("Shell's method: {people:?}");

    // Floating-point numbers need a total order: sort their bit-level total ordering keys.
    let mut x = [2.5f64, -1.0, 0.0, 3.25, -7.5];
    let mut keys: Vec<i64> = x.iter().map(|v| { let b = v.to_bits() as i64; b ^ (((b >> 63) as u64) >> 1) as i64 }).collect();
    quicksort(&mut keys);
    for (slot, k) in x.iter_mut().zip(keys) {
        *slot = f64::from_bits((k ^ (((k >> 63) as u64) >> 1) as i64) as u64);
    }
    println!("floats:         {x:?}");
}
