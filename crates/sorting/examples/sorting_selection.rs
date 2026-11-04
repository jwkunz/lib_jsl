//! Selecting the largest elements, and grouping items into equivalence classes.

use lib_jsl_sorting::{equivalence_classes, select_m_largest};

fn main() {
    let scores = [72, 95, 61, 88, 95, 79, 100, 54, 83];
    println!("top three scores: {:?}", select_m_largest(&scores, 3));

    // Words are equivalent when they are anagrams; classes keep their first appearance order.
    let words = ["listen", "google", "enlist", "inlets", "banana", "silent", "elgoog"];
    let sorted = |w: &&str| { let mut c: Vec<char> = w.chars().collect(); c.sort_unstable(); c };
    let anagrams = equivalence_classes(&words, |a, b| sorted(a) == sorted(b));
    println!("anagram classes: {anagrams:?}");

    // The relation need only link some pairs: classes are its transitive closure. Here numbers
    // within 2 of each other are linked, so chains of close numbers form one class.
    let readings = [1.0, 2.5, 4.0, 10.0, 11.5, 30.0];
    let clusters = equivalence_classes(&readings, |a: &f64, b: &f64| (a - b).abs() <= 2.0);
    println!("chained clusters: {clusters:?}");
}
