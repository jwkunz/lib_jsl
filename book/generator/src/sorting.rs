//! Figures and tables for the chapter on sorting and selection.

use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_sorting::{equivalence_classes, heapsort, quicksort, select_m_largest, shell_sort};
use std::cell::Cell;
use std::cmp::Ordering;
use std::time::Instant;

const CHAPTER: &str = "sorting";

thread_local! {
    static COMPARISONS: Cell<u64> = const { Cell::new(0) };
}

/// A key that counts how often it is compared.
#[derive(Clone, PartialEq, Eq)]
struct Counted(u64);

impl PartialOrd for Counted {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Counted {
    fn cmp(&self, other: &Self) -> Ordering {
        COMPARISONS.with(|c| c.set(c.get() + 1));
        self.0.cmp(&other.0)
    }
}

fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        s >> 11
    }
}

type Sorter = fn(&mut [Counted]);
type KeySorter = fn(&mut [u64]);

fn sorters() -> [(&'static str, Sorter); 4] {
    [("quicksort", |s| quicksort(s)), ("heapsort", |s| heapsort(s)), ("Shell's method", |s| shell_sort(s)), ("std sort_unstable", |s| s.sort_unstable())]
}

fn input(kind: &str, n: usize, seed: u64) -> Vec<u64> {
    let mut r = rng(seed);
    match kind {
        "random" => (0..n).map(|_| r()).collect(),
        "sorted" => (0..n as u64).collect(),
        "reversed" => (0..n as u64).rev().collect(),
        "nearly sorted" => {
            let mut v: Vec<u64> = (0..n as u64).collect();
            for _ in 0..n / 100 {
                let (a, b) = (r() as usize % n, r() as usize % n);
                v.swap(a, b);
            }
            v
        }
        "three distinct keys" => (0..n).map(|_| r() % 3).collect(),
        _ => unreachable!(),
    }
}

pub fn generate() {
    comparison_counts();
    timing_table();
    selection_table();
}

fn comparison_counts() {
    let mut plot = Plot::new("Comparisons per n log₂ n, random keys", "n", "comparisons / (n log₂ n)");
    plot.log_x = true;
    for (k, (name, sort)) in sorters().into_iter().enumerate() {
        let mut pts = Vec::new();
        for e in 4..=20 {
            let n = 1usize << e;
            let mut v: Vec<Counted> = input("random", n, e as u64).into_iter().map(Counted).collect();
            COMPARISONS.with(|c| c.set(0));
            sort(&mut v);
            assert!(v.windows(2).all(|w| w[0].0 <= w[1].0));
            let c = COMPARISONS.with(|c| c.get()) as f64;
            pts.push((n as f64, c / (n as f64 * (n as f64).log2())));
        }
        plot.series.push(Series::new(name, pts, Style::LineMarkers, k));
    }
    // The information-theoretic minimum, log₂ n! ≈ n log₂ n − n log₂ e.
    let bound: Vec<(f64, f64)> = (4..=20).map(|e| { let n = (1u64 << e) as f64; (n, 1.0 - std::f64::consts::LOG2_E / n.log2()) }).collect();
    plot.series.push(Series::new("lower bound log₂ n!", bound, Style::Dashed, 6));
    plot.legend = Legend::BottomRight;
    plot.y_range = Some((0.0, 2.5));
    write(CHAPTER, "comparisons.svg", &plot.render());
}

fn timing_table() {
    let n = 1_000_000;
    let kinds = ["random", "nearly sorted", "sorted", "reversed", "three distinct keys"];
    let mut rows = Vec::new();
    let all: [(&str, KeySorter); 5] = [("quicksort", |s| quicksort(s)), ("heapsort", |s| heapsort(s)), ("Shell's method", |s| shell_sort(s)), ("std `sort_unstable`", |s| s.sort_unstable()), ("std `sort` (stable)", |s| s.sort())];
    for (name, sort) in all {
        let mut row = vec![name.to_string()];
        for kind in kinds {
            let data = input(kind, n, 7);
            // The best of three runs.
            let best = (0..3)
                .map(|_| {
                    let mut v = data.clone();
                    let t0 = Instant::now();
                    sort(&mut v);
                    let t = t0.elapsed().as_secs_f64();
                    assert!(v.windows(2).all(|w| w[0] <= w[1]));
                    t
                })
                .fold(f64::INFINITY, f64::min);
            row.push(format!("{:.1} ms", 1e3 * best));
        }
        rows.push(row);
    }
    let mut header = vec!["one million 64-bit keys"];
    header.extend(kinds);
    write(CHAPTER, "timing.md", &md_table(&header, &rows));
}

fn selection_table() {
    let n = 1_000_000;
    let data = input("random", n, 3);
    let mut rows = Vec::new();
    for m in [10usize, 1000, 100_000] {
        let t0 = Instant::now();
        let top = select_m_largest(&data, m);
        let t_heap = t0.elapsed().as_secs_f64();
        let mut v = data.clone();
        let t0 = Instant::now();
        v.sort_unstable_by(|a, b| b.cmp(a));
        let t_sort = t0.elapsed().as_secs_f64();
        assert_eq!(top, v[..m]);
        rows.push(vec![format!("{m}"), format!("{:.1} ms", 1e3 * t_heap), format!("{:.1} ms", 1e3 * t_sort)]);
    }
    write(CHAPTER, "selection.md", &md_table(&["m largest of one million", "`select_m_largest`", "sorting everything"], &rows));
    // Equivalence classes are O(n²) in the number of items.
    let mut rows = Vec::new();
    for n in [500usize, 1000, 2000] {
        let items: Vec<u64> = (0..n as u64).collect();
        let t0 = Instant::now();
        let classes = equivalence_classes(&items, |a, b| a % 7 == b % 7);
        rows.push(vec![format!("{n}"), format!("{}", classes.len()), format!("{:.1} ms", 1e3 * t0.elapsed().as_secs_f64())]);
    }
    write(CHAPTER, "equivalence.md", &md_table(&["items", "classes", "time"], &rows));
}
