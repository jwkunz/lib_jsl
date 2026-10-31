//! Helpers shared by the demonstrations: text scatter plots and tables.

#![allow(dead_code)] // each example uses only some of the helpers

use lib_jsl_core::R2D;
use lib_jsl_utils::terminal_plot::{PlotOptions, PlotSeries, render_terminal_plot};

/// Prints a coloured scatter plot of the first two columns of `points`, one series per label
/// (`None` is drawn as the series "noise").
pub fn scatter(title: &str, points: &R2D, labels: &[Option<usize>]) {
    let groups: Vec<Option<usize>> = {
        let mut g: Vec<Option<usize>> = labels.iter().copied().collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        g.sort();
        g
    };
    let names: Vec<String> = groups.iter().map(|g| g.map_or("noise".to_string(), |c| format!("cluster {c}"))).collect();
    let data: Vec<(Vec<f64>, Vec<f64>)> = groups
        .iter()
        .map(|g| {
            let idx: Vec<usize> = (0..points.nrows()).filter(|&i| labels[i] == *g).collect();
            (idx.iter().map(|&i| points[[i, 0]]).collect(), idx.iter().map(|&i| points[[i, 1]]).collect())
        })
        .collect();
    let series: Vec<PlotSeries> = data.iter().zip(&names).map(|((x, y), n)| PlotSeries { label: n, x, y }).collect();
    let opts = PlotOptions { width: 56, height: 14, title: Some(title.to_string()), color: true, scatter: true, ..Default::default() };
    println!("{}\n", render_terminal_plot(&series, &opts));
}

/// Wraps true class labels as clustering-style labels.
pub fn as_labels(labels: &[usize]) -> Vec<Option<usize>> {
    labels.iter().map(|&l| Some(l)).collect()
}

/// Prints a titled two-column table.
pub fn table(title: &str, rows: &[(String, String)]) {
    println!("{title}");
    let width = rows.iter().map(|r| r.0.len()).max().unwrap_or(0);
    for (name, value) in rows {
        println!("  {name:<width$}  {value}");
    }
    println!();
}
