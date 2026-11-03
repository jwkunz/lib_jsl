//! Generates the figures (SVG) and tables (Markdown) used by the book, by calling the library.
//!
//! Run from the repository root with `cargo run -p lib_jsl_book_generator --release`; outputs go
//! to `book/src/generated/`. Set `GEN_ONLY=linear_algebra` (a comma-separated list of chapter
//! names) to regenerate only some chapters.

mod derivatives;
mod fft;
mod integration;
mod interpolation;
mod linear_algebra;
mod optimization;
mod optimization_more;
mod random;
mod root_finding;
mod special_functions;
mod statistics;
mod svg;

use std::path::{Path, PathBuf};

fn output_dir(chapter: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../src/generated").join(chapter);
    std::fs::create_dir_all(&dir).expect("create output directory");
    dir
}

pub fn write(chapter: &str, name: &str, contents: &str) {
    let path = output_dir(chapter).join(name);
    std::fs::write(&path, contents).expect("write output");
    println!("wrote {}", path.display());
}

fn main() {
    // `GEN_ONLY=name` (a comma-separated list) regenerates only the named chapters.
    let only = std::env::var("GEN_ONLY").ok();
    let wanted = |name: &str| only.as_deref().is_none_or(|list| list.split(',').any(|n| n == name));
    let chapters: [(&str, fn()); 11] = [
        ("derivatives", derivatives::generate),
        ("interpolation", interpolation::generate),
        ("integration", integration::generate),
        ("root_finding", root_finding::generate),
        ("optimization", optimization::generate),
        ("optimization_more", optimization_more::generate),
        ("special_functions", special_functions::generate),
        ("linear_algebra", linear_algebra::generate),
        ("fft", fft::generate),
        ("random", random::generate),
        ("statistics", statistics::generate),
    ];
    for (name, generate) in chapters {
        if wanted(name) {
            generate();
        }
    }
}
