//! Generates the figures (SVG) and tables (Markdown) used by the book, by calling the library.
//!
//! Run from the repository root with `cargo run -p lib_jsl_book_generator --release`; outputs go
//! to `book/src/generated/`.

mod derivatives;
mod integration;
mod interpolation;
mod optimization;
mod optimization_more;
mod root_finding;
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
    derivatives::generate();
    interpolation::generate();
    integration::generate();
    root_finding::generate();
    optimization::generate();
    optimization_more::generate();
}
