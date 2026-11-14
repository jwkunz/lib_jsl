//! Generates the figures (SVG) and tables (Markdown) used by the book, by calling the library.
//!
//! Run from the repository root with `cargo run -p lib_jsl_book_generator --release`; outputs go
//! to `book/src/generated/`. Set `GEN_ONLY=linear_algebra` (a comma-separated list of chapter
//! names) to regenerate only some chapters.

mod coding;
mod comms;
mod continuous_time;
mod derivatives;
mod em_arrays;
mod em_waves;
mod fft;
mod foundations;
mod filters;
mod draw;
mod geometry;
mod graph;
mod image;
mod image_ndimage;
mod information;
mod integration;
mod interpolation;
mod ivp;
mod learning;
mod b_splines;
mod linear_algebra;
mod matrix_equations;
mod machine_learning;
mod number_theory;
mod optimization;
mod optimization_more;
mod random;
mod raster;
mod refs;
mod sorting;
mod root_finding;
mod spectral;
mod special_functions;
mod statistics;
mod svg;
mod unsupervised;

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
    // `INTEROP_OUT=dir` only writes files with the library's encoders, for
    // `data/make_foundations_refs.py` to read back with Python's libraries.
    if let Some(dir) = std::env::var_os("INTEROP_OUT") {
        foundations::write_interop_files(Path::new(&dir));
        return;
    }
    // `GEN_ONLY=name` (a comma-separated list) regenerates only the named chapters.
    let only = std::env::var("GEN_ONLY").ok();
    let wanted = |name: &str| only.as_deref().is_none_or(|list| list.split(',').any(|n| n == name));
    let chapters: [(&str, fn()); 32] = [
        ("derivatives", derivatives::generate),
        ("interpolation", interpolation::generate),
        ("b_splines", b_splines::generate),
        ("integration", integration::generate),
        ("ivp", ivp::generate),
        ("root_finding", root_finding::generate),
        ("optimization", optimization::generate),
        ("optimization_more", optimization_more::generate),
        ("special_functions", special_functions::generate),
        ("linear_algebra", linear_algebra::generate),
        ("matrix_equations", matrix_equations::generate),
        ("fft", fft::generate),
        ("random", random::generate),
        ("statistics", statistics::generate),
        ("machine_learning", machine_learning::generate),
        ("unsupervised", unsupervised::generate),
        ("learning", learning::generate),
        ("filters", filters::generate),
        ("spectral", spectral::generate),
        ("comms", comms::generate),
        ("continuous_time", continuous_time::generate),
        ("image", image::generate),
        ("image_ndimage", image_ndimage::generate),
        ("information", information::generate),
        ("coding", coding::generate),
        ("em-arrays", em_arrays::generate),
        ("em-waves", em_waves::generate),
        ("number_theory", number_theory::generate),
        ("graph", graph::generate),
        ("geometry", geometry::generate),
        ("sorting", sorting::generate),
        ("foundations", foundations::generate),
    ];
    for (name, generate) in chapters {
        if wanted(name) {
            generate();
        }
    }
}
