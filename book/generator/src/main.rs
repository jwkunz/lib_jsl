//! Generates the figures (SVG) and tables (Markdown) used by the book, by calling the library.
//!
//! Run from the repository root with `cargo run -p lib_jsl_book_generator --release`; outputs go
//! to `book/src/generated/`. Set `GEN_ONLY=linear_algebra` (a comma-separated list of chapter
//! names) to regenerate only some chapters.

mod coding;
mod comms;
mod continuous_time;
mod derivatives;
mod distributions;
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
mod bounded;
mod linear_algebra;
mod matrix_equations;
mod machine_learning;
mod more_tests;
mod number_theory;
mod optimization;
mod optimization_more;
mod random;
mod raster;
mod refs;
mod signal_tools;
mod sorting;
mod root_finding;
mod sampled;
mod spectral;
mod special_functions;
mod statistics;
mod sparse;
mod utilities;
mod piecewise;
mod sparse2;
mod eigs;
mod qmcnk;
mod decomp;
mod genstruct;
mod stats27;
mod calc28;
mod fitpack29;
mod geograph210;
mod opt211;
mod io212;
mod qz213;
mod mv214;
mod sys215;
mod opt216;
mod geo217;
mod glob218;
mod optim2;
mod signal2;
mod quadode;
mod subspaces;
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
    let chapters: [(&str, fn()); 61] = [
        ("derivatives", derivatives::generate),
        ("interpolation", interpolation::generate),
        ("b_splines", b_splines::generate),
        ("integration", integration::generate),
        ("ivp", ivp::generate),
        ("sampled", sampled::generate),
        ("root_finding", root_finding::generate),
        ("optimization", optimization::generate),
        ("bounded", bounded::generate),
        ("optimization_more", optimization_more::generate),
        ("special_functions", special_functions::generate),
        ("linear_algebra", linear_algebra::generate),
        ("matrix_equations", matrix_equations::generate),
        ("subspaces", subspaces::generate),
        ("sparse", sparse::generate),
        ("utilities", utilities::generate),
        ("piecewise", piecewise::generate),
        ("sparse2", sparse2::generate),
        ("optim2", optim2::generate),
        ("signal2", signal2::generate),
        ("quadode", quadode::generate),
        ("eigs", eigs::generate),
        ("qmcnk", qmcnk::generate),
        ("decomp", decomp::generate),
        ("genstruct", genstruct::generate),
        ("stats27", stats27::generate),
        ("calc28", calc28::generate),
        ("fitpack29", fitpack29::generate),
        ("geograph210", geograph210::generate),
        ("opt211", opt211::generate),
        ("io212", io212::generate),
        ("qz213", qz213::generate),
        ("mv214", mv214::generate),
        ("sys215", sys215::generate),
        ("opt216", opt216::generate),
        ("geo217", geo217::generate),
        ("glob218", glob218::generate),
        ("fft", fft::generate),
        ("random", random::generate),
        ("statistics", statistics::generate),
        ("distributions", distributions::generate),
        ("more_tests", more_tests::generate),
        ("machine_learning", machine_learning::generate),
        ("unsupervised", unsupervised::generate),
        ("learning", learning::generate),
        ("filters", filters::generate),
        ("spectral", spectral::generate),
        ("signal_tools", signal_tools::generate),
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
