//! Otsu thresholding, Canny edges and marker-controlled watershed segmentation.

use lib_jsl_dsp::discrete::image::filters::{gaussian_blur::gaussian_blur, sobel::sobel};
use lib_jsl_dsp::discrete::image::segmentation::canny_edge_detection::canny_edge_detection;
use lib_jsl_dsp::discrete::image::segmentation::otsu::otsu;
use lib_jsl_dsp::discrete::image::segmentation::watershed::{label_boundaries, watershed};
use ndarray::Array2;

fn main() {
    // Two touching squares of different brightness on a dark background.
    let img = Array2::from_shape_fn((40, 60), |(r, c)| {
        if !(10..30).contains(&r) { 40.0 } else if (10..30).contains(&c) { 200.0 } else if (30..50).contains(&c) { 140.0 } else { 40.0 }
    });
    let img8 = img.mapv(|v| v as u8);
    let o = otsu(&img8).unwrap();
    let fg = o.binary.iter().filter(|&&v| v > 0).count();
    println!("Otsu threshold {}, separability {:.3}, foreground pixels {fg} (true 800)", o.threshold, o.separability);
    let f = img.mapv(|v| v / 255.0);
    let edges = canny_edge_detection(&f, 1.0, 0.03, 0.07).unwrap();
    let row: String = (0..60).map(|c| if edges[(20, c)] { '|' } else { '.' }).collect();
    println!("Canny edges along row 20: {row}");
    let elevation = sobel(&gaussian_blur(&f, 1.0).unwrap(), true).unwrap().magnitude;
    let mut markers = Array2::<i32>::zeros((40, 60));
    markers[(20, 20)] = 1;
    markers[(20, 40)] = 2;
    markers[(2, 2)] = 3;
    let labels = watershed(&elevation, &markers).unwrap();
    let counts: Vec<usize> = (1..=3).map(|l| labels.iter().filter(|&&v| v == l).count()).collect();
    let boundary_cols: Vec<usize> = (0..60).filter(|&c| label_boundaries(&labels)[(20, c)]).collect();
    println!("watershed regions {counts:?} (true 400, 400, 1600); boundaries on row 20 at columns {boundary_cols:?}");
}
