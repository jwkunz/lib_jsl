//! Figures and tables for the image chapter's morphology, distance-transform, labelling, rank-filter,
//! derivative-filter and geometric-transform entries (1.3.0), and their comparison with
//! `scipy.ndimage`. Written into the `image` chapter's directory; run with `GEN_ONLY=image_ndimage`.

use std::collections::HashMap;

use lib_jsl_dsp::discrete::image::boundary::Mode;
use lib_jsl_dsp::discrete::image::derivative_filters::{gaussian_filter, gaussian_gradient_magnitude, gaussian_laplace, laplace};
use lib_jsl_dsp::discrete::image::distance::{ChamferMetric, distance_transform_cdt, distance_transform_edt};
use lib_jsl_dsp::discrete::image::geometric::{affine_transform, map_coordinates, rotate, shift, zoom};
use lib_jsl_dsp::discrete::image::labelling::{label, regions};
use lib_jsl_dsp::discrete::image::morphology::*;
use lib_jsl_dsp::discrete::image::rank_filters::{maximum_filter, median_filter, minimum_filter, percentile_filter, rank_filter};
use ndarray::Array2;

use crate::spectral::md_table;
use crate::write;

const CHAPTER: &str = "image";

pub fn generate() {
    reference_table();
}

fn parse() -> HashMap<String, Vec<f64>> {
    include_str!("../data/ndimage_reference.txt")
        .lines()
        .map(|l| {
            let (k, v) = l.split_once('|').unwrap();
            let vals = v
                .split(',')
                .map(|s| match s {
                    "inf" => f64::INFINITY,
                    "-inf" => f64::NEG_INFINITY,
                    _ => s.parse().unwrap(),
                })
                .collect();
            (k.to_string(), vals)
        })
        .collect()
}

pub(crate) fn test_images() -> (Array2<f64>, Array2<bool>) {
    let refs = parse();
    let grey = Array2::from_shape_vec((20, 24), refs["image grey"].clone()).unwrap();
    let binary = Array2::from_shape_vec((20, 24), refs["image binary"].iter().map(|&v| v != 0.0).collect()).unwrap();
    (grey, binary)
}

fn mode(name: &str) -> Mode {
    match name {
        "nearest" => Mode::Nearest,
        "mirror" => Mode::Mirror,
        "wrap" => Mode::Wrap,
        "constant" => Mode::Constant(0.0),
        _ => Mode::Reflect,
    }
}

fn b2f(a: Array2<bool>) -> Vec<f64> {
    a.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect()
}

fn flat(a: Array2<f64>) -> Vec<f64> {
    a.iter().copied().collect()
}

fn disk5() -> Array2<bool> {
    disk(2)
}

/// The library's value for a reference case.
fn library_case(name: &str, grey: &Array2<f64>, binary: &Array2<bool>) -> Option<Vec<f64>> {
    let words: Vec<&str> = name.split(' ').collect();
    let structure = |s: &str| if s == "cross" { cross() } else { square(3) };
    Some(match words[0] {
        "binary_erosion" if name.ends_with("2 iterations") => b2f(binary_erosion(binary, &cross(), 2).ok()?),
        "binary_dilation" if words[1] == "disk5" => b2f(binary_dilation(binary, &disk5(), 1).ok()?),
        "binary_erosion" => b2f(binary_erosion(binary, &structure(words[1]), 1).ok()?),
        "binary_dilation" => b2f(binary_dilation(binary, &structure(words[1]), 1).ok()?),
        "binary_opening" => b2f(binary_opening(binary, &structure(words[1]), 1).ok()?),
        "binary_closing" => b2f(binary_closing(binary, &structure(words[1]), 1).ok()?),
        "binary_fill_holes" => b2f(binary_fill_holes(binary, &structure(words[1])).ok()?),
        "grey_erosion" => flat(grey_erosion(grey, &square(3), mode(words[2])).ok()?),
        "grey_dilation" => flat(grey_dilation(grey, &disk5(), mode(words[2])).ok()?),
        "grey_opening" => flat(grey_opening(grey, &square(3), Mode::Reflect).ok()?),
        "grey_closing" => flat(grey_closing(grey, &square(3), Mode::Reflect).ok()?),
        "white_tophat" => flat(white_tophat(grey, &disk5(), Mode::Reflect).ok()?),
        "black_tophat" => flat(black_tophat(grey, &disk5(), Mode::Reflect).ok()?),
        "morphological_gradient" => flat(morphological_gradient(grey, &square(3), Mode::Reflect).ok()?),
        "distance_transform_edt" if words.len() > 1 => flat(distance_transform_edt(binary, Some((2.0, 0.5))).ok()?.distances),
        "distance_transform_edt" => flat(distance_transform_edt(binary, None).ok()?.distances),
        "distance_transform_cdt" => {
            let metric = if words[1] == "taxicab" { ChamferMetric::Taxicab } else { ChamferMetric::Chessboard };
            distance_transform_cdt(binary, metric).iter().map(|v| v.map_or(-1.0, |d| d as f64)).collect()
        }
        "label" => {
            let (labels, n) = label(binary, &structure(words[1])).ok()?;
            if words.len() == 2 {
                return Some(labels.iter().map(|&l| l as f64).collect());
            }
            let r = regions(&labels, n, Some(grey)).ok()?;
            match words[2] {
                "sizes" => r.iter().map(|g| g.area as f64).collect(),
                "sum" => r.iter().map(|g| g.sum.unwrap()).collect(),
                "mean" => r.iter().map(|g| g.mean.unwrap()).collect(),
                "minimum" => r.iter().map(|g| g.minimum.unwrap()).collect(),
                "maximum" => r.iter().map(|g| g.maximum.unwrap()).collect(),
                "center_of_mass" => r.iter().flat_map(|g| { let c = g.center_of_mass.unwrap(); [c.0, c.1] }).collect(),
                _ => r.iter().flat_map(|g| [g.bbox.0 as f64, g.bbox.1 as f64, g.bbox.2 as f64, g.bbox.3 as f64]).collect(),
            }
        }
        "median_filter" => {
            let size = match words[1] { "3x3" => (3, 3), "4x4" => (4, 4), _ => (2, 5) };
            flat(median_filter(grey, size, mode(words.get(2).copied().unwrap_or("reflect"))).ok()?)
        }
        "percentile_filter" if words[1] == "25" => flat(percentile_filter(grey, 25.0, (5, 5), Mode::Reflect).ok()?),
        "percentile_filter" => flat(percentile_filter(grey, 100.0, (3, 3), Mode::Reflect).ok()?),
        "minimum_filter" => flat(minimum_filter(grey, (3, 2), Mode::Reflect).ok()?),
        "maximum_filter" => flat(maximum_filter(grey, (2, 3), Mode::Reflect).ok()?),
        "rank_filter" => flat(rank_filter(grey, -2, &disk5(), Mode::Reflect).ok()?),
        "gaussian_filter" => match name {
            "gaussian_filter (1, 2.5) order (0, 1)" => flat(gaussian_filter(grey, (1.0, 2.5), (0, 1), Mode::Reflect, 4.0).ok()?),
            "gaussian_filter 2 order (2, 0)" => flat(gaussian_filter(grey, (2.0, 2.0), (2, 0), Mode::Reflect, 4.0).ok()?),
            "gaussian_filter 1.2 order (1, 1)" => flat(gaussian_filter(grey, (1.2, 1.2), (1, 1), Mode::Reflect, 4.0).ok()?),
            _ => flat(gaussian_filter(grey, (1.5, 1.5), (0, 0), mode(words[2]), 4.0).ok()?),
        },
        "gaussian_laplace" => flat(gaussian_laplace(grey, 1.5, Mode::Reflect).ok()?),
        "gaussian_gradient_magnitude" => flat(gaussian_gradient_magnitude(grey, 1.5, Mode::Reflect).ok()?),
        "laplace" => flat(laplace(grey, Mode::Reflect).ok()?),
        "map_coordinates" => {
            let pts = [(-0.7, 2.5), (0.0, 0.0), (3.3, 7.8), (10.25, 15.5), (19.0, 23.0), (19.4, 23.2), (20.6, -1.2), (8.5, 23.9), (-2.0, 30.0), (12.0, 0.4)];
            map_coordinates(grey, &pts, words[2].parse().ok()?, mode(words[3])).ok()?
        }
        "affine_transform" => flat(affine_transform(grey, [[1.0, 0.3], [-0.2, 0.9]], (1.5, -2.0), (18, 26), 3, Mode::Constant(0.0)).ok()?),
        "shift" if name.ends_with("nearest") => flat(shift(grey, (2.5, -1.25), 1, Mode::Nearest).ok()?),
        "shift" => flat(shift(grey, (2.5, -1.25), 3, Mode::Constant(0.0)).ok()?),
        "zoom" if words[1] == "(1.5," => flat(zoom(grey, (1.5, 0.75), 3, Mode::Constant(0.0)).ok()?),
        "zoom" => flat(zoom(grey, (0.5, 2.0), 1, Mode::Constant(0.0)).ok()?),
        "rotate" if name.ends_with("shape") => {
            let r = rotate(grey, 30.0, true, 3, Mode::Constant(0.0)).ok()?;
            vec![r.dim().0 as f64, r.dim().1 as f64]
        }
        "rotate" if words[1] == "30" => flat(rotate(grey, 30.0, true, 3, Mode::Constant(0.0)).ok()?),
        "rotate" => flat(rotate(grey, -20.0, false, 1, Mode::Nearest).ok()?),
        _ => return None,
    })
}

fn reference_table() {
    let refs = parse();
    let (grey, binary) = test_images();
    let mut order: Vec<&String> = refs.keys().filter(|k| !k.starts_with("image ")).collect();
    let text = include_str!("../data/ndimage_reference.txt");
    order.sort_by_key(|k| text.find(&format!("\n{k}|")).unwrap_or(0));
    let mut rows = Vec::new();
    for name in order {
        let want = &refs[name];
        let cell = match library_case(name, &grey, &binary) {
            None => "missing".to_string(),
            Some(got) if got.len() != want.len() => format!("length {} against {}", got.len(), want.len()),
            Some(got) => {
                let scale = want.iter().filter(|v| v.is_finite()).fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
                let err = got.iter().zip(want).map(|(g, w)| if g == w { 0.0 } else { (g - w).abs() }).fold(0.0, f64::max) / scale;
                if err == 0.0 { "identical".to_string() } else { format!("{err:.1e}") }
            }
        };
        rows.push(vec![name.clone(), cell]);
    }
    write(CHAPTER, "ndimage_reference.md", &md_table(&["scipy.ndimage function and case", "largest difference ÷ largest value"], &rows));
}
