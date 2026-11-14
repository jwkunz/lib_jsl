//! Figures and tables for the image chapter's morphology, distance-transform, labelling, rank-filter,
//! derivative-filter and geometric-transform entries (1.3.0), and their comparison with
//! `scipy.ndimage`. Written into the `image` chapter's directory; run with `GEN_ONLY=image_ndimage`.

use std::collections::HashMap;

use lib_jsl_dsp::discrete::image::boundary::Mode;
use lib_jsl_dsp::discrete::image::derivative_filters::{gaussian_filter, gaussian_gradient_magnitude, gaussian_laplace, laplace};
use lib_jsl_dsp::discrete::image::distance::{ChamferMetric, distance_transform_cdt, distance_transform_edt};
use lib_jsl_dsp::discrete::image::fourier::{Spectrum, fourier_ellipsoid, fourier_filter, fourier_gaussian, fourier_shift, fourier_uniform};
use lib_jsl_dsp::discrete::image::geometric::{affine_transform, map_coordinates, rotate, shift, zoom};
use lib_jsl_dsp::discrete::image::labelling::{label, regions};
use lib_jsl_dsp::discrete::image::morphology::*;
use lib_jsl_dsp::discrete::image::rank_filters::{maximum_filter, median_filter, minimum_filter, percentile_filter, rank_filter};
use lib_jsl_ffts::real_fft::rfft;
use ndarray::Array2;
use num_complex::Complex;

type C64 = Complex<f64>;

use crate::raster::{Colours, Mark, Panel, panels};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;

const CHAPTER: &str = "image";

pub fn generate() {
    reference_table();
    morphology_figures();
    rank_geometric_figures();
    fourier_figures();
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
        w0 if w0.starts_with("fourier_") => {
            let kind = &words[0]["fourier_".len()..];
            let params = match kind { "gaussian" => (1.5, 0.7), "uniform" => (3.0, 4.0), "ellipsoid" => (5.0, 3.0), _ => (2.5, -1.25) };
            let filter = |s: &Array2<C64>, spectrum: Spectrum| match kind {
                "gaussian" => fourier_gaussian(s, params, spectrum),
                "uniform" => fourier_uniform(s, params, spectrum),
                "ellipsoid" => fourier_ellipsoid(s, params, spectrum),
                _ => fourier_shift(s, params, spectrum),
            };
            if name.ends_with("real rows") {
                let rows: Vec<Vec<C64>> = grey.rows().into_iter().map(|r| rfft(&r.to_vec()).unwrap()).collect();
                let w = rows[0].len();
                // A real FFT along each row, as numpy.fft.rfft(axis=-1) gives; the untransformed
                // columns are treated as a full spectrum, as SciPy does.
                let spec = Array2::from_shape_fn((grey.nrows(), w), |(i, j)| rows[i][j]);
                let out = filter(&spec, Spectrum::RealRows { columns: grey.ncols() }).ok()?;
                out.iter().map(|z| z.re).chain(out.iter().map(|z| z.im)).collect()
            } else {
                flat(fourier_filter(grey, |s| filter(s, Spectrum::Full)).ok()?)
            }
        }
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

// ------------------------------------------------------------------ test scenes

/// A reproducible uniform generator in [0, 1).
fn uniform(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A binary scene of six objects: two discs (one with a hole), a ring, a rectangle with a small
/// hole, a bar, and two touching discs; plus scattered specks of noise and pinholes.
fn binary_scene() -> (Array2<bool>, Array2<bool>) {
    let n = 120;
    let disc = |r: f64, c: f64, cr: f64, cc: f64, rad: f64| (r - cr).powi(2) + (c - cc).powi(2) <= rad * rad;
    let clean = Array2::from_shape_fn((n, n), |(r, c)| {
        let (r, c) = (r as f64, c as f64);
        (disc(r, c, 25.0, 25.0, 14.0) && !disc(r, c, 25.0, 25.0, 4.0))
            || (disc(r, c, 28.0, 80.0, 16.0) && !disc(r, c, 28.0, 80.0, 10.0))
            || ((60.0..=95.0).contains(&r) && (12.0..=45.0).contains(&c) && !((75.0..=78.0).contains(&r) && (27.0..=30.0).contains(&c)))
            || ((65.0..=70.0).contains(&r) && (60.0..=110.0).contains(&c))
            || disc(r, c, 95.0, 75.0, 10.0)
            || disc(r, c, 95.0, 94.0, 10.0)
    });
    let mut u = uniform(5);
    let noisy = clean.mapv(|b| {
        let x = u();
        if x < 0.012 { !b } else { b }
    });
    (clean, noisy)
}

fn as_f(b: &Array2<bool>) -> Array2<f64> {
    b.mapv(|v| if v { 1.0 } else { 0.0 })
}

fn morphology_figures() {
    let (clean, noisy) = binary_scene();
    let d = disk(1);
    let opened = binary_opening(&noisy, &d, 1).unwrap();
    let cleaned = binary_closing(&opened, &d, 1).unwrap();
    let filled = binary_fill_holes(&cleaned, &cross()).unwrap();
    let g = Colours::Gray(0.0, 1.0);
    write(
        CHAPTER,
        "morphology.svg",
        &panels(
            "Cleaning a noisy binary image",
            &[
                Panel::new("noisy: 1.2% of pixels flipped", as_f(&noisy), g),
                Panel::new("opening (removes specks)", as_f(&opened), g),
                Panel::new("then closing (fills pinholes)", as_f(&cleaned), g),
                Panel::new("then hole filling", as_f(&filled), g),
            ],
            4,
            170.0,
        ),
    );

    let dt = distance_transform_edt(&cleaned, None).unwrap().distances;
    let (labels, n) = label(&cleaned, &cross()).unwrap();
    let peak = dt.iter().cloned().fold(0.0, f64::max);
    let mut lab = Panel::new(&format!("labels: {n} components"), labels.mapv(|v| v as f64), Colours::Labels);
    let regs = regions(&labels, n, None).unwrap();
    for r in &regs {
        lab.marks.push(Mark::Dot { x: r.centroid.1, y: r.centroid.0, colour: "#ffffff" });
    }
    let mut dist = Panel::new("Euclidean distance to the background", dt.clone(), Colours::Ramp(0.0, peak));
    dist.marks.push(Mark::Mask(cleaned.mapv(|b| !b), "#000000"));
    write(CHAPTER, "distance_labels.svg", &panels("Distance transform and connected components of the cleaned image", &[dist, lab], 2, 260.0));

    // Counting objects: six true objects (the two touching discs count as one connected object).
    let count = |b: &Array2<bool>, s: &Array2<bool>| label(b, s).unwrap().1;
    let truth_4 = count(&clean, &cross());
    let mut rows = vec![
        vec!["clean image".to_string(), truth_4.to_string(), count(&clean, &square(3)).to_string()],
        vec!["noisy image".to_string(), count(&noisy, &cross()).to_string(), count(&noisy, &square(3)).to_string()],
        vec!["after opening".to_string(), count(&opened, &cross()).to_string(), count(&opened, &square(3)).to_string()],
        vec!["after opening and closing".to_string(), count(&cleaned, &cross()).to_string(), count(&cleaned, &square(3)).to_string()],
    ];
    // Splitting the touching discs: threshold the distance transform.
    let cores = dt.mapv(|v| v > 6.0);
    rows.push(vec!["cores: distance > 6".to_string(), count(&cores, &cross()).to_string(), count(&cores, &square(3)).to_string()]);
    write(CHAPTER, "object_counts.md", &md_table(&["image", "components (4-connected)", "components (8-connected)"], &rows));

    let mut rows = Vec::new();
    for r in &regs {
        rows.push(vec![
            r.label.to_string(),
            r.area.to_string(),
            format!("({:.1}, {:.1})", r.centroid.0, r.centroid.1),
            format!("rows {}–{}, columns {}–{}", r.bbox.0, r.bbox.1 - 1, r.bbox.2, r.bbox.3 - 1),
            format!("{:.1}", (0..r.bbox.1 - r.bbox.0).flat_map(|i| (0..r.bbox.3 - r.bbox.2).map(move |j| (i, j))).filter(|&(i, j)| labels[(r.bbox.0 + i, r.bbox.2 + j)] == r.label).map(|(i, j)| dt[(r.bbox.0 + i, r.bbox.2 + j)]).fold(0.0, f64::max)),
        ]);
    }
    write(CHAPTER, "region_table.md", &md_table(&["label", "area (pixels)", "centroid (row, column)", "bounding box", "largest inscribed radius"], &rows));
}

// ------------------------------------------------------------------ rank, derivative and geometric

/// A smooth greyscale scene: a gradient, a bright disc, a dark square and fine stripes.
fn grey_scene() -> Array2<f64> {
    let n = 96;
    Array2::from_shape_fn((n, n), |(r, c)| {
        let (y, x) = (r as f64, c as f64);
        let mut v = 0.25 + 0.003 * x;
        if (y - 30.0).powi(2) + (x - 30.0).powi(2) < 15.0f64.powi(2) {
            v = 0.85;
        }
        if (55.0..80.0).contains(&y) && (50.0..80.0).contains(&x) {
            v = 0.1;
        }
        if (15.0..40.0).contains(&y) && (60.0..88.0).contains(&x) {
            v = 0.5 + 0.25 * (x * 0.9).sin();
        }
        v
    })
}

fn psnr(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let mse = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64;
    10.0 * (1.0 / mse).log10()
}

fn rank_geometric_figures() {
    let clean = grey_scene();
    let mut u = uniform(9);
    let impulsive = clean.mapv(|v| {
        let x = u();
        if x < 0.05 { 0.0 } else if x < 0.10 { 1.0 } else { v }
    });
    let mut u = uniform(13);
    let gaussian = clean.mapv(|v| {
        let s: f64 = (0..4).map(|_| u() - 0.5).sum::<f64>() * 3f64.sqrt();
        v + 0.08 * s
    });
    let m = Mode::Reflect;
    let g = Colours::Gray(0.0, 1.0);
    let med = median_filter(&impulsive, (3, 3), m).unwrap();
    let gau = gaussian_filter(&impulsive, (1.0, 1.0), (0, 0), m, 4.0).unwrap();
    write(
        CHAPTER,
        "median.svg",
        &panels(
            "Salt-and-pepper noise: Gaussian smoothing against the median",
            &[Panel::new("10% impulse noise", impulsive.clone(), g), Panel::new("Gaussian, σ = 1", gau, g), Panel::new("median, 3 × 3", med, g)],
            3,
            200.0,
        ),
    );
    let mut rows = Vec::new();
    for (name, noisy) in [("impulse noise (10%)", &impulsive), ("Gaussian noise (σ = 0.08)", &gaussian)] {
        let mut row = vec![name.to_string(), format!("{:.1}", psnr(noisy, &clean))];
        for s in [0.8, 1.5] {
            row.push(format!("{:.1}", psnr(&gaussian_filter(noisy, (s, s), (0, 0), m, 4.0).unwrap(), &clean)));
        }
        for k in [3, 5] {
            row.push(format!("{:.1}", psnr(&median_filter(noisy, (k, k), m).unwrap(), &clean)));
        }
        rows.push(row);
    }
    write(CHAPTER, "median_psnr.md", &md_table(&["noise", "noisy", "Gaussian σ = 0.8", "Gaussian σ = 1.5", "median 3 × 3", "median 5 × 5"], &rows));

    let gm1 = gaussian_gradient_magnitude(&gaussian, 1.0, m).unwrap();
    let gm3 = gaussian_gradient_magnitude(&gaussian, 3.0, m).unwrap();
    let log3 = gaussian_laplace(&gaussian, 3.0, m).unwrap();
    let top = |a: &Array2<f64>| a.iter().cloned().fold(0.0, f64::max);
    let lmax = log3.iter().map(|v| v.abs()).fold(0.0, f64::max);
    write(
        CHAPTER,
        "derivatives.svg",
        &panels(
            "Gaussian derivatives of a noisy image",
            &[
                Panel::new("noisy image (σ = 0.08)", gaussian.clone(), g),
                Panel::new("gradient magnitude, σ = 1", gm1.clone(), Colours::Gray(0.0, top(&gm1))),
                Panel::new("gradient magnitude, σ = 3", gm3.clone(), Colours::Gray(0.0, top(&gm3))),
                Panel::new("Laplacian of Gaussian, σ = 3", log3, Colours::Gray(-lmax, lmax)),
            ],
            4,
            170.0,
        ),
    );

    // Interpolation orders on a step and a slope.
    let samples: Vec<f64> = (0..16).map(|i| if i < 6 { 0.0 } else if i < 10 { 1.0 } else { 1.0 - 0.15 * (i - 9) as f64 }).collect();
    let row = Array2::from_shape_vec((1, 16), samples.clone()).unwrap();
    let xs: Vec<f64> = (0..=600).map(|k| k as f64 * 15.0 / 600.0).collect();
    let pts: Vec<(f64, f64)> = xs.iter().map(|&x| (0.0, x)).collect();
    let mut plot = Plot::new("Interpolating samples of a step and a slope", "position (pixels)", "value");
    plot.series.push(Series::new("samples", samples.iter().enumerate().map(|(i, &v)| (i as f64, v)).collect(), Style::Markers, 6));
    for (k, order) in [0usize, 1, 3, 5].into_iter().enumerate() {
        let v = map_coordinates(&row, &pts, order, Mode::Nearest).unwrap();
        plot.series.push(Series::new(&format!("order {order}"), xs.iter().zip(v).map(|(&x, y)| (x, y)).collect(), Style::Line, k));
    }
    plot.legend = Legend::TopRight;
    plot.y_range = Some((-0.3, 1.4));
    write(CHAPTER, "interpolation_orders.svg", &plot.render());

    // Rotating forth and back: what each order loses.
    let sharp = clean.clone();
    let smooth = gaussian_filter(&clean, (2.0, 2.0), (0, 0), m, 4.0).unwrap();
    let mut rows = Vec::new();
    for order in [0usize, 1, 2, 3, 5] {
        let mut row = vec![order.to_string()];
        for img in [&smooth, &sharp] {
            let mut x = img.clone();
            for _ in 0..6 {
                x = rotate(&x, 15.0, false, order, Mode::Nearest).unwrap();
            }
            for _ in 0..6 {
                x = rotate(&x, -15.0, false, order, Mode::Nearest).unwrap();
            }
            // Compare away from the corners, which rotation moves outside the frame.
            let (n, c) = (img.dim().0, img.dim().0 as f64 / 2.0);
            let mut se = 0.0;
            let mut count = 0.0;
            for i in 0..n {
                for j in 0..n {
                    if (i as f64 - c).powi(2) + (j as f64 - c).powi(2) < (0.35 * n as f64).powi(2) {
                        se += (x[(i, j)] - img[(i, j)]).powi(2);
                        count += 1.0;
                    }
                }
            }
            row.push(format!("{:.4}", (se / count).sqrt()));
        }
        rows.push(row);
    }
    write(CHAPTER, "rotation_roundtrip.md", &md_table(&["spline order", "smooth image: RMS error", "sharp-edged image: RMS error"], &rows));

    let rot = rotate(&clean, 30.0, true, 3, Mode::Constant(0.0)).unwrap();
    let crop = clean.slice(ndarray::s![18..42, 18..42]).to_owned();
    let z0 = zoom(&crop, (4.0, 4.0), 0, Mode::Nearest).unwrap();
    let z3 = zoom(&crop, (4.0, 4.0), 3, Mode::Nearest).unwrap();
    write(
        CHAPTER,
        "geometric.svg",
        &panels(
            "Rotation and magnification",
            &[
                Panel::new("rotated 30°, enlarged frame, cubic", rot, g),
                Panel::new("detail ×4, order 0 (nearest)", z0, g),
                Panel::new("detail ×4, order 3 (cubic)", z3, g),
            ],
            3,
            200.0,
        ),
    );
}

// ------------------------------------------------------------------ Fourier-domain filters

fn fourier_figures() {
    let scene = grey_scene();
    let g = Colours::Gray(0.0, 1.0);
    let gauss = fourier_filter(&scene, |s| fourier_gaussian(s, (3.0, 3.0), Spectrum::Full)).unwrap();
    let boxed = fourier_filter(&scene, |s| fourier_uniform(s, (7.0, 7.0), Spectrum::Full)).unwrap();
    let disc = fourier_filter(&scene, |s| fourier_ellipsoid(s, (9.0, 9.0), Spectrum::Full)).unwrap();
    write(
        CHAPTER,
        "fourier_filters.svg",
        &panels(
            "Smoothing in the Fourier domain (the image wraps around at its edges)",
            &[
                Panel::new("test image", scene.clone(), g),
                Panel::new("Gaussian, σ = 3", gauss, g),
                Panel::new("box, 7 × 7", boxed, g),
                Panel::new("disc, diameter 9", disc, g),
            ],
            4,
            170.0,
        ),
    );

    // Fourier against spatial Gaussian smoothing: agreement and cost.
    let n = 256;
    let mut u = uniform(21);
    let big = Array2::from_shape_fn((n, n), |_| u());
    let time_ms = |f: &dyn Fn()| {
        f();
        (0..5)
            .map(|_| {
                let t = std::time::Instant::now();
                f();
                t.elapsed().as_secs_f64() * 1e3
            })
            .fold(f64::INFINITY, f64::min)
    };
    let mut rows = Vec::new();
    for sigma in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0] {
        let spatial = gaussian_filter(&big, (sigma, sigma), (0, 0), Mode::Wrap, 4.0).unwrap();
        let fourier = fourier_filter(&big, |s| fourier_gaussian(s, (sigma, sigma), Spectrum::Full)).unwrap();
        let diff = spatial.iter().zip(&fourier).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let ts = time_ms(&|| {
            std::hint::black_box(gaussian_filter(&big, (sigma, sigma), (0, 0), Mode::Wrap, 4.0).unwrap());
        });
        let tf = time_ms(&|| {
            std::hint::black_box(fourier_filter(&big, |s| fourier_gaussian(s, (sigma, sigma), Spectrum::Full)).unwrap());
        });
        rows.push(vec![format!("{sigma}"), format!("{:.1e}", diff), format!("{ts:.1} ms"), format!("{tf:.1} ms")]);
    }
    write(CHAPTER, "fourier_vs_spatial.md", &md_table(&["σ (256 × 256 image)", "largest difference, spatial against Fourier", "spatial (separable, wrap)", "Fourier (FFT, filter, inverse FFT)"], &rows));

    // Sub-pixel shifts of a smooth bump and a step.
    let m = 64;
    let bump = |x: f64| (-0.5 * ((x - 20.0) / 3.0).powi(2)).exp();
    let step = |x: f64| if (36.0..52.0).contains(&x) { 1.0 } else { 0.0 };
    let row = Array2::from_shape_fn((1, m), |(_, j)| bump(j as f64) + step(j as f64));
    let shifted_f = fourier_filter(&row, |s| fourier_shift(s, (0.0, 0.5), Spectrum::Full)).unwrap();
    let shifted_s = shift(&row, (0.0, 0.5), 3, Mode::Wrap).unwrap();
    let mut plot = Plot::new("Shifting by half a pixel: a smooth bump and a step", "position (pixels)", "value");
    let fine: Vec<f64> = (0..=640).map(|k| k as f64 / 10.0).collect();
    plot.series.push(Series::new("signal moved exactly", fine.iter().map(|&x| (x, bump(x - 0.5) + step(x - 0.5))).collect(), Style::Line, 7));
    plot.series.push(Series::new("phase ramp (fourier_shift)", shifted_f.iter().enumerate().map(|(j, &v)| (j as f64, v)).collect(), Style::LineMarkers, 0));
    plot.series.push(Series::new("cubic spline (shift, order 3)", shifted_s.iter().enumerate().map(|(j, &v)| (j as f64, v)).collect(), Style::LineMarkers, 1));
    plot.legend = Legend::TopLeft;
    plot.y_range = Some((-0.3, 1.4));
    plot.x_range = Some((8.0, 60.0));
    write(CHAPTER, "fourier_shift.svg", &plot.render());
    // Errors with each signal shifted on its own (in one row the step's ringing reaches the bump).
    let alone = |f: &dyn Fn(f64) -> f64| -> (f64, f64) {
        let r = Array2::from_shape_fn((1, m), |(_, j)| f(j as f64));
        let a = fourier_filter(&r, |s| fourier_shift(s, (0.0, 0.5), Spectrum::Full)).unwrap();
        let b = shift(&r, (0.0, 0.5), 3, Mode::Wrap).unwrap();
        let e = |y: &Array2<f64>| (0..m).map(|j| (y[(0, j)] - f(j as f64 - 0.5)).abs()).fold(0.0, f64::max);
        (e(&a), e(&b))
    };
    let (bf, bs) = alone(&bump);
    let (sf, ss) = alone(&step);
    let rows = vec![
        vec!["smooth bump (σ = 3 pixels)".to_string(), format!("{bf:.1e}"), format!("{bs:.1e}")],
        vec!["step".to_string(), format!("{sf:.1e}"), format!("{ss:.1e}")],
    ];
    write(CHAPTER, "fourier_shift_errors.md", &md_table(&["signal (shift by 0.5 pixel)", "phase ramp: largest error", "cubic spline: largest error"], &rows));
}
