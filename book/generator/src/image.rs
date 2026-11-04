//! Figures and tables for the DSP chapter on image processing.

use crate::raster::{Colours, Mark, Panel, panels};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_dsp::discrete::image::convolution::{Border, convolve2d, correlate2d};
use lib_jsl_dsp::discrete::image::feature_extraction::blob_detection::blob_detection;
use lib_jsl_dsp::discrete::image::feature_extraction::hough_transform::{hough_circles, hough_line_peaks, hough_transform};
use lib_jsl_dsp::discrete::image::feature_extraction::match_descriptors;
use lib_jsl_dsp::discrete::image::feature_extraction::sift::{SiftParams, sift};
use lib_jsl_dsp::discrete::image::feature_extraction::surf::{SurfParams, surf};
use lib_jsl_dsp::discrete::image::filters::gaussian_blur::{gaussian_blur, gaussian_blur_with_border};
use lib_jsl_dsp::discrete::image::filters::histogram_equalization::{histogram, histogram_equalization};
use lib_jsl_dsp::discrete::image::filters::sharpening::sharpening;
use lib_jsl_dsp::discrete::image::filters::sobel::sobel;
use lib_jsl_dsp::discrete::image::filters::wiener_filter::{wiener_deconvolution, wiener_filter};
use lib_jsl_dsp::discrete::image::segmentation::canny_edge_detection::canny_edge_detection;
use lib_jsl_dsp::discrete::image::segmentation::otsu::otsu;
use lib_jsl_dsp::discrete::image::segmentation::watershed::{label_boundaries, watershed};
use lib_jsl_random::uniform_generator::UniformGenerator;
use lib_jsl_random::xoshiro256plusplus::Xoshiro256PlusPlus;
use ndarray::Array2;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::time::Instant;

const CHAPTER: &str = "image";
const N: usize = 96;

pub fn generate() {
    let refs = References::load();
    reference_table(&refs);
    filtering_figures(&refs);
    restoration_figures(&refs);
    segmentation_figures(&refs);
    hough_figures(&refs);
    feature_figures();
}

// ------------------------------------------------------------------ reference data

struct References {
    values: HashMap<String, Vec<Vec<f64>>>,
    digits: HashMap<String, Vec<u8>>,
}

impl References {
    fn load() -> Self {
        let text = include_str!("../data/image_reference.txt");
        let (mut values, mut digits) = (HashMap::new(), HashMap::new());
        for line in text.lines() {
            let (name, rest) = line.split_once('|').unwrap();
            if !rest.contains(',') && rest.len() > 100 {
                digits.insert(name.to_string(), rest.bytes().map(|b| b - b'0').collect());
            } else {
                values.insert(name.to_string(), rest.split(';').map(|l| l.split(',').map(|v| v.parse().unwrap()).collect()).collect());
            }
        }
        References { values, digits }
    }
    fn image(&self, name: &str) -> Array2<f64> {
        let v = &self.values[name][0];
        Array2::from_shape_vec((v.len() / N, N), v.clone()).unwrap()
    }
    fn mask(&self, name: &str) -> Array2<bool> {
        Array2::from_shape_vec((N, N), self.digits[name].iter().map(|&d| d != 0).collect()).unwrap()
    }
    fn labels(&self, name: &str) -> Array2<i32> {
        Array2::from_shape_vec((N, N), self.digits[name].iter().map(|&d| d as i32).collect()).unwrap()
    }
    /// The 8-bit test scene, and the same scaled to [0, 1].
    fn scene(&self) -> (Array2<u8>, Array2<f64>) {
        let s = self.image("scene");
        (s.mapv(|v| v as u8), s.mapv(|v| v / 255.0))
    }
}

/// The largest absolute difference over the first `rows` rows, optionally skipping `skip` pixels at
/// every edge.
fn max_diff(a: &Array2<f64>, b: &Array2<f64>, skip: usize) -> f64 {
    let rows = b.nrows().min(N - skip);
    let mut m = 0.0f64;
    for r in skip..rows {
        for c in skip..N - skip {
            m = m.max((a[(r, c)] - b[(r, c)]).abs());
        }
    }
    m
}

/// The fraction of the true pixels of `a` that lie within one pixel (8-neighbourhood) of a true
/// pixel of `b`.
fn near_fraction(a: &Array2<bool>, b: &Array2<bool>) -> f64 {
    let (rows, cols) = a.dim();
    let (mut hit, mut total) = (0, 0);
    for ((r, c), &v) in a.indexed_iter() {
        if !v {
            continue;
        }
        total += 1;
        let near = (r.saturating_sub(1)..=(r + 1).min(rows - 1)).any(|rr| (c.saturating_sub(1)..=(c + 1).min(cols - 1)).any(|cc| b[(rr, cc)]));
        hit += near as usize;
    }
    hit as f64 / total.max(1) as f64
}

fn blob_image() -> Array2<f64> {
    Array2::from_shape_fn((N, N), |(r, c)| {
        [(20.0, 20.0, 2.0, 1.0), (30.0, 65.0, 4.0, 1.0), (70.0, 30.0, 6.0, 1.0), (72.0, 75.0, 3.0, 0.7)]
            .iter()
            .map(|&(br, bc, s, a): &(f64, f64, f64, f64)| a * (-((r as f64 - br).powi(2) + (c as f64 - bc).powi(2)) / (2.0 * s * s)).exp())
            .sum()
    })
}

fn reference_table(refs: &References) {
    let (scene8, f) = refs.scene();
    let kernel = Array2::from_shape_vec((3, 5), refs.values["kernel"][0].clone()).unwrap();
    let mut rows = Vec::new();
    let mut row = |what: &str, measure: &str, result: String| rows.push(vec![what.to_string(), measure.to_string(), result]);
    for (mode, border) in [("constant", Border::Zero), ("nearest", Border::Replicate), ("reflect", Border::Reflect)] {
        let a = correlate2d(&f, &kernel, border).unwrap();
        let b = convolve2d(&f, &kernel, border).unwrap();
        row(&format!("`ndimage.correlate`, mode {mode}"), "largest difference", format!("{:.1e}", max_diff(&a, &refs.image(&format!("correlate {mode}")), 0)));
        row(&format!("`ndimage.convolve`, mode {mode}"), "largest difference", format!("{:.1e}", max_diff(&b, &refs.image(&format!("convolve {mode}")), 0)));
    }
    for sigma in [1.5, 2.3] {
        row(&format!("`ndimage.gaussian_filter`, σ = {sigma}, mode nearest"), "largest difference", format!("{:.1e}", max_diff(&gaussian_blur(&f, sigma).unwrap(), &refs.image(&format!("gaussian nearest {sigma}")), 0)));
    }
    row("`ndimage.gaussian_filter`, σ = 1.5, mode reflect", "largest difference", format!("{:.1e}", max_diff(&gaussian_blur_with_border(&f, 1.5, Border::Reflect).unwrap(), &refs.image("gaussian reflect 1.5"), 0)));
    let s = sobel(&f, false).unwrap();
    row("`ndimage.sobel`, along columns", "largest difference", format!("{:.1e}", max_diff(&s.gx, &refs.image("sobel x"), 0)));
    row("`ndimage.sobel`, along rows", "largest difference", format!("{:.1e}", max_diff(&s.gy, &refs.image("sobel y"), 0)));
    let w = wiener_filter(&f, 5, Some((12.0f64 / 255.0).powi(2))).unwrap();
    row("`signal.wiener`, 5 × 5, given noise", "largest difference away from the edges (SciPy pads with zeros)", format!("{:.1e}", max_diff(&w, &refs.image("wiener 5"), 2)));
    let o = otsu(&scene8).unwrap();
    row("`filters.threshold_otsu`", "threshold", format!("{} (scikit-image {})", o.threshold, refs.values["otsu"][0][0]));
    for (name, sigma, low, high) in [("canny 2.0 0.05 0.15", 2.0, 0.05, 0.15), ("canny 1.0 0.1 0.3", 1.0, 0.1, 0.3)] {
        // scikit-image thresholds the raw Sobel magnitude, 8 times the gradient.
        let mine = canny_edge_detection(&f, sigma, low / 8.0, high / 8.0).unwrap();
        let theirs = refs.mask(name);
        row(
            &format!("`feature.canny`, σ = {sigma}"),
            "edge pixels; fraction of each within one pixel of the other's",
            format!("{} and {}; {:.0}% and {:.0}%", mine.iter().filter(|&&v| v).count(), theirs.iter().filter(|&&v| v).count(), 100.0 * near_fraction(&mine, &theirs), 100.0 * near_fraction(&theirs, &mine)),
        );
    }
    let elevation = refs.image("watershed elevation");
    let mut markers = Array2::<i32>::zeros((N, N));
    for m in refs.values["watershed markers"][0].chunks(3) {
        markers[(m[0] as usize, m[1] as usize)] = m[2] as i32;
    }
    let labels = watershed(&elevation, &markers).unwrap();
    let theirs = refs.labels("watershed");
    let same = labels.iter().zip(theirs.iter()).filter(|(a, b)| a == b).count();
    row("`segmentation.watershed`, 3 markers", "pixels labelled the same", format!("{same} of {} ({:.2}%)", N * N, 100.0 * same as f64 / (N * N) as f64));
    let h = hough_transform(&refs.mask("hough edges"), 180, 1.0).unwrap();
    let peaks = hough_line_peaks(&h, 10, 2, 4, 8);
    let want = &refs.values["hough peaks"];
    let mine: Vec<String> = peaks.iter().map(|l| format!("({:.0}°, {:.1})", l.theta.to_degrees(), l.rho)).collect();
    let them: Vec<String> = (0..want[0].len()).map(|k| format!("({:.0}°, {:.1})", want[1][k].to_degrees(), want[2][k])).collect();
    row("`transform.hough_line` and `hough_line_peaks`", "the two strongest lines (θ, ρ)", format!("{}; scikit-image {}", mine.join(", "), them.join(", ")));
    let circles = hough_circles(&refs.mask("circle edges"), &[12, 20], 0.6, 5).unwrap();
    let want = &refs.values["circle peaks"];
    let mine: Vec<String> = circles.iter().take(2).map(|c| format!("({}, {}, r {})", c.row, c.col, c.radius)).collect();
    let them: Vec<String> = (0..want[0].len()).map(|k| format!("({}, {}, r {})", want[0][k], want[1][k], want[2][k])).collect();
    row("`transform.hough_circle` and `hough_circle_peaks`", "centres (row, column) and radii", format!("{}; scikit-image {}", mine.join(", "), them.join(", ")));
    let mut blobs = blob_detection(&blob_image(), 1.0, 10.0, 10, 0.1, 0.5).unwrap();
    blobs.sort_by_key(|b| (b.row, b.col));
    let want = &refs.values["blobs"];
    let mine: Vec<String> = blobs.iter().map(|b| format!("({}, {}, σ {:.2})", b.row, b.col, b.sigma)).collect();
    let them: Vec<String> = (0..want[0].len()).map(|k| format!("({}, {}, σ {:.2})", want[0][k], want[1][k], want[2][k])).collect();
    row("`feature.blob_log`, σ from 1 to 10 in 10 log steps", "blobs (row, column, σ)", format!("{}; scikit-image {}", mine.join(", "), them.join(", ")));
    write(CHAPTER, "reference.md", &md_table(&["SciPy or scikit-image function and case", "compared", "result"], &rows));
}

// ------------------------------------------------------------------ filtering

fn filtering_figures(refs: &References) {
    let (_, f) = refs.scene();
    let blurred = gaussian_blur(&f, 1.5).unwrap();
    let grad = sobel(&blurred, true).unwrap();
    let sharp = sharpening(&blurred, 2.0, 1.5).unwrap();
    let max_g = grad.magnitude.iter().cloned().fold(0.0, f64::max);
    let figure = panels(
        "A noisy test scene, blurred, differentiated and re-sharpened",
        &[
            Panel::new("scene (noise σ = 12 of 255)", f.clone(), Colours::Gray(0.0, 1.0)),
            Panel::new("Gaussian blur, σ = 1.5", blurred.clone(), Colours::Gray(0.0, 1.0)),
            Panel::new("Sobel magnitude of the blur", grad.magnitude.clone(), Colours::Gray(0.0, max_g)),
            Panel::new("the blur, unsharp-masked", sharp, Colours::Gray(0.0, 1.0)),
        ],
        2,
        240.0,
    );
    write(CHAPTER, "filtering.svg", &figure);
    // A profile across the disc's edge: the blur spreads it, the unsharp mask overshoots.
    let mut plot = Plot::new("A profile along row 30, through the disc", "column", "intensity");
    let soft = gaussian_blur(&Array2::from_shape_fn((N, N), |(r, c)| if ((r as f64 - 30.0).powi(2) + (c as f64 - 34.0).powi(2)) < 196.0 { 200.0 / 255.0 } else { 50.0 / 255.0 }), 1.5).unwrap();
    let profile = |a: &Array2<f64>| (0..N).map(|c| (c as f64, a[(30, c)])).collect::<Vec<_>>();
    plot.series.push(Series::new("clean edge, blurred σ = 1.5", profile(&soft), Style::Line, 7));
    plot.series.push(Series::new("unsharp mask, amount 1", profile(&sharpening(&soft, 2.0, 1.0).unwrap()), Style::Line, 0));
    plot.series.push(Series::new("unsharp mask, amount 2", profile(&sharpening(&soft, 2.0, 2.0).unwrap()), Style::Line, 1));
    plot.x_range = Some((5.0, 63.0));
    plot.y_range = Some((0.0, 1.25));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "unsharp_profile.svg", &plot.render());
    // Gaussian blurs compose; the separable form is much faster.
    let a = gaussian_blur(&gaussian_blur(&f, 1.2).unwrap(), 1.6).unwrap();
    let b = gaussian_blur(&f, 2.0).unwrap();
    let composed = max_diff(&a, &b, 12);
    let sigma = 3.0;
    let k1 = lib_jsl_dsp::discrete::image::convolution::gaussian_kernel_1d(sigma, 4.0).unwrap();
    let k2 = Array2::from_shape_fn((k1.len(), k1.len()), |(i, j)| k1[i] * k1[j]);
    let big: Array2<f64> = Array2::from_shape_fn((256, 256), |(r, c)| ((r * 7 + c * 13) % 17) as f64);
    let t0 = Instant::now();
    let sep = gaussian_blur(&big, sigma).unwrap();
    let t_sep = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let full = correlate2d(&big, &k2, Border::Replicate).unwrap();
    let t_full = t0.elapsed().as_secs_f64();
    let diff = sep.iter().zip(full.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    write(
        CHAPTER,
        "gaussian_facts.md",
        &format!(
            "Blurring by σ = 1.2 and then 1.6 differs from one blur of σ = 2 by at most {composed:.1e} away from the borders. On a 256 × 256 image with σ = 3 (a {}-tap kernel), the separable blur took {:.1} ms and the full {} × {} kernel {:.1} ms, a factor of {:.0}, with results equal to {diff:.0e}.",
            k1.len(),
            1e3 * t_sep,
            k1.len(),
            k1.len(),
            1e3 * t_full,
            t_full / t_sep
        ),
    );
}

// ------------------------------------------------------------------ restoration

fn mse(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64
}

fn psnr(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    -10.0 * mse(a, b).log10()
}

fn clean_scene() -> Array2<f64> {
    Array2::from_shape_fn((N, N), |(r, c)| {
        let (r, c) = (r as f64, c as f64);
        if (r - 30.0).powi(2) + (c - 34.0).powi(2) < 196.0 {
            200.0 / 255.0
        } else if (55.0..85.0).contains(&r) && (50.0..85.0).contains(&c) {
            130.0 / 255.0
        } else {
            50.0 / 255.0 + 0.08 * (c / 9.0).sin() * (r / 13.0).cos()
        }
    })
}

fn restoration_figures(refs: &References) {
    let (scene8, _) = refs.scene();
    // A dull, low-contrast copy of the scene: levels squeezed into 90..150.
    let dull = scene8.mapv(|v| (90.0 + 60.0 * v as f64 / 255.0).round() as u8);
    let eq = histogram_equalization(&dull).unwrap();
    let to_f = |a: &Array2<u8>| a.mapv(|v| v as f64 / 255.0);
    write(
        CHAPTER,
        "equalisation.svg",
        &panels("Histogram equalisation", &[Panel::new("low contrast (levels 90 to 150)", to_f(&dull), Colours::Gray(0.0, 1.0)), Panel::new("equalised", to_f(&eq), Colours::Gray(0.0, 1.0))], 2, 220.0),
    );
    let cdf = |a: &Array2<u8>| {
        let h = histogram(a);
        let mut run = 0usize;
        (0..256).map(|v| { run += h[v]; (v as f64, run as f64 / a.len() as f64) }).collect::<Vec<_>>()
    };
    let mut plot = Plot::new("Cumulative histograms before and after equalisation", "gray level", "fraction of pixels at or below");
    plot.series.push(Series::new("low contrast", cdf(&dull), Style::Line, 1));
    plot.series.push(Series::new("equalised", cdf(&eq), Style::Line, 0));
    plot.series.push(Series::new("uniform", vec![(0.0, 0.0), (255.0, 1.0)], Style::Dashed, 7));
    plot.legend = Legend::BottomRight;
    write(CHAPTER, "equalisation_cdf.svg", &plot.render());
    // Denoising.
    let clean = clean_scene();
    let mut rng = Rng::new(3);
    let sigma_n = 0.08;
    let noisy = clean.mapv(|v| v + sigma_n * rng.gauss());
    let mut rows = vec![vec!["noisy".to_string(), format!("{:.1}", psnr(&noisy, &clean))]];
    for s in [0.7, 1.0, 1.5] {
        rows.push(vec![format!("Gaussian blur, σ = {s}"), format!("{:.1}", psnr(&gaussian_blur(&noisy, s).unwrap(), &clean))]);
    }
    for w in [3, 5, 7] {
        rows.push(vec![format!("adaptive Wiener, {w} × {w}, noise variance given"), format!("{:.1}", psnr(&wiener_filter(&noisy, w, Some(sigma_n * sigma_n)).unwrap(), &clean))]);
    }
    rows.push(vec!["adaptive Wiener, 5 × 5, noise variance estimated".to_string(), format!("{:.1}", psnr(&wiener_filter(&noisy, 5, None).unwrap(), &clean))]);
    write(CHAPTER, "denoise_table.md", &md_table(&["method (noise σ = 0.08)", "PSNR (dB)"], &rows));
    let wf = wiener_filter(&noisy, 5, Some(sigma_n * sigma_n)).unwrap();
    let gb = gaussian_blur(&noisy, 1.5).unwrap();
    write(
        CHAPTER,
        "denoise.svg",
        &panels(
            "Denoising: blur everything, or only where the image is flat",
            &[
                Panel::new("noisy", noisy, Colours::Gray(0.0, 1.0)),
                Panel::new("Gaussian blur, σ = 1.5", gb, Colours::Gray(0.0, 1.0)),
                Panel::new("adaptive Wiener, 5 × 5", wf, Colours::Gray(0.0, 1.0)),
            ],
            3,
            200.0,
        ),
    );
    // Deconvolution of a known Gaussian blur.
    let psf_k = lib_jsl_dsp::discrete::image::convolution::gaussian_kernel_1d(2.0, 4.0).unwrap();
    let psf = Array2::from_shape_fn((psf_k.len(), psf_k.len()), |(i, j)| psf_k[i] * psf_k[j]);
    // Blur circularly over a power-of-two image so the model is exact.
    let big = Array2::from_shape_fn((128, 128), |(r, c)| clean[(r.min(N - 1), c.min(N - 1))]);
    let blurred_clean = circular_blur(&big, &psf);
    let noise_sd = 0.002;
    let blurred = blurred_clean.mapv(|v| v + noise_sd * rng.gauss());
    let mut rows = vec![vec!["blurred and noisy (no restoration)".to_string(), format!("{:.1}", psnr(&blurred, &big))]];
    let mut best = (f64::NEG_INFINITY, 0.0);
    for k in [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2] {
        let r = wiener_deconvolution(&blurred, &psf, k).unwrap();
        let p = psnr(&r, &big);
        if p > best.0 {
            best = (p, k);
        }
        rows.push(vec![if k == 0.0 { "K = 0 (inverse filter)".to_string() } else { format!("K = {k:.0e}") }, format!("{p:.1}")]);
    }
    write(CHAPTER, "deconvolution_table.md", &md_table(&["Wiener deconvolution (Gaussian blur σ = 2, noise σ = 0.002)", "PSNR (dB)"], &rows));
    let crop = |a: &Array2<f64>| Array2::from_shape_fn((N, N), |i| a[i]);
    write(
        CHAPTER,
        "deconvolution.svg",
        &panels(
            "Wiener deconvolution of a known blur",
            &[
                Panel::new("blurred (σ = 2) and noisy", crop(&blurred), Colours::Gray(0.0, 1.0)),
                Panel::new("K = 0: inverse filtering", crop(&wiener_deconvolution(&blurred, &psf, 0.0).unwrap()), Colours::Gray(0.0, 1.0)),
                Panel::new(&format!("K = {:.0e}", best.1), crop(&wiener_deconvolution(&blurred, &psf, best.1).unwrap()), Colours::Gray(0.0, 1.0)),
            ],
            3,
            200.0,
        ),
    );
}

/// Circular convolution with a PSF centred on its middle element.
fn circular_blur(image: &Array2<f64>, psf: &Array2<f64>) -> Array2<f64> {
    let (rows, cols) = image.dim();
    let (pr, pc) = psf.dim();
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        let mut acc = 0.0;
        for ((i, j), &w) in psf.indexed_iter() {
            let rr = (r + rows + pr / 2 - i) % rows;
            let cc = (c + cols + pc / 2 - j) % cols;
            acc += w * image[(rr, cc)];
        }
        acc
    })
}

// ------------------------------------------------------------------ segmentation

fn segmentation_figures(refs: &References) {
    let (scene8, f) = refs.scene();
    let o = otsu(&scene8).unwrap();
    let h = histogram(&scene8);
    let mut plot = Plot::new(&format!("Histogram of the scene and Otsu's threshold ({})", o.threshold), "gray level", "pixels");
    plot.series.push(Series::new("histogram", (0..256).map(|v| (v as f64, h[v] as f64)).collect(), Style::Line, 0));
    let top = *h.iter().max().unwrap() as f64;
    plot.series.push(Series::new("threshold", vec![(o.threshold as f64, 0.0), (o.threshold as f64, top)], Style::Dashed, 1));
    // The between-class variance, scaled to the plot.
    let sb = between_class(&h);
    let sb_max = sb.iter().cloned().fold(0.0, f64::max);
    plot.series.push(Series::new("between-class variance (scaled)", sb.iter().enumerate().map(|(t, v)| (t as f64, v / sb_max * top)).collect(), Style::Line, 2));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "otsu_histogram.svg", &plot.render());
    write(CHAPTER, "otsu_facts.md", &format!("Otsu's threshold is {}, with separability $\\sigma_b^2/\\sigma^2 = {:.2}$.", o.threshold, o.separability));
    let edges1 = canny_edge_detection(&f, 1.0, 0.02, 0.06).unwrap();
    let edges2 = canny_edge_detection(&f, 2.0, 0.01, 0.03).unwrap();
    let raw = sobel(&f, true).unwrap();
    let raw_edges = raw.magnitude.mapv(|v| v > 0.06);
    let elevation = refs.image("watershed elevation");
    let mut markers = Array2::<i32>::zeros((N, N));
    let mut marks = Vec::new();
    for m in refs.values["watershed markers"][0].chunks(3) {
        markers[(m[0] as usize, m[1] as usize)] = m[2] as i32;
        marks.push(Mark::Dot { x: m[1], y: m[0], colour: "#ffffff" });
    }
    let labels = watershed(&elevation, &markers).unwrap();
    let mut seg_panel = Panel::new("watershed from 3 markers", labels.mapv(|v| v as f64), Colours::Labels);
    seg_panel.marks.push(Mark::Mask(label_boundaries(&labels), "#ffffff"));
    seg_panel.marks.extend(marks);
    let mut otsu_panel = Panel::new(&format!("Otsu threshold ({})", o.threshold), o.binary.mapv(|v| v as f64 / 255.0), Colours::Gray(0.0, 1.0));
    otsu_panel.marks.clear();
    let mut c1 = Panel::new("Canny, σ = 1", f.clone(), Colours::Gray(0.0, 1.0));
    c1.marks.push(Mark::Mask(edges1, "#ff3030"));
    let mut c2 = Panel::new("Canny, σ = 2", f.clone(), Colours::Gray(0.0, 1.0));
    c2.marks.push(Mark::Mask(edges2, "#ff3030"));
    let mut thr = Panel::new("Sobel magnitude > 0.06", f.clone(), Colours::Gray(0.0, 1.0));
    thr.marks.push(Mark::Mask(raw_edges, "#ff3030"));
    write(CHAPTER, "segmentation.svg", &panels("Segmenting the test scene", &[otsu_panel, thr, c1, c2, Panel::new("elevation: smoothed gradient", elevation.clone(), Colours::Gray(0.0, elevation.iter().cloned().fold(0.0, f64::max))), seg_panel], 3, 200.0));
    // Otsu on noisier and less separated images.
    let mut rows = Vec::new();
    let truth = Array2::from_shape_fn((N, N), |(r, c)| ((r as f64 - 30.0).powi(2) + (c as f64 - 34.0).powi(2)) < 196.0 || ((55..85).contains(&r) && (50..85).contains(&c)));
    let mut rng = Rng::new(11);
    for (gap, noise) in [(80.0, 12.0), (80.0, 30.0), (40.0, 12.0), (40.0, 30.0)] {
        let img = Array2::from_shape_fn((N, N), |i| ((if truth[i] { 100.0 + gap } else { 100.0 }) + noise * rng.gauss()).round().clamp(0.0, 255.0) as u8);
        let o = otsu(&img).unwrap();
        let wrong = o.binary.iter().zip(truth.iter()).filter(|(b, t)| (**b > 0) != **t).count();
        rows.push(vec![format!("{gap}"), format!("{noise}"), format!("{}", o.threshold), format!("{:.2}", o.separability), format!("{:.1}%", 100.0 * wrong as f64 / (N * N) as f64)]);
    }
    write(CHAPTER, "otsu_table.md", &md_table(&["gap between the means", "noise σ", "threshold", "separability", "pixels misclassified"], &rows));
}

fn between_class(h: &[usize; 256]) -> Vec<f64> {
    let total: f64 = h.iter().sum::<usize>() as f64;
    let sum_all: f64 = h.iter().enumerate().map(|(i, &v)| i as f64 * v as f64).sum();
    let (mut w0, mut s0) = (0.0, 0.0);
    (0..256)
        .map(|t| {
            w0 += h[t] as f64;
            s0 += t as f64 * h[t] as f64;
            let w1 = total - w0;
            if w0 == 0.0 || w1 == 0.0 { 0.0 } else { (w0 / total) * (w1 / total) * (s0 / w0 - (sum_all - s0) / w1).powi(2) }
        })
        .collect()
}

// ------------------------------------------------------------------ Hough

fn hough_figures(refs: &References) {
    // Edges of a few lines and circles, with gaps and clutter.
    let mut rng = Rng::new(5);
    let mut edges = Array2::from_elem((N, N), false);
    let put = |r: f64, c: f64, e: &mut Array2<bool>| {
        let (r, c) = (r.round(), c.round());
        if r >= 0.0 && c >= 0.0 && (r as usize) < N && (c as usize) < N {
            e[(r as usize, c as usize)] = true;
        }
    };
    for k in 0..200 {
        let t = k as f64 / 199.0;
        if (t * 7.0).fract() < 0.75 {
            put(8.0 + 70.0 * t, 5.0 + 60.0 * t, &mut edges);
        }
        if (t * 5.0).fract() < 0.8 {
            put(80.0 - 10.0 * t, 10.0 + 80.0 * t, &mut edges);
        }
    }
    for k in 0..360 {
        let a = k as f64 * PI / 180.0;
        if (k / 30) % 3 != 2 {
            put(32.0 + 15.0 * a.sin(), 66.0 + 15.0 * a.cos(), &mut edges);
        }
    }
    for _ in 0..150 {
        put(rng.uniform() * 95.0, rng.uniform() * 95.0, &mut edges);
    }
    let h = hough_transform(&edges, 180, 1.0).unwrap();
    let lines = hough_line_peaks(&h, 30, 2, 4, 8);
    let circles = hough_circles(&edges, &[10, 15, 20], 0.5, 5).unwrap();
    let mut panel = Panel::new("edges, lines and circles found", edges.mapv(|v| if v { 1.0 } else { 0.0 }), Colours::Gray(0.0, 1.5));
    for l in &lines {
        // Clip the infinite line x cos θ + y sin θ = ρ to the image.
        let (c, s) = (l.theta.cos(), l.theta.sin());
        let pts: Vec<(f64, f64)> = if s.abs() > 0.5 {
            vec![(0.0, l.rho / s), ((N - 1) as f64, (l.rho - (N - 1) as f64 * c) / s)]
        } else {
            vec![(l.rho / c, 0.0), ((l.rho - (N - 1) as f64 * s) / c, (N - 1) as f64)]
        };
        panel.marks.push(Mark::Line { from: pts[0], to: pts[1], colour: "#ff3030" });
    }
    for c in circles.iter().take(1) {
        panel.marks.push(Mark::Circle { x: c.col as f64, y: c.row as f64, radius: c.radius as f64, colour: "#30a0ff" });
    }
    write(CHAPTER, "hough.svg", &panels("The Hough transform finds broken lines and circles among clutter", &[panel], 1, 300.0));
    write(
        CHAPTER,
        "hough_facts.md",
        &format!(
            "The two strongest lines have {} and {} votes, at (θ, ρ) = {}; the clutter's best cell has {}. The best circle is centred at row {}, column {} with radius {} and support {:.2} (the true circle: row 32, column 66, radius 15, one third missing).",
            lines[0].votes,
            lines[1].votes,
            lines.iter().map(|l| format!("({:.0}°, {:.0})", l.theta.to_degrees(), l.rho)).collect::<Vec<_>>().join(" and "),
            clutter_best(&edges),
            circles[0].row,
            circles[0].col,
            circles[0].radius,
            circles[0].support
        ),
    );
    // The accumulator, drawn as an image (square-root scale, to show the faint sinusoids): rows are
    // θ, columns ρ bins (left = most negative).
    let max = (*h.accumulator.iter().max().unwrap() as f64).sqrt();
    let transposed = h.accumulator.t().mapv(|v| (v as f64).sqrt());
    let mut acc = Panel::new(&format!("accumulator (√votes): ρ from {:.0} (left) to {:.0}, θ from 0° (top) to 180°", h.rho_min, h.rho(h.accumulator.nrows() - 1)), transposed, Colours::Ramp(0.0, max));
    for l in &lines {
        acc.marks.push(Mark::Circle { x: (l.rho - h.rho_min) / h.rho_step, y: l.theta.to_degrees(), radius: 6.0, colour: "#ff3030" });
    }
    write(CHAPTER, "hough_accumulator.svg", &panels("Every edge pixel votes along a sinusoid; lines are the peaks", &[acc], 1, 540.0));
    let _ = refs;
}

/// The most votes any line receives from the clutter alone.
fn clutter_best(edges: &Array2<bool>) -> u32 {
    let mut rng = Rng::new(5);
    let mut clutter = Array2::from_elem(edges.dim(), false);
    for _ in 0..150 {
        let (r, c) = ((rng.uniform() * 95.0).round() as usize, (rng.uniform() * 95.0).round() as usize);
        clutter[(r, c)] = true;
    }
    *hough_transform(&clutter, 180, 1.0).unwrap().accumulator.iter().max().unwrap()
}

// ------------------------------------------------------------------ features

/// A synthetic scene with shapes and texture, rich in features.
fn feature_scene(n: usize) -> Array2<f64> {
    let mut rng = Rng::new(21);
    let noise = Array2::from_shape_fn((n, n), |_| rng.uniform());
    let texture = gaussian_blur(&noise, 2.5).unwrap();
    let shapes: Vec<(f64, f64, f64, f64)> = (0..14).map(|_| (rng.uniform() * n as f64, rng.uniform() * n as f64, 4.0 + 12.0 * rng.uniform(), rng.uniform())).collect();
    Array2::from_shape_fn((n, n), |(r, c)| {
        let mut v = 0.4 + 2.0 * (texture[(r, c)] - 0.5);
        for (k, &(sr, sc, size, level)) in shapes.iter().enumerate() {
            let (dr, dc) = (r as f64 - sr, c as f64 - sc);
            let inside = if k % 2 == 0 { dr * dr + dc * dc < size * size } else { dr.abs() < size && dc.abs() < 0.6 * size };
            if inside {
                v = level;
            }
        }
        v
    })
}

/// Samples `image` under the similarity transform mapping output pixel `p` to source
/// `centre + R(−angle) (p − centre) / scale`, bilinearly; outside pixels take `fill`.
fn warp(image: &Array2<f64>, angle: f64, scale: f64, fill: f64) -> Array2<f64> {
    let (rows, cols) = image.dim();
    let (cy, cx) = ((rows as f64 - 1.0) / 2.0, (cols as f64 - 1.0) / 2.0);
    let (ca, sa) = (angle.cos(), angle.sin());
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        let (x, y) = ((c as f64 - cx) / scale, (r as f64 - cy) / scale);
        let (sx, sy) = (ca * x + sa * y + cx, -sa * x + ca * y + cy);
        if sx < 0.0 || sy < 0.0 || sx > (cols - 1) as f64 || sy > (rows - 1) as f64 {
            return fill;
        }
        let (x0, y0) = (sx.floor() as usize, sy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(cols - 1), (y0 + 1).min(rows - 1));
        let (fx, fy) = (sx - x0 as f64, sy - y0 as f64);
        (1.0 - fy) * ((1.0 - fx) * image[(y0, x0)] + fx * image[(y0, x1)]) + fy * ((1.0 - fx) * image[(y1, x0)] + fx * image[(y1, x1)])
    })
}

/// Where the warp sends source point (x, y).
fn forward(x: f64, y: f64, n: usize, angle: f64, scale: f64) -> (f64, f64) {
    let centre = (n as f64 - 1.0) / 2.0;
    let (dx, dy) = (x - centre, y - centre);
    let (ca, sa) = (angle.cos(), angle.sin());
    (centre + scale * (ca * dx - sa * dy), centre + scale * (sa * dx + ca * dy))
}

fn feature_figures() {
    // Blobs.
    let img = blob_image();
    let blobs = blob_detection(&img, 1.0, 10.0, 10, 0.1, 0.5).unwrap();
    let mut panel = Panel::new("Gaussian spots of σ 2, 3, 4 and 6", img.clone(), Colours::Gray(0.0, 1.0));
    for b in &blobs {
        panel.marks.push(Mark::Circle { x: b.col as f64, y: b.row as f64, radius: b.radius(), colour: "#ff3030" });
    }
    let lap_scales: Vec<f64> = (0..40).map(|k| 0.5 + 0.25 * k as f64).collect();
    write(CHAPTER, "blobs.svg", &panels("Laplacian-of-Gaussian blobs, drawn with radius σ√2", &[panel], 1, 300.0));
    let mut plot = Plot::new("Scale-normalised LoG response at each spot's centre", "σ (pixels)", "−σ² ∇²G * image");
    for (k, &(r, c, s)) in [(20usize, 20usize, 2.0), (72, 75, 3.0), (30, 65, 4.0), (70, 30, 6.0)].iter().enumerate() {
        let pts: Vec<(f64, f64)> = lap_scales
            .iter()
            .map(|&sigma| {
                let g = gaussian_blur(&img, sigma).unwrap();
                let lap = g[(r - 1, c)] + g[(r + 1, c)] + g[(r, c - 1)] + g[(r, c + 1)] - 4.0 * g[(r, c)];
                (sigma, -sigma * sigma * lap)
            })
            .collect();
        plot.series.push(Series::new(&format!("spot σ = {s}"), pts, Style::Line, k));
    }
    plot.legend = Legend::TopRight;
    write(CHAPTER, "blob_scale.svg", &plot.render());
    // SIFT and SURF under rotation, scaling and a change of contrast.
    let n = 160;
    let base = feature_scene(n);
    let mut rows = Vec::new();
    let mut shown = None;
    for &(angle_deg, scale, gain) in &[(0.0, 1.0, 0.6), (30.0, 1.0, 1.0), (0.0, 0.75, 1.0), (45.0, 0.8, 0.7)] {
        let angle = (angle_deg as f64).to_radians();
        let other = warp(&base, angle, scale, 0.4).mapv(|v| gain * v + 0.1);
        let t0 = Instant::now();
        let ka = sift(&base, &SiftParams::default()).unwrap();
        let kb = sift(&other, &SiftParams::default()).unwrap();
        let t_sift = t0.elapsed().as_secs_f64() / 2.0;
        let da: Vec<Vec<f64>> = ka.iter().map(|k| k.descriptor.clone()).collect();
        let db: Vec<Vec<f64>> = kb.iter().map(|k| k.descriptor.clone()).collect();
        let m = match_descriptors(&da, &db, 0.8, true).unwrap();
        let correct = m.iter().filter(|mm| {
            let (x, y) = forward(ka[mm.index_a].x, ka[mm.index_a].y, n, angle, scale);
            (x - kb[mm.index_b].x).hypot(y - kb[mm.index_b].y) < 3.0
        }).count();
        let t0 = Instant::now();
        let sa = surf(&base, &SurfParams::default()).unwrap();
        let sb = surf(&other, &SurfParams::default()).unwrap();
        let t_surf = t0.elapsed().as_secs_f64() / 2.0;
        let dsa: Vec<Vec<f64>> = sa.iter().map(|k| k.descriptor.clone()).collect();
        let dsb: Vec<Vec<f64>> = sb.iter().map(|k| k.descriptor.clone()).collect();
        let ms: Vec<_> = match_descriptors(&dsa, &dsb, 0.8, true).unwrap().into_iter().filter(|mm| sa[mm.index_a].laplacian_sign == sb[mm.index_b].laplacian_sign).collect();
        let correct_s = ms.iter().filter(|mm| {
            let (x, y) = forward(sa[mm.index_a].x, sa[mm.index_a].y, n, angle, scale);
            (x - sb[mm.index_b].x).hypot(y - sb[mm.index_b].y) < 3.0
        }).count();
        let what = format!("rotated {angle_deg}°, scaled {scale}, contrast × {gain}");
        rows.push(vec![
            what,
            format!("{} / {}", ka.len(), kb.len()),
            format!("{} ({:.0}% correct)", m.len(), 100.0 * correct as f64 / m.len().max(1) as f64),
            format!("{:.0}", 1e3 * t_sift),
            format!("{} / {}", sa.len(), sb.len()),
            format!("{} ({:.0}% correct)", ms.len(), 100.0 * correct_s as f64 / ms.len().max(1) as f64),
            format!("{:.0}", 1e3 * t_surf),
        ]);
        if angle_deg == 45.0 {
            shown = Some((other, ka, kb, m));
        }
    }
    write(CHAPTER, "matching_table.md", &md_table(&["second image", "SIFT keypoints", "SIFT matches", "SIFT ms per image", "SURF keypoints", "SURF matches", "SURF ms per image"], &rows));
    let (other, ka, kb, m) = shown.unwrap();
    // Side by side, with match lines.
    let pair = Array2::from_shape_fn((n, 2 * n), |(r, c)| if c < n { base[(r, c)] } else { other[(r, c - n)] });
    let mut panel = Panel::new("SIFT matches between the scene and a copy rotated 45°, scaled 0.8, contrast × 0.7", pair, Colours::Gray(0.0, 1.0));
    for mm in &m {
        let (a, b) = (&ka[mm.index_a], &kb[mm.index_b]);
        let (x, y) = forward(a.x, a.y, n, 45f64.to_radians(), 0.8);
        let good = (x - b.x).hypot(y - b.y) < 3.0;
        panel.marks.push(Mark::Line { from: (a.x, a.y), to: (b.x + n as f64, b.y), colour: if good { "#30d040" } else { "#ff3030" } });
        panel.marks.push(Mark::Circle { x: a.x, y: a.y, radius: 1.5 * a.sigma, colour: "#ffd000" });
        panel.marks.push(Mark::Circle { x: b.x + n as f64, y: b.y, radius: 1.5 * b.sigma, colour: "#ffd000" });
    }
    write(CHAPTER, "sift_matches.svg", &panels("Matched SIFT keypoints (circles of radius 1.5σ): green correct, red wrong", &[panel], 1, 640.0));
}

// ------------------------------------------------------------------ helpers

struct Rng(Xoshiro256PlusPlus);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(Xoshiro256PlusPlus::from_seed(seed))
    }
    fn uniform(&mut self) -> f64 {
        self.0.next_f64()
    }
    fn gauss(&mut self) -> f64 {
        let (u, v) = (1.0 - self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos()
    }
}
