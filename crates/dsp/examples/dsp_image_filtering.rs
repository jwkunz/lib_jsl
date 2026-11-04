//! Image filtering: 2D convolution with border modes, Gaussian blur, Sobel gradients and unsharp masking.

use lib_jsl_dsp::discrete::image::convolution::{Border, convolve2d};
use lib_jsl_dsp::discrete::image::filters::gaussian_blur::gaussian_blur;
use lib_jsl_dsp::discrete::image::filters::sharpening::sharpening;
use lib_jsl_dsp::discrete::image::filters::sobel::sobel;
use ndarray::Array2;

fn main() {
    // A vertical step edge: dark on the left, bright on the right.
    let step = Array2::from_shape_fn((9, 9), |(_, c)| if c < 4 { 0.2 } else { 0.8 });
    let box3 = Array2::from_elem((3, 3), 1.0 / 9.0);
    for border in [Border::Zero, Border::Replicate, Border::Reflect] {
        let out = convolve2d(&step, &box3, border).unwrap();
        println!("3x3 mean, {border:?}: corner {:.3}, centre row {:.3?}", out[(0, 0)], out.row(4).to_vec());
    }
    let blurred = gaussian_blur(&step, 1.0).unwrap();
    println!("Gaussian σ = 1 across the edge: {:.3?}", blurred.row(4).to_vec());
    let g = sobel(&step, true).unwrap();
    println!("Sobel gx across the edge: {:.2?} (gy {:.2})", g.gx.row(4).to_vec(), g.gy[(4, 4)]);
    let sharp = sharpening(&blurred, 1.0, 1.0).unwrap();
    println!("unsharp mask of the blur: {:.3?}", sharp.row(4).to_vec());
    let ramp = Array2::from_shape_fn((5, 5), |(r, c)| 2.0 * c as f64 + 3.0 * r as f64);
    let g = sobel(&ramp, true).unwrap();
    println!("Sobel of 2x + 3y at the centre: gx {} gy {}, direction {:.4} rad", g.gx[(2, 2)], g.gy[(2, 2)], g.direction[(2, 2)]);
}
