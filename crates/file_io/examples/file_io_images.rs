//! PNG, JPEG and Netpbm images.

use lib_jsl_file_io::{RasterImage, jpeg_decode, jpeg_encode, png_decode, png_encode, ppm_decode, ppm_encode, read_png_file, write_png_file};

fn main() {
    // A 64 x 48 RGB test card: colour ramps and a diagonal stripe.
    let (w, h) = (64, 48);
    let samples: Vec<u16> = (0..h)
        .flat_map(|y| (0..w).flat_map(move |x| [(x * 4) as u16, (y * 5) as u16, if (x + y) % 16 < 8 { 220 } else { 30 }]))
        .collect();
    let image = RasterImage::new(w, h, 3, 8, samples.clone()).unwrap();

    let png = png_encode(&image, 9).unwrap();
    println!("PNG: {} bytes for {} raw, decodes identically: {}", png.len(), w * h * 3, png_decode(&png).unwrap() == image);
    let path = std::env::temp_dir().join("lib_jsl_card.png");
    write_png_file(&path, &image).unwrap();
    println!("PNG file round trip identical: {}", read_png_file(&path).unwrap() == image);

    let ppm = ppm_encode(&image).unwrap();
    println!("PPM: {} bytes, header {:?}, identical: {}", ppm.len(), String::from_utf8_lossy(&ppm[..13]), ppm_decode(&ppm).unwrap() == image);

    let bytes: Vec<u8> = samples.iter().map(|&v| v as u8).collect();
    for quality in [25, 75, 95] {
        let jpeg = jpeg_encode(&bytes, w, h, 3, quality).unwrap();
        let back = jpeg_decode(&jpeg).unwrap();
        let mse = bytes.iter().zip(&back.pixels).map(|(&a, &b)| (a as f64 - b as f64).powi(2)).sum::<f64>() / bytes.len() as f64;
        println!("JPEG quality {quality}: {} bytes, PSNR {:.1} dB", jpeg.len(), 10.0 * (255.0f64.powi(2) / mse).log10());
    }
}
