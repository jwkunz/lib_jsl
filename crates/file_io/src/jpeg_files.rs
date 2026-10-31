//! Reading and writing JPEG files with the workspace's baseline codec.

use lib_jsl_core::ErrorsJSL;
use crate::jpeg::{JpegImage, jpeg_decode, jpeg_encode};
use std::path::Path;

/// Reads and decodes a baseline JPEG file (grayscale or RGB).
pub fn read_jpeg_file(path: impl AsRef<Path>) -> Result<JpegImage, ErrorsJSL> {
    let data = std::fs::read(path).map_err(|_| ErrorsJSL::RuntimeError("Could not read the JPEG file."))?;
    jpeg_decode(&data)
}

/// Encodes `image` at `quality` (1 to 100) and writes it to `path`.
pub fn write_jpeg_file(path: impl AsRef<Path>, image: &JpegImage, quality: u8) -> Result<(), ErrorsJSL> {
    let bytes = jpeg_encode(&image.pixels, image.width, image.height, image.channels, quality)?;
    std::fs::write(path, bytes).map_err(|_| ErrorsJSL::RuntimeError("Could not write the JPEG file."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_round_trip_is_close() {
        let (w, h) = (40, 24);
        let pixels: Vec<u8> = (0..w * h * 3).map(|i| ((i / 3 % w) * 6 + (i % 3) * 20) as u8).collect();
        let img = JpegImage { width: w, height: h, channels: 3, pixels };
        let path = std::env::temp_dir().join(format!("lib_jsl_jpeg_test_{}.jpg", std::process::id()));
        write_jpeg_file(&path, &img, 95).unwrap();
        let back = read_jpeg_file(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!((back.width, back.height, back.channels), (w, h, 3));
        let mean_err = img.pixels.iter().zip(&back.pixels).map(|(&a, &b)| (a as f64 - b as f64).abs()).sum::<f64>() / img.pixels.len() as f64;
        assert!(mean_err < 4.0, "mean error {mean_err}");
        assert!(read_jpeg_file(&path).is_err());
    }
}
