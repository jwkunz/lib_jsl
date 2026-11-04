//! Figures made of images: a grid of panels, each a raster (encoded as PNG by the library's own
//! codec and embedded in the SVG) with optional vector overlays in pixel coordinates.

use lib_jsl_file_io::png::{RasterImage, png_encode};
use ndarray::Array2;
use std::fmt::Write;

/// How pixel values become colours.
#[derive(Clone, Copy)]
pub enum Colours {
    /// Black at the low end of the range to white at the high end.
    Gray(f64, f64),
    /// The viridis-like ramp of the line plots' heat maps.
    Ramp(f64, f64),
    /// Small non-negative integers as distinct colours (0 is black).
    Labels,
}

/// Something drawn over an image, in pixel coordinates (`x` the column, `y` the row, both measured
/// to pixel centres).
pub enum Mark {
    /// Pixels where the mask is true, painted in a colour.
    Mask(Array2<bool>, &'static str),
    Circle { x: f64, y: f64, radius: f64, colour: &'static str },
    Line { from: (f64, f64), to: (f64, f64), colour: &'static str },
    Dot { x: f64, y: f64, colour: &'static str },
}

pub struct Panel {
    pub caption: String,
    pub image: Array2<f64>,
    pub colours: Colours,
    pub marks: Vec<Mark>,
}

impl Panel {
    pub fn new(caption: &str, image: Array2<f64>, colours: Colours) -> Self {
        Panel { caption: caption.to_string(), image, colours, marks: Vec::new() }
    }
}

const LABEL_COLOURS: [(u8, u8, u8); 8] = [(0, 0, 0), (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75), (227, 119, 194)];

fn ramp(t: f64) -> (u8, u8, u8) {
    const STOPS: [(f64, f64, f64); 5] = [(68.0, 1.0, 84.0), (59.0, 82.0, 139.0), (33.0, 145.0, 140.0), (94.0, 201.0, 98.0), (253.0, 231.0, 37.0)];
    let t = t.clamp(0.0, 1.0) * 4.0;
    let i = (t.floor() as usize).min(3);
    let f = t - i as f64;
    let mix = |a: f64, b: f64| (a + f * (b - a)).round() as u8;
    (mix(STOPS[i].0, STOPS[i + 1].0), mix(STOPS[i].1, STOPS[i + 1].1), mix(STOPS[i].2, STOPS[i + 1].2))
}

fn hex(colour: &str) -> (u8, u8, u8) {
    let v = u32::from_str_radix(colour.trim_start_matches('#'), 16).expect("a #rrggbb colour");
    ((v >> 16) as u8, (v >> 8) as u8, v as u8)
}

fn base64(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        for k in 0..4 {
            s.push(if k <= chunk.len() { TABLE[(n >> (18 - 6 * k) & 63) as usize] as char } else { '=' });
        }
    }
    s
}

fn png_uri(width: usize, height: usize, channels: usize, samples: Vec<u16>) -> String {
    let image = RasterImage::new(width, height, channels, 8, samples).expect("valid raster");
    format!("data:image/png;base64,{}", base64(&png_encode(&image, 9).expect("PNG encoding")))
}

fn image_uri(image: &Array2<f64>, colours: Colours) -> String {
    let (rows, cols) = image.dim();
    let mut samples = Vec::with_capacity(rows * cols * 3);
    for &v in image.iter() {
        let (r, g, b) = match colours {
            Colours::Gray(lo, hi) => {
                let g = (255.0 * ((v - lo) / (hi - lo)).clamp(0.0, 1.0)).round() as u8;
                (g, g, g)
            }
            Colours::Ramp(lo, hi) => ramp((v - lo) / (hi - lo)),
            Colours::Labels => LABEL_COLOURS[(v.max(0.0) as usize) % LABEL_COLOURS.len()],
        };
        samples.extend([r as u16, g as u16, b as u16]);
    }
    png_uri(cols, rows, 3, samples)
}

fn mask_uri(mask: &Array2<bool>, colour: &str) -> String {
    let (rows, cols) = mask.dim();
    let (r, g, b) = hex(colour);
    let samples = mask.iter().flat_map(|&m| [r as u16, g as u16, b as u16, if m { 255 } else { 0 }]).collect();
    png_uri(cols, rows, 4, samples)
}

/// Renders panels in a grid of `columns`, each image drawn `size` pixels wide.
pub fn panels(title: &str, panels: &[Panel], columns: usize, size: f64) -> String {
    let (gap, caption, head) = (16.0, 34.0, 40.0);
    let rows_of_panels = panels.len().div_ceil(columns);
    let heights: Vec<f64> = panels.iter().map(|p| size * p.image.nrows() as f64 / p.image.ncols() as f64).collect();
    let row_height = |k: usize| heights[k * columns..((k + 1) * columns).min(panels.len())].iter().cloned().fold(0.0, f64::max) + caption;
    let width = columns as f64 * (size + gap) + gap;
    let height = head + (0..rows_of_panels).map(row_height).sum::<f64>() + gap;
    let mut s = String::new();
    let _ = write!(s, r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif" font-size="13">"#);
    let _ = write!(s, r##"<rect width="{width}" height="{height}" fill="#ffffff" rx="6"/>"##);
    let _ = write!(s, r##"<text x="{}" y="26" text-anchor="middle" font-size="15" font-weight="bold" fill="#222">{}</text>"##, width / 2.0, escape(title));
    let mut y0 = head;
    for k in 0..rows_of_panels {
        for (j, p) in panels.iter().enumerate().skip(k * columns).take(columns) {
            let x0 = gap + (j - k * columns) as f64 * (size + gap);
            let (rows, cols) = p.image.dim();
            let scale = size / cols as f64;
            let h = heights[j];
            let _ = write!(s, r#"<image x="{x0}" y="{y0}" width="{size}" height="{h}" preserveAspectRatio="none" style="image-rendering:pixelated" href="{}"/>"#, image_uri(&p.image, p.colours));
            // Overlays: pixel (row, col) has its centre at (col + 0.5, row + 0.5) in image units.
            let _ = write!(s, r#"<clipPath id="panel{j}"><rect x="{x0}" y="{y0}" width="{size}" height="{h}"/></clipPath><g clip-path="url(#panel{j})"><g transform="translate({x0} {y0}) scale({scale})">"#);
            for m in &p.marks {
                match m {
                    Mark::Mask(mask, colour) => {
                        let _ = write!(s, r#"<image x="0" y="0" width="{cols}" height="{rows}" preserveAspectRatio="none" style="image-rendering:pixelated" href="{}"/>"#, mask_uri(mask, colour));
                    }
                    Mark::Circle { x, y, radius, colour } => {
                        let _ = write!(s, r#"<circle cx="{:.2}" cy="{:.2}" r="{radius:.2}" fill="none" stroke="{colour}" stroke-width="2" vector-effect="non-scaling-stroke"/>"#, x + 0.5, y + 0.5);
                    }
                    Mark::Line { from, to, colour } => {
                        let _ = write!(s, r#"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="{colour}" stroke-width="2" vector-effect="non-scaling-stroke"/>"#, from.0 + 0.5, from.1 + 0.5, to.0 + 0.5, to.1 + 0.5);
                    }
                    Mark::Dot { x, y, colour } => {
                        let _ = write!(s, r#"<circle cx="{:.2}" cy="{:.2}" r="{:.2}" fill="{colour}"/>"#, x + 0.5, y + 0.5, 3.0 / scale);
                    }
                }
            }
            let _ = write!(s, r##"</g></g><rect x="{x0}" y="{y0}" width="{size}" height="{h}" fill="none" stroke="#666"/>"##);
            let _ = write!(s, r##"<text x="{}" y="{}" text-anchor="middle" fill="#222">{}</text>"##, x0 + size / 2.0, y0 + h + 20.0, escape(&p.caption));
        }
        y0 += row_height(k);
    }
    s.push_str("</svg>");
    s
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}
