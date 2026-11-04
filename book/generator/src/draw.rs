//! Geometric drawings for the book: panels with equal scales on both axes, holding filled
//! polygons, segments and dots in data coordinates. Three-dimensional scenes are drawn by
//! projecting onto the page first (see `project`).

use crate::svg::PALETTE;
use std::fmt::Write;

/// One thing to draw, in data coordinates.
pub enum Shape {
    /// A closed polygon with a fill (`None` for an outline only), a stroke colour and width.
    Polygon { points: Vec<[f64; 2]>, fill: Option<String>, stroke: Option<String>, width: f64 },
    /// An open polyline.
    Path { points: Vec<[f64; 2]>, colour: &'static str, width: f64, dashed: bool },
    /// A filled dot of a radius in pixels.
    Dot { at: [f64; 2], radius: f64, colour: &'static str },
}

/// One panel: a caption, the data-space window `[xmin, xmax] × [ymin, ymax]`, and its shapes.
pub struct Scene {
    pub caption: String,
    pub window: [f64; 4],
    pub shapes: Vec<Shape>,
}

impl Scene {
    pub fn new(caption: &str, window: [f64; 4]) -> Self {
        Scene { caption: caption.to_string(), window, shapes: Vec::new() }
    }
    pub fn polygon(&mut self, points: Vec<[f64; 2]>, fill: Option<String>, stroke: Option<&str>, width: f64) {
        self.shapes.push(Shape::Polygon { points, fill, stroke: stroke.map(str::to_string), width });
    }
    pub fn path(&mut self, points: Vec<[f64; 2]>, colour: &'static str, width: f64) {
        self.shapes.push(Shape::Path { points, colour, width, dashed: false });
    }
    pub fn dashed(&mut self, points: Vec<[f64; 2]>, colour: &'static str, width: f64) {
        self.shapes.push(Shape::Path { points, colour, width, dashed: true });
    }
    pub fn dot(&mut self, at: [f64; 2], radius: f64, colour: &'static str) {
        self.shapes.push(Shape::Dot { at, radius, colour });
    }
    /// The window enclosing the given points with a relative margin.
    pub fn fit(points: impl IntoIterator<Item = [f64; 2]>, margin: f64) -> [f64; 4] {
        let mut w = [f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY];
        for p in points {
            w = [w[0].min(p[0]), w[1].max(p[0]), w[2].min(p[1]), w[3].max(p[1])];
        }
        let m = margin * (w[1] - w[0]).max(w[3] - w[2]);
        [w[0] - m, w[1] + m, w[2] - m, w[3] + m]
    }
}

/// A palette colour with an opacity, as an SVG fill.
pub fn tint(colour: usize, opacity: f64) -> String {
    let hex = PALETTE[colour % PALETTE.len()];
    let v = |k: usize| u8::from_str_radix(&hex[k..k + 2], 16).unwrap();
    format!("rgba({},{},{},{opacity})", v(1), v(3), v(5))
}

/// A grey level from 0 (black) to 1 (white), for shading faces.
pub fn grey(level: f64, tone: usize) -> String {
    let hex = PALETTE[tone % PALETTE.len()];
    let v = |k: usize| u8::from_str_radix(&hex[k..k + 2], 16).unwrap() as f64;
    let l = level.clamp(0.0, 1.0);
    // Blend the palette colour with white by the light level.
    let mix = |c: f64| (c * (0.35 + 0.65 * l) + 255.0 * 0.45 * l).min(255.0);
    format!("rgb({:.0},{:.0},{:.0})", mix(v(1)), mix(v(3)), mix(v(5)))
}

/// Renders the scenes as a grid with `columns` panels per row, each `width` pixels wide and as
/// tall as its window's aspect ratio requires (rows take the tallest panel).
pub fn render(title: &str, scenes: &[Scene], columns: usize, width: f64) -> String {
    let (pad, cap, head) = (12.0, 22.0, if title.is_empty() { 8.0 } else { 30.0 });
    let rows = scenes.len().div_ceil(columns);
    let height = |s: &Scene| width * (s.window[3] - s.window[2]) / (s.window[1] - s.window[0]);
    let row_heights: Vec<f64> = (0..rows).map(|r| scenes[r * columns..((r + 1) * columns).min(scenes.len())].iter().map(height).fold(0.0, f64::max)).collect();
    let total_w = columns.min(scenes.len()) as f64 * (width + pad) + pad;
    let total_h = head + row_heights.iter().map(|h| h + cap + pad).sum::<f64>();
    let mut svg = String::new();
    write!(svg, r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w:.0} {total_h:.0}" font-family="sans-serif" font-size="13">"#).unwrap();
    write!(svg, r#"<rect width="100%" height="100%" fill="white"/>"#).unwrap();
    if !title.is_empty() {
        write!(svg, r#"<text x="{:.1}" y="20" text-anchor="middle" font-size="15">{}</text>"#, total_w / 2.0, escape(title)).unwrap();
    }
    let mut y0 = head;
    for (r, &rh) in row_heights.iter().enumerate() {
        for (c, scene) in scenes[r * columns..((r + 1) * columns).min(scenes.len())].iter().enumerate() {
            let x0 = pad + c as f64 * (width + pad);
            let h = height(scene);
            let top = y0 + (rh - h) / 2.0;
            let [xa, xb, _, yb] = scene.window;
            let sx = width / (xb - xa);
            let map = |p: [f64; 2]| (x0 + (p[0] - xa) * sx, top + (yb - p[1]) * sx);
            let id = format!("c{r}_{c}");
            write!(svg, r#"<clipPath id="{id}"><rect x="{x0:.1}" y="{top:.1}" width="{width:.1}" height="{h:.1}"/></clipPath>"#).unwrap();
            write!(svg, r##"<rect x="{x0:.1}" y="{top:.1}" width="{width:.1}" height="{h:.1}" fill="none" stroke="#cccccc"/>"##).unwrap();
            write!(svg, r#"<g clip-path="url(#{id})">"#).unwrap();
            for shape in &scene.shapes {
                match shape {
                    Shape::Polygon { points, fill, stroke, width } => {
                        let pts: Vec<String> = points.iter().map(|&p| { let (x, y) = map(p); format!("{x:.1},{y:.1}") }).collect();
                        write!(svg, r#"<polygon points="{}" fill="{}" stroke="{}" stroke-width="{width}" stroke-linejoin="round"/>"#, pts.join(" "), fill.as_deref().unwrap_or("none"), stroke.as_deref().unwrap_or("none")).unwrap();
                    }
                    Shape::Path { points, colour, width, dashed } => {
                        let pts: Vec<String> = points.iter().map(|&p| { let (x, y) = map(p); format!("{x:.1},{y:.1}") }).collect();
                        let dash = if *dashed { r#" stroke-dasharray="5,3""# } else { "" };
                        write!(svg, r#"<polyline points="{}" fill="none" stroke="{colour}" stroke-width="{width}" stroke-linejoin="round"{dash}/>"#, pts.join(" ")).unwrap();
                    }
                    Shape::Dot { at, radius, colour } => {
                        let (x, y) = map(*at);
                        write!(svg, r#"<circle cx="{x:.1}" cy="{y:.1}" r="{radius}" fill="{colour}"/>"#).unwrap();
                    }
                }
            }
            svg.push_str("</g>");
            write!(svg, r#"<text x="{:.1}" y="{:.1}" text-anchor="middle">{}</text>"#, x0 + width / 2.0, y0 + rh + 16.0, escape(&scene.caption)).unwrap();
        }
        y0 += rh + cap + pad;
    }
    svg.push_str("</svg>");
    svg
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// Orthographic projection of 3D points for a view from azimuth `az` and elevation `el` (radians):
/// returns page coordinates `(x, y)` and the depth towards the viewer.
pub fn project(p: [f64; 3], az: f64, el: f64) -> ([f64; 2], f64) {
    let (sa, ca) = az.sin_cos();
    let (se, ce) = el.sin_cos();
    let x = -sa * p[0] + ca * p[1];
    let y = -ca * se * p[0] - sa * se * p[1] + ce * p[2];
    let depth = ca * ce * p[0] + sa * ce * p[1] + se * p[2];
    ([x, y], depth)
}

/// Adds a triangle mesh to a scene, seen from `(azimuth, elevation)`, shaded by a fixed light and
/// painted back to front. Faces
/// pointing away from the viewer are skipped when `cull` is set (closed surfaces).
pub fn add_mesh(scene: &mut Scene, vertices: &[[f64; 3]], triangles: &[[usize; 3]], (az, el): (f64, f64), tone: usize, cull: bool, edges: Option<&'static str>) {
    let light = {
        let l = [0.3f64, -0.5, 0.8];
        let n = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt();
        [l[0] / n, l[1] / n, l[2] / n]
    };
    let (sa, ca) = az.sin_cos();
    let (se, ce) = el.sin_cos();
    let toward = [ca * ce, sa * ce, se];
    let mut faces: Vec<(f64, Vec<[f64; 2]>, f64)> = Vec::new();
    for t in triangles {
        let [a, b, c] = t.map(|i| vertices[i]);
        let (u, v) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
        let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len == 0.0 {
            continue;
        }
        let n = [n[0] / len, n[1] / len, n[2] / len];
        let facing = n[0] * toward[0] + n[1] * toward[1] + n[2] * toward[2];
        if cull && facing < 0.0 {
            continue;
        }
        let s = if facing < 0.0 { -1.0 } else { 1.0 };
        let lit = (s * (n[0] * light[0] + n[1] * light[1] + n[2] * light[2])).max(0.0);
        let pts: Vec<([f64; 2], f64)> = [a, b, c].iter().map(|&p| project(p, az, el)).collect();
        let depth = pts.iter().map(|p| p.1).sum::<f64>() / 3.0;
        faces.push((depth, pts.into_iter().map(|p| p.0).collect(), 0.25 + 0.75 * lit));
    }
    faces.sort_by(|x, y| x.0.total_cmp(&y.0));
    for (_, pts, light) in faces {
        let fill = grey(light, tone);
        // Without drawn edges, a hairline in the fill colour hides the seams between triangles.
        let (stroke, width) = match edges {
            Some(colour) => (colour.to_string(), 0.4),
            None => (fill.clone(), 0.3),
        };
        scene.shapes.push(Shape::Polygon { points: pts, fill: Some(fill), stroke: Some(stroke), width });
    }
}
