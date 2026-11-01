//! A minimal SVG line-plot writer for the book's figures (no external dependencies).

use std::fmt::Write;

/// Okabe-Ito colours, distinguishable with common colour-vision deficiencies.
pub const PALETTE: [&str; 8] = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#AAAAAA"];

#[derive(Clone, Copy, PartialEq)]
pub enum Style {
    Line,
    Markers,
    LineMarkers,
    Dashed,
    /// Vertical stems from y = 0 with a marker on top.
    Stems,
}

pub struct Series {
    pub label: String,
    pub points: Vec<(f64, f64)>,
    pub style: Style,
    pub color: &'static str,
}

impl Series {
    pub fn new(label: &str, points: Vec<(f64, f64)>, style: Style, color: usize) -> Self {
        Series { label: label.to_string(), points, style, color: PALETTE[color % PALETTE.len()] }
    }
}

/// Where the legend box sits inside the plot area.
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq)]
pub enum Legend {
    TopRight,
    TopLeft,
    BottomRight,
    BottomLeft,
}

pub struct Plot {
    pub legend: Legend,
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub log_x: bool,
    pub log_y: bool,
    pub x_range: Option<(f64, f64)>,
    pub y_range: Option<(f64, f64)>,
    pub series: Vec<Series>,
}

impl Plot {
    pub fn new(title: &str, x_label: &str, y_label: &str) -> Self {
        Plot { legend: Legend::TopRight, title: title.into(), x_label: x_label.into(), y_label: y_label.into(), log_x: false, log_y: false, x_range: None, y_range: None, series: Vec::new() }
    }
}

fn nice_step(span: f64) -> f64 {
    let raw = span / 6.0;
    let mag = 10f64.powf(raw.log10().floor());
    let r = raw / mag;
    mag * if r < 1.5 { 1.0 } else if r < 3.5 { 2.0 } else if r < 7.5 { 5.0 } else { 10.0 }
}

fn format_tick(v: f64, step: f64) -> String {
    if v.abs() < step * 1e-9 {
        return "0".into();
    }
    let decimals = (-step.log10().floor()).max(0.0) as usize;
    format!("{v:.decimals$}")
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

impl Plot {
    pub fn render(&self) -> String {
        let (w, h) = (680.0, 420.0);
        let (left, right, top, bottom) = (78.0, 20.0, 40.0, 58.0);
        let (pw, ph) = (w - left - right, h - top - bottom);
        let tx = |v: f64| if self.log_x { v.log10() } else { v };
        let ty = |v: f64| if self.log_y { v.log10() } else { v };
        let finite: Vec<(f64, f64)> = self.series.iter().flat_map(|s| s.points.iter().copied()).filter(|&(x, y)| x.is_finite() && y.is_finite() && (!self.log_x || x > 0.0) && (!self.log_y || y > 0.0)).collect();
        let span = |vals: Vec<f64>| vals.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
        let (mut x0, mut x1) = self.x_range.map(|(a, b)| (tx(a), tx(b))).unwrap_or_else(|| span(finite.iter().map(|p| tx(p.0)).collect()));
        let (mut y0, mut y1) = self.y_range.map(|(a, b)| (ty(a), ty(b))).unwrap_or_else(|| span(finite.iter().map(|p| ty(p.1)).collect()));
        if self.log_y && self.y_range.is_none() {
            y0 = y0.floor();
            y1 = y1.ceil();
        }
        if self.log_x && self.x_range.is_none() {
            x0 = x0.floor();
            x1 = x1.ceil();
        }
        if x1 <= x0 {
            x1 = x0 + 1.0;
        }
        if y1 <= y0 {
            y1 = y0 + 1.0;
        }
        if !self.log_y && self.y_range.is_none() {
            let pad = 0.05 * (y1 - y0);
            y0 -= pad;
            y1 += pad;
        }
        let px = |v: f64| left + (tx(v) - x0) / (x1 - x0) * pw;
        let py = |v: f64| top + ph - (ty(v) - y0) / (y1 - y0) * ph;
        let mut s = String::new();
        let _ = write!(s, r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="Helvetica, Arial, sans-serif" font-size="13">"#);
        let _ = write!(s, r##"<rect width="{w}" height="{h}" fill="#ffffff" rx="6"/>"##);
        let _ = write!(s, r##"<text x="{}" y="24" text-anchor="middle" font-size="15" font-weight="bold" fill="#222">{}</text>"##, left + pw / 2.0, escape(&self.title));
        // Grid and ticks.
        let axis_ticks = |log: bool, lo: f64, hi: f64, horizontal: bool, s: &mut String| {
            let ticks: Vec<(f64, String)> = if log {
                // Label at most about ten decades, thinning evenly.
                let (first, last) = (lo.ceil() as i64, hi.floor() as i64);
                let stride = ((last - first) / 10 + 1).max(1);
                (first..=last).filter(|e| (e - first) % stride == 0).map(|e| (e as f64, format!("10<tspan dy=\"-6\" font-size=\"10\">{e}</tspan>"))).collect()
            } else {
                let step = nice_step(hi - lo);
                let mut v = (lo / step).ceil() * step;
                let mut out = Vec::new();
                while v <= hi + 1e-9 * step {
                    out.push((v, format_tick(v, step)));
                    v += step;
                }
                out
            };
            for (t, label) in ticks {
                if horizontal {
                    let x = left + (t - lo) / (hi - lo) * pw;
                    let _ = write!(s, r##"<line x1="{x:.1}" y1="{top}" x2="{x:.1}" y2="{}" stroke="#e5e5e5"/>"##, top + ph);
                    let _ = write!(s, r##"<text x="{x:.1}" y="{}" text-anchor="middle" fill="#333">{label}</text>"##, top + ph + 18.0);
                } else {
                    let y = top + ph - (t - lo) / (hi - lo) * ph;
                    let _ = write!(s, r##"<line x1="{left}" y1="{y:.1}" x2="{}" y2="{y:.1}" stroke="#e5e5e5"/>"##, left + pw);
                    let _ = write!(s, r##"<text x="{}" y="{:.1}" text-anchor="end" fill="#333">{label}</text>"##, left - 8.0, y + 4.0);
                }
            }
        };
        axis_ticks(self.log_x, x0, x1, true, &mut s);
        axis_ticks(self.log_y, y0, y1, false, &mut s);
        let _ = write!(s, r##"<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#666"/>"##);
        let _ = write!(s, r##"<text x="{}" y="{}" text-anchor="middle" fill="#222">{}</text>"##, left + pw / 2.0, h - 14.0, escape(&self.x_label));
        let _ = write!(s, r##"<text transform="translate(18 {}) rotate(-90)" text-anchor="middle" fill="#222">{}</text>"##, top + ph / 2.0, escape(&self.y_label));
        let _ = write!(s, r#"<clipPath id="plot"><rect x="{left}" y="{top}" width="{pw}" height="{ph}"/></clipPath><g clip-path="url(#plot)">"#);
        for series in &self.series {
            let pts: Vec<(f64, f64)> = series.points.iter().copied().filter(|&(x, y)| x.is_finite() && y.is_finite() && (!self.log_x || x > 0.0) && (!self.log_y || y > 0.0)).collect();
            let path: String = pts.iter().enumerate().map(|(i, &(x, y))| format!("{}{:.2},{:.2}", if i == 0 { "M" } else { "L" }, px(x), py(y))).collect();
            match series.style {
                Style::Line | Style::LineMarkers | Style::Dashed => {
                    let dash = if series.style == Style::Dashed { r#" stroke-dasharray="6 4""# } else { "" };
                    let _ = write!(s, r#"<path d="{path}" fill="none" stroke="{}" stroke-width="2"{dash}/>"#, series.color);
                }
                Style::Stems => {
                    let base = if self.log_y { y0 } else { 0.0 };
                    for &(x, y) in &pts {
                        let b = if self.log_y { top + ph } else { py(base) };
                        let _ = write!(s, r#"<line x1="{:.2}" y1="{b:.2}" x2="{:.2}" y2="{:.2}" stroke="{}" stroke-width="1.5"/>"#, px(x), px(x), py(y), series.color);
                    }
                }
                Style::Markers => {}
            }
            if matches!(series.style, Style::Markers | Style::LineMarkers | Style::Stems) {
                for &(x, y) in &pts {
                    let _ = write!(s, r##"<circle cx="{:.2}" cy="{:.2}" r="3.5" fill="{}" stroke="#fff" stroke-width="0.8"/>"##, px(x), py(y), series.color);
                }
            }
        }
        s.push_str("</g>");
        // Legend.
        let labelled: Vec<&Series> = self.series.iter().filter(|s| !s.label.is_empty()).collect();
        if !labelled.is_empty() {
            let lw = 18.0 + 7.2 * labelled.iter().map(|s| s.label.chars().count()).max().unwrap_or(0) as f64 + 30.0;
            let lh = 8.0 + 20.0 * labelled.len() as f64;
            let (lx, ly) = match self.legend {
                Legend::TopRight => (left + pw - lw - 8.0, top + 8.0),
                Legend::TopLeft => (left + 8.0, top + 8.0),
                Legend::BottomRight => (left + pw - lw - 8.0, top + ph - lh - 8.0),
                Legend::BottomLeft => (left + 8.0, top + ph - lh - 8.0),
            };
            let _ = write!(s, r##"<rect x="{lx:.1}" y="{ly:.1}" width="{lw:.1}" height="{}" fill="#ffffff" fill-opacity="0.92" stroke="#bbb" rx="4"/>"##, 8.0 + 20.0 * labelled.len() as f64);
            for (i, series) in labelled.iter().enumerate() {
                let y = ly + 18.0 + 20.0 * i as f64;
                let dash = if series.style == Style::Dashed { r#" stroke-dasharray="6 4""# } else { "" };
                if series.style == Style::Markers || series.style == Style::Stems {
                    let _ = write!(s, r#"<circle cx="{:.1}" cy="{:.1}" r="4" fill="{}"/>"#, lx + 18.0, y - 4.0, series.color);
                } else {
                    let _ = write!(s, r#"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{}" stroke-width="2.5"{dash}/>"#, lx + 8.0, y - 4.0, lx + 30.0, y - 4.0, series.color);
                }
                let _ = write!(s, r##"<text x="{:.1}" y="{y:.1}" fill="#222">{}</text>"##, lx + 38.0, escape(&series.label));
            }
        }
        s.push_str("</svg>\n");
        s
    }
}
