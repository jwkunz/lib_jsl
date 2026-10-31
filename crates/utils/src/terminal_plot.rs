//! Plots rendered as text for terminals.
//!
//! Curves are drawn on a Braille-dot canvas (each character cell holds 2×4 dots), which gives
//! four times the vertical and twice the horizontal resolution of one-character-per-point plots.
//! Axes carry min/mid/max tick labels, and several series can be overlaid (distinguished by a
//! legend and, optionally, ANSI colour).

/// One curve: paired `x` and `y` samples, drawn as connected line segments.
#[derive(Debug, Clone)]
pub struct PlotSeries<'a> {
    pub label: &'a str,
    pub x: &'a [f64],
    pub y: &'a [f64],
}

/// Layout options for [`render_terminal_plot`].
#[derive(Debug, Clone)]
pub struct PlotOptions {
    /// Plot area width in character cells (at least 4).
    pub width: usize,
    /// Plot area height in character cells (at least 2).
    pub height: usize,
    pub title: Option<String>,
    pub x_label: Option<String>,
    pub y_label: Option<String>,
    /// Colour each series with ANSI escape codes.
    pub color: bool,
    /// Draw only the sample points instead of connecting them.
    pub scatter: bool,
}

impl Default for PlotOptions {
    fn default() -> Self {
        PlotOptions { width: 60, height: 16, title: None, x_label: None, y_label: None, color: false, scatter: false }
    }
}

const GUTTER: usize = 10;
const BRAILLE_BASE: u32 = 0x2800;
// Bit for the dot at (column, row) inside a 2×4 Braille cell.
const DOT_BITS: [[u8; 4]; 2] = [[0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80]];
const COLORS: [&str; 6] = ["\x1b[36m", "\x1b[33m", "\x1b[32m", "\x1b[35m", "\x1b[31m", "\x1b[34m"];

fn format_tick(v: f64) -> String {
    let a = v.abs();
    let s = if a != 0.0 && !(1e-3..1e5).contains(&a) { format!("{v:.2e}") } else { format!("{v:.4}") };
    // Trim trailing zeros in plain decimals.
    if s.contains('e') || !s.contains('.') { s } else { s.trim_end_matches('0').trim_end_matches('.').to_string() }
}

fn extent(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for v in values.filter(|v| v.is_finite()) {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if lo > hi {
        (-1.0, 1.0)
    } else if lo == hi {
        (lo - 1.0, hi + 1.0)
    } else {
        (lo, hi)
    }
}

/// Renders one or more series as a multi-line string (no trailing newline).
pub fn render_terminal_plot(series: &[PlotSeries], opts: &PlotOptions) -> String {
    let (w, h) = (opts.width.max(4), opts.height.max(2));
    let (xmin, xmax) = extent(series.iter().flat_map(|s| s.x.iter().copied()));
    let (ymin, ymax) = extent(series.iter().flat_map(|s| s.y.iter().copied()));
    let (dw, dh) = (w * 2, h * 4);
    // Per cell: dot bits and the index of the last series that touched it.
    let mut cells = vec![(0u8, 0usize); w * h];
    let to_dot = |x: f64, y: f64| -> (i64, i64) {
        let px = ((x - xmin) / (xmax - xmin) * (dw - 1) as f64).round() as i64;
        let py = ((ymax - y) / (ymax - ymin) * (dh - 1) as f64).round() as i64;
        (px, py)
    };
    let mut plot_dot = |px: i64, py: i64, s: usize| {
        if px >= 0 && py >= 0 && (px as usize) < dw && (py as usize) < dh {
            let (cx, cy) = (px as usize / 2, py as usize / 4);
            let cell = &mut cells[cy * w + cx];
            cell.0 |= DOT_BITS[px as usize % 2][py as usize % 4];
            cell.1 = s;
        }
    };
    for (si, s) in series.iter().enumerate() {
        let mut prev: Option<(i64, i64)> = None;
        for (&x, &y) in s.x.iter().zip(s.y) {
            if !x.is_finite() || !y.is_finite() {
                prev = None;
                continue;
            }
            let p = to_dot(x, y);
            match (prev, opts.scatter) {
                (Some(q), false) => {
                    // Bresenham between consecutive samples.
                    let (mut x0, mut y0) = q;
                    let (dx, dy) = ((p.0 - x0).abs(), -(p.1 - y0).abs());
                    let (sx, sy) = (if x0 < p.0 { 1 } else { -1 }, if y0 < p.1 { 1 } else { -1 });
                    let mut err = dx + dy;
                    loop {
                        plot_dot(x0, y0, si);
                        if (x0, y0) == p {
                            break;
                        }
                        let e2 = 2 * err;
                        if e2 >= dy {
                            err += dy;
                            x0 += sx;
                        }
                        if e2 <= dx {
                            err += dx;
                            y0 += sy;
                        }
                    }
                }
                _ => plot_dot(p.0, p.1, si),
            }
            prev = Some(p);
        }
    }

    let mut out: Vec<String> = Vec::new();
    if let Some(t) = &opts.title {
        out.push(format!("{:^width$}", t, width = GUTTER + w + 1));
    }
    if let Some(l) = &opts.y_label {
        out.push(l.clone());
    }
    for row in 0..h {
        let tick = if row == 0 {
            format_tick(ymax)
        } else if row == h - 1 {
            format_tick(ymin)
        } else if row == h / 2 {
            format_tick(0.5 * (ymin + ymax))
        } else {
            String::new()
        };
        let mut line = format!("{tick:>width$} │", width = GUTTER - 1);
        for col in 0..w {
            let (bits, owner) = cells[row * w + col];
            let ch = char::from_u32(BRAILLE_BASE + bits as u32).expect("valid braille");
            if opts.color && bits != 0 {
                line.push_str(COLORS[owner % COLORS.len()]);
                line.push(ch);
                line.push_str("\x1b[0m");
            } else {
                line.push(if bits == 0 { ' ' } else { ch });
            }
        }
        out.push(line);
    }
    out.push(format!("{:>width$} └{}", "", "─".repeat(w), width = GUTTER - 1));
    let (lo, mid, hi) = (format_tick(xmin), format_tick(0.5 * (xmin + xmax)), format_tick(xmax));
    let mut axis = vec![' '; w];
    let mut put = |text: &str, start: usize| {
        for (i, c) in text.chars().enumerate() {
            if start + i < w {
                axis[start + i] = c;
            }
        }
    };
    put(&lo, 0);
    put(&mid, (w.saturating_sub(mid.chars().count())) / 2);
    put(&hi, w.saturating_sub(hi.chars().count()));
    out.push(format!("{:>width$}  {}", "", axis.iter().collect::<String>(), width = GUTTER - 1));
    if let Some(l) = &opts.x_label {
        out.push(format!("{:^width$}", l, width = GUTTER + w + 1));
    }
    if series.len() > 1 || series.first().is_some_and(|s| !s.label.is_empty()) {
        for (i, s) in series.iter().enumerate() {
            let mark = if opts.color { format!("{}━\x1b[0m", COLORS[i % COLORS.len()]) } else { format!("{}", i + 1) };
            out.push(format!("{:>width$} {mark} {}", "", s.label, width = GUTTER - 1));
        }
    }
    out.join("\n")
}

/// Plots a single curve `y(x)` with default styling at the given size in character cells.
pub fn generate_terminal_plot(x: &[f64], y: &[f64], width: usize, height: usize) -> String {
    let series = [PlotSeries { label: "", x, y }];
    render_terminal_plot(&series, &PlotOptions { width, height, ..Default::default() })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine() -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..200).map(|i| i as f64 * 0.05).collect();
        (x.clone(), x.iter().map(|v| v.sin()).collect())
    }

    #[test]
    fn layout_and_labels() {
        let (x, y) = sine();
        let text = generate_terminal_plot(&x, &y, 50, 12);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 12 + 2); // rows + axis line + tick labels
        assert!(lines[0].trim_start().starts_with('1'), "{}", lines[0]);
        assert!(lines[11].trim_start().starts_with('-'));
        assert!(lines[13].contains('0') && lines[13].contains("9.95"));
        assert!(text.chars().any(|c| ('\u{2801}'..='\u{28FF}').contains(&c)));
    }

    #[test]
    fn curve_occupies_expected_rows() {
        let x: Vec<f64> = (0..=40).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| v).collect();
        let text = generate_terminal_plot(&x, &y, 20, 8);
        let lines: Vec<&str> = text.lines().collect();
        // A rising line starts in the bottom-left and ends in the top-right.
        let plot_part = |l: &str| l.split('│').nth(1).unwrap().to_string();
        assert_ne!(plot_part(lines[7]).chars().next().unwrap(), ' ');
        assert_ne!(plot_part(lines[0]).chars().last().unwrap(), ' ');
        assert_eq!(plot_part(lines[0]).chars().next().unwrap(), ' ');
    }

    #[test]
    fn degenerate_data_title_legend_and_colour() {
        let flat = [2.0; 5];
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let a = PlotSeries { label: "flat", x: &x, y: &flat };
        let b = PlotSeries { label: "ramp", x: &x, y: &x };
        let opts = PlotOptions { title: Some("T".into()), x_label: Some("time".into()), color: true, ..Default::default() };
        let text = render_terminal_plot(&[a, b], &opts);
        assert!(text.contains("flat") && text.contains("ramp") && text.contains("time"));
        assert!(text.contains("\x1b[36m") && text.contains("\x1b[33m"));
        assert!(text.lines().next().unwrap().contains('T'));
        // Empty / non-finite input does not panic.
        let nan = [f64::NAN, f64::INFINITY];
        assert!(!generate_terminal_plot(&nan, &nan, 10, 4).is_empty());
        assert!(!generate_terminal_plot(&[], &[], 10, 4).is_empty());
    }

    #[test]
    fn scatter_draws_isolated_points() {
        let opts = PlotOptions { width: 10, height: 4, scatter: true, ..Default::default() };
        let s = PlotSeries { label: "", x: &[0.0, 10.0], y: &[0.0, 10.0] };
        let text = render_terminal_plot(&[s], &opts);
        let dots: usize = text.chars().filter(|c| ('\u{2801}'..='\u{28FF}').contains(c)).count();
        assert_eq!(dots, 2);
    }
}
