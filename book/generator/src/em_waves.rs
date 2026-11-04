//! Figures and tables for the chapter on transmission lines, propagation and computational
//! electromagnetics.

use crate::raster::{Colours, Mark, Panel, panels};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_core::constants::em::C0;
use lib_jsl_electromagnetics::algorithms::{Fdtd1D, FdtdTmz, WireDipole, gaussian_pulse, planar_near_to_far, scan_taper, spherical_near_to_far};
use lib_jsl_electromagnetics::antenna_models::HertzianDipole;
use lib_jsl_electromagnetics::conventions::{C64, Medium, TimeConvention, power_to_db};
use lib_jsl_electromagnetics::coordinates::{CoordinateSystem, Position};
use lib_jsl_electromagnetics::networks::s_parameters::{abcd_to_s, cascade, is_lossless, is_reciprocal, s_to_abcd, s_to_t, s_to_z, t_to_s, z_to_s};
use lib_jsl_electromagnetics::networks::smith_chart::{circle_points, impedance_locus, impedance_to_gamma, reactance_circle, resistance_circle};
use lib_jsl_electromagnetics::networks::transmission_lines::{TransmissionLine, quarter_wave_transformer, reflection_coefficient, single_stub_match, vswr};
use lib_jsl_electromagnetics::propagation::atmosphere::{cloud_specific_attenuation_coefficient, oxygen_specific_attenuation, water_vapour_specific_attenuation};
use lib_jsl_electromagnetics::propagation::multipath::{MultipathChannel, ReflectionPolarization, ground_permittivity, two_ray_asymptotic_power, two_ray_crossover_distance, two_ray_factor};
use lib_jsl_electromagnetics::propagation::path_loss::{HataEnvironment, free_space_path_loss_db, okumura_hata_db};
use lib_jsl_electromagnetics::propagation::radar_equation::Radar;
use ndarray::Array2;
use std::f64::consts::{FRAC_PI_2, PI, TAU};

const CHAPTER: &str = "em-waves";
const CONV: TimeConvention = TimeConvention::PlusJ;

pub fn generate() {
    line_figures();
    propagation_figures();
    near_far_figures();
    mom_figures();
    fdtd_figures();
}

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

// ------------------------------------------------------------------ transmission lines

fn line_figures() {
    let f = 1e9;
    let line = TransmissionLine::lossless(50.0, 1.0).unwrap();
    let lambda = line.wavelength(f);
    let mut plot = Plot::new("Standing waves on a 50 Ω line: |V| along the line, from the load", "distance from the load (wavelengths)", "|V| ÷ |V⁺|");
    for (k, (name, load)) in [("matched (50 Ω)", c(50.0, 0.0)), ("100 Ω", c(100.0, 0.0)), ("25 + 50j Ω", c(25.0, 50.0)), ("short circuit", c(0.0, 0.0))].into_iter().enumerate() {
        let d: Vec<f64> = (0..=400).map(|i| i as f64 / 400.0 * lambda).collect();
        let v = line.standing_wave(load, &d, f);
        plot.series.push(Series::new(&format!("{name}, VSWR {}", fmt_vswr(vswr(reflection_coefficient(load, c(50.0, 0.0))))), d.iter().zip(&v).map(|(x, y)| (x / lambda, *y)).collect(), Style::Line, k));
    }
    plot.y_range = Some((0.0, 2.6));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "standing_waves.svg", &plot.render());
    // A Smith chart with the locus of a load and the stub-match solution.
    let mut plot = Plot::new("Smith chart: 25 + 50j Ω moving towards the generator", "Re Γ", "Im Γ");
    let grid = |pts: Vec<C64>| pts.into_iter().map(|p| (p.re, p.im)).collect::<Vec<_>>();
    for r in [0.0, 0.2, 0.5, 1.0, 2.0, 5.0] {
        let (cen, rad) = resistance_circle(r);
        let mut pts = grid(circle_points(cen, rad, 400));
        pts.push(pts[0]);
        plot.series.push(Series::new(if r == 0.0 { "constant r and x" } else { "" }, pts, Style::Line, 7));
    }
    for x in [-5.0, -2.0, -1.0, -0.5, -0.2, 0.2, 0.5, 1.0, 2.0, 5.0] {
        let (cen, rad) = reactance_circle(x);
        let mut pts: Vec<(f64, f64)> = grid(circle_points(cen, rad, 2000));
        // Order the clipped arc by angle about the centre so it draws as one curve.
        pts.sort_by(|a, b| (a.1 - cen.im).atan2(a.0 - cen.re).total_cmp(&(b.1 - cen.im).atan2(b.0 - cen.re)));
        plot.series.push(Series::new("", pts, Style::Line, 7));
    }
    let z_load = c(0.5, 1.0); // 25 + 50j on 50 Ω
    let distances: Vec<f64> = (0..=200).map(|i| 0.5 * i as f64 / 200.0).collect();
    plot.series.push(Series::new("Γ along the line (half a wavelength)", grid(impedance_locus(z_load, &distances)), Style::Line, 0));
    let stub = single_stub_match(c(25.0, 50.0), 50.0).unwrap();
    let at = impedance_locus(z_load, &[stub[0].0])[0];
    plot.series.push(Series::new(&format!("stub position {:.3}λ", stub[0].0), vec![(at.re, at.im)], Style::Markers, 1));
    let g = impedance_to_gamma(z_load);
    plot.series.push(Series::new("load", vec![(g.re, g.im)], Style::Markers, 2));
    // Equal scales on both axes (the plot area is 1.81 times wider than tall), with room on the
    // left for the legend.
    plot.x_range = Some((-2.6, 1.02));
    plot.y_range = Some((-1.0, 1.0));
    plot.legend = Legend::TopLeft;
    write(CHAPTER, "smith_chart.svg", &plot.render());
    // Stub matching check.
    let mut rows = Vec::new();
    for (k, (d, s)) in stub.iter().enumerate() {
        // Admittance after the line section plus the shorted stub.
        let zin = line.input_impedance(c(25.0, 50.0), d * lambda, f);
        let y_stub = c(0.0, -1.0 / (TAU * s).tan()) / 50.0;
        let ytot = 1.0 / zin + y_stub;
        rows.push(vec![format!("{}", k + 1), format!("{d:.4}"), format!("{s:.4}"), format!("{:.2e}", reflection_coefficient(1.0 / ytot, c(50.0, 0.0)).norm())]);
    }
    write(CHAPTER, "stub_table.md", &md_table(&["solution", "stub position from the load (λ)", "short-circuited stub length (λ)", "‖Γ‖ after matching"], &rows));
    // Quarter-wave transformers against frequency.
    let mut plot = Plot::new("Quarter-wave transformers: reflection against frequency", "f ÷ f₀", "|Γ|");
    for (k, rl) in [100.0, 200.0, 500.0].into_iter().enumerate() {
        let zt = quarter_wave_transformer(50.0, rl);
        let tl = TransmissionLine::lossless(zt, 1.0).unwrap();
        let quarter = tl.wavelength(f) / 4.0;
        let pts: Vec<(f64, f64)> = (0..=200).map(|i| 0.2 + 1.6 * i as f64 / 200.0).map(|r| (r, reflection_coefficient(tl.input_impedance(c(rl, 0.0), quarter, r * f), c(50.0, 0.0)).norm())).collect();
        plot.series.push(Series::new(&format!("50 → {rl} Ω (Z_T = {zt:.1} Ω)"), pts, Style::Line, k));
    }
    plot.legend = Legend::TopRight;
    write(CHAPTER, "quarter_wave.svg", &plot.render());
    // Two-port parameter round trips and cascades.
    let s_line = abcd_to_s(&line.abcd(lambda / 8.0, f), 50.0).unwrap();
    let s_pair = cascade(&s_line, &s_line, 50.0).unwrap();
    let s_quarter = abcd_to_s(&line.abcd(lambda / 4.0, f), 50.0).unwrap();
    let diff = |a: &[[C64; 2]; 2], b: &[[C64; 2]; 2]| (0..2).flat_map(|i| (0..2).map(move |j| (i, j))).map(|(i, j)| (a[i][j] - b[i][j]).norm()).fold(0.0, f64::max);
    let lossy = TransmissionLine::distributed(2.0, 250e-9, 1e-5, 100e-12).unwrap();
    let s_lossy = abcd_to_s(&lossy.abcd(0.3, f), 50.0).unwrap();
    let rows = vec![
        vec!["two λ/8 lines in cascade vs one λ/4 line".to_string(), format!("{:.1e}", diff(&s_pair, &s_quarter))],
        vec!["S → Z → S (lossy line, 30 cm)".to_string(), format!("{:.1e}", diff(&z_to_s(&s_to_z(&s_lossy, 50.0).unwrap(), 50.0).unwrap(), &s_lossy))],
        vec!["S → ABCD → S".to_string(), format!("{:.1e}", diff(&abcd_to_s(&s_to_abcd(&s_lossy, 50.0).unwrap(), 50.0).unwrap(), &s_lossy))],
        vec!["S → T → S".to_string(), format!("{:.1e}", diff(&t_to_s(&s_to_t(&s_lossy).unwrap()).unwrap(), &s_lossy))],
        vec!["lossless line: SᴴS = I".to_string(), format!("{}", is_lossless(&s_quarter, 1e-12))],
        vec!["lossy line: SᴴS = I".to_string(), format!("{}", is_lossless(&s_lossy, 1e-6))],
        vec!["lossy line: S = Sᵀ (reciprocal)".to_string(), format!("{}", is_reciprocal(&s_lossy, 1e-12))],
    ];
    write(CHAPTER, "two_port_table.md", &md_table(&["check", "largest difference or result"], &rows));
}

fn fmt_vswr(v: f64) -> String {
    if v.is_infinite() { "∞".to_string() } else { format!("{v:.2}") }
}

// ------------------------------------------------------------------ propagation

fn propagation_figures() {
    // Two-ray model, 900 MHz, 30 m and 1.5 m antennas.
    let (f, ht, hr) = (900e6, 30.0, 1.5);
    let ground = (ground_permittivity(15.0, 0.005, f), ReflectionPolarization::Perpendicular);
    let ds: Vec<f64> = (0..=600).map(|i| 10f64.powf(1.0 + 3.7 * i as f64 / 600.0)).collect();
    let fspl = |d: f64| -free_space_path_loss_db(d, f);
    let mut plot = Plot::new("Two-ray propagation at 900 MHz, antennas 30 m and 1.5 m high", "distance (m)", "received power re transmitted (dB, isotropic antennas)");
    plot.series.push(Series::new("free space", ds.iter().map(|&d| (d, fspl(d))).collect(), Style::Dashed, 7));
    plot.series.push(Series::new("two-ray, perfect reflector", ds.iter().map(|&d| (d, fspl(d) + power_to_db(two_ray_factor(d, ht, hr, f, None).unwrap().norm_sqr().max(1e-30)))).collect(), Style::Line, 0));
    plot.series.push(Series::new("two-ray, average ground", ds.iter().map(|&d| (d, fspl(d) + power_to_db(two_ray_factor(d, ht, hr, f, Some(ground)).unwrap().norm_sqr().max(1e-30)))).collect(), Style::Line, 1));
    plot.series.push(Series::new("h_t² h_r² / d⁴", ds.iter().filter(|&&d| d > 300.0).map(|&d| (d, power_to_db(two_ray_asymptotic_power(1.0, 1.0, 1.0, ht, hr, d)))).collect(), Style::Dashed, 2));
    plot.log_x = true;
    plot.y_range = Some((-150.0, -40.0));
    plot.legend = Legend::BottomLeft;
    write(CHAPTER, "two_ray.svg", &plot.render());
    write(CHAPTER, "two_ray_facts.md", &format!("The crossover distance $4\\pi h_th_r/\\lambda$ is {:.0} m.", two_ray_crossover_distance(ht, hr, f)));
    // Okumura-Hata.
    let mut plot = Plot::new("Okumura-Hata median path loss, 900 MHz, base 30 m, mobile 1.5 m", "distance (km)", "path loss (dB)");
    let dk: Vec<f64> = (0..=100).map(|i| 10f64.powf(1.3010 * i as f64 / 100.0)).collect();
    for (k, (name, env)) in [("urban, large city", HataEnvironment::LargeCity), ("urban, small/medium city", HataEnvironment::Urban), ("suburban", HataEnvironment::Suburban), ("open area", HataEnvironment::OpenArea)].into_iter().enumerate() {
        plot.series.push(Series::new(name, dk.iter().map(|&d| (d, okumura_hata_db(900.0, 30.0, 1.5, d, env).unwrap())).collect(), Style::Line, k));
    }
    plot.series.push(Series::new("free space", dk.iter().map(|&d| (d, free_space_path_loss_db(1e3 * d, f))).collect(), Style::Dashed, 7));
    plot.log_x = true;
    plot.legend = Legend::BottomRight;
    write(CHAPTER, "hata.svg", &plot.render());
    write(CHAPTER, "hata_facts.md", &format!("At 900 MHz with a 30 m base station, a 1.5 m mobile and 1 km in a small or medium city, the model gives {:.1} dB, the textbook value of 126.4 dB.", okumura_hata_db(900.0, 30.0, 1.5, 1.0, HataEnvironment::Urban).unwrap()));
    // Atmospheric specific attenuation.
    let fs: Vec<f64> = (0..=700).map(|i| 1.0 + 348.5 * i as f64 / 700.0).collect();
    let mut plot = Plot::new("Specific attenuation at sea level (simplified ITU-R models)", "frequency (GHz)", "dB/km");
    plot.series.push(Series::new("oxygen (below 57 GHz)", fs.iter().filter(|&&g| g < 57.0).map(|&g| (g, oxygen_specific_attenuation(g).unwrap())).collect(), Style::Line, 0));
    plot.series.push(Series::new("water vapour, 7.5 g/m³", fs.iter().map(|&g| (g, water_vapour_specific_attenuation(g, 7.5).unwrap())).collect(), Style::Line, 1));
    plot.series.push(Series::new("cloud, 0.5 g/m³ liquid water at 0 °C", fs.iter().filter(|&&g| g <= 200.0).map(|&g| (g, 0.5 * cloud_specific_attenuation_coefficient(g, 273.15).unwrap())).collect(), Style::Line, 2));
    plot.log_x = true;
    plot.log_y = true;
    plot.y_range = Some((1e-3, 100.0));
    plot.legend = Legend::TopLeft;
    write(CHAPTER, "atmosphere.svg", &plot.render());
    // Radar equation example.
    let radar = Radar { transmit_power: 1e6, gain: 10f64.powf(3.5), frequency: 3e9, bandwidth: 1e6, noise_figure: 10f64.powf(0.3), losses: 10f64.powf(0.5), noise_temperature: 290.0 };
    let mut rows = Vec::new();
    for (name, rcs) in [("small aircraft", 1.0), ("large aircraft", 100.0), ("small drone", 0.01)] {
        let r1 = radar.maximum_range(rcs, 10f64.powf(1.3), 1.0).unwrap();
        let r10 = radar.maximum_range(rcs, 10f64.powf(1.3), 10.0).unwrap();
        rows.push(vec![name.to_string(), format!("{rcs}"), format!("{:.0}", r1 / 1e3), format!("{:.0}", r10 / 1e3)]);
    }
    write(CHAPTER, "radar_table.md", &md_table(&["target", "RCS (m²)", "range for 13 dB SNR, one pulse (km)", "with 10 pulses integrated coherently (km)"], &rows));
    // A multipath channel's frequency response.
    let ch = MultipathChannel::new(vec![0.0, 0.4e-6, 1.1e-6, 2.3e-6], vec![c(1.0, 0.0), c(0.0, 0.6), c(-0.4, 0.1), c(0.15, -0.1)]).unwrap();
    let (mean, rms) = ch.delay_statistics();
    let mut plot = Plot::new(&format!("A four-path channel: mean delay {:.2} µs, RMS delay spread {:.2} µs", 1e6 * mean, 1e6 * rms), "frequency offset (MHz)", "|H(f)| (dB)");
    plot.series.push(Series::new("frequency response", (0..=800).map(|i| -5.0 + 10.0 * i as f64 / 800.0).map(|m| (m, power_to_db(ch.frequency_response(m * 1e6).norm_sqr()))).collect(), Style::Line, 0));
    plot.legend = Legend::BottomRight;
    write(CHAPTER, "multipath.svg", &plot.render());
    write(CHAPTER, "multipath_facts.md", &format!("Its coherence bandwidth, about $1/(5\\sigma_\\tau)$, is {:.0} kHz: frequencies further apart fade independently.", ch.coherence_bandwidth() / 1e3));
}

// ------------------------------------------------------------------ spherical waves and near-to-far transforms

/// The exact field (Cartesian components) at `q` of x-directed Hertzian dipoles at `sources`
/// with complex weights.
fn x_dipoles(q: [f64; 3], sources: &[([f64; 3], C64)], f: f64) -> [C64; 3] {
    let dipole = HertzianDipole::new(0.01);
    let mut e = [c(0.0, 0.0); 3];
    for &(p, w) in sources {
        // In the dipole's frame its axis is local z: (local x, y, z) = (global y, z, x).
        let local = Position::cartesian(q[1] - p[1], q[2] - p[2], q[0] - p[0]);
        let (es, _) = dipole.near_field(&local, f, CONV);
        let frame = local.frame(CoordinateSystem::Spherical);
        let el: [C64; 3] = std::array::from_fn(|k| (0..3).map(|a| es[a] * frame[a][k]).sum());
        let g = [el[2], el[0], el[1]];
        for k in 0..3 {
            e[k] += w * g[k];
        }
    }
    e
}

/// The exact far field `[F_θ, F_φ]` of the same sources.
fn x_dipoles_far(theta: f64, phi: f64, sources: &[([f64; 3], C64)], f: f64) -> [C64; 2] {
    let k = TAU * f / C0;
    let eta = Medium::VACUUM.impedance(CONV).re;
    let dir = Position::spherical(1.0, theta, phi);
    let r = dir.to_cartesian();
    let frame = dir.frame(CoordinateSystem::Spherical);
    // A z-dipole's E_θ = jηkIl sin θ/(4π) is −jηkIl/(4π) (ẑ − r̂(r̂·ẑ)) projected on θ̂.
    let amp = c(0.0, -eta * k * 0.01 / (4.0 * PI));
    let mut out = [c(0.0, 0.0); 2];
    for &(p, w) in sources {
        let phase = C64::from_polar(1.0, k * (r[0] * p[0] + r[1] * p[1] + r[2] * p[2]));
        // E ∝ x̂ − r̂ (r̂·x̂): its θ and φ components are x̂·θ̂ and x̂·φ̂.
        out[0] += w * amp * phase * frame[1][0];
        out[1] += w * amp * phase * frame[2][0];
    }
    out
}

fn to_spherical_components(e: [C64; 3], q: &Position) -> [C64; 3] {
    let frame = q.frame(CoordinateSystem::Spherical);
    std::array::from_fn(|a| (0..3).map(|k| e[k] * frame[a][k]).sum())
}

fn near_far_figures() {
    let f = 3e8; // λ = 1 m
    // A 2 × 2 array of x-dipoles, 0.6 λ apart, phased to tilt the beam 20° in the xz plane.
    let k = TAU;
    let tilt = 20f64.to_radians();
    let sources: Vec<([f64; 3], C64)> = [(-0.3, -0.3), (0.3, -0.3), (-0.3, 0.3), (0.3, 0.3)].iter().map(|&(x, y)| ([x, y, 0.0], C64::from_polar(1.0, -k * x * tilt.sin()))).collect();
    // Spherical near-to-far: error against the maximum degree.
    let r0 = 1.5;
    let mut pts = Vec::new();
    let cut: Vec<f64> = (1..=89).map(|i| PI * i as f64 / 90.0).collect();
    let peak = cut.iter().map(|&t| x_dipoles_far(t, 0.0, &sources, f)[0].norm()).fold(0.0, f64::max);
    for n in [2u32, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14] {
        let swe = spherical_near_to_far(|t, p| {
            let q = Position::spherical(r0, t, p);
            let e = to_spherical_components(x_dipoles(q.to_cartesian(), &sources, f), &q);
            [e[1], e[2]]
        }, r0, f, n, CONV).unwrap();
        let err = cut.iter().flat_map(|&t| [0.0, FRAC_PI_2].map(|p| (t, p))).map(|(t, p)| {
            let (a, b) = (swe.far_field(t, p, CONV), x_dipoles_far(t, p, &sources, f));
            ((a[0] - b[0]).norm_sqr() + (a[1] - b[1]).norm_sqr()).sqrt()
        }).fold(0.0, f64::max) / peak;
        pts.push((n as f64, err.max(1e-16)));
    }
    let mut plot = Plot::new("Spherical near-to-far transform from a sphere of radius 1.5λ", "maximum degree N", "largest far-field error ÷ peak");
    plot.series.push(Series::new("2 × 2 dipole array (minimum sphere 0.42λ, kr ≈ 2.7)", pts, Style::LineMarkers, 0));
    plot.log_y = true;
    plot.legend = Legend::TopRight;
    write(CHAPTER, "spherical_nf_ff.svg", &plot.render());
    // Planar near-to-far: scan plane 1λ in front, sampled every 0.4λ over ±extent.
    let z0 = 1.0;
    let dirs: Vec<(f64, f64)> = (-80..=80).map(|d| {
        let t = (d as f64).to_radians();
        if t >= 0.0 { (t, 0.0) } else { (-t, PI) }
    }).collect();
    let exact: Vec<f64> = dirs.iter().map(|&(t, p)| x_dipoles_far(t, p, &sources, f)[0].norm()).collect();
    let pk = exact.iter().cloned().fold(0.0, f64::max);
    let mut plot = Plot::new("Planar near-to-far transform: scan 1λ in front, samples 0.4λ apart", "θ in the xz plane (degrees)", "E_θ (dB re peak)");
    plot.series.push(Series::new("exact", dirs.iter().zip(&exact).map(|(&(t, p), &v)| (if p == 0.0 { t.to_degrees() } else { -t.to_degrees() }, power_to_db((v / pk).powi(2)))).collect(), Style::Line, 7));
    let mut rows = Vec::new();
    for (ci, (name, half, taper)) in [("±4λ scan, abrupt edges", 4.0f64, false), ("±4λ scan, tapered edges", 4.0, true), ("±10λ scan, tapered edges", 10.0, true)].into_iter().enumerate() {
        let n = (2.0 * half / 0.4).round() as usize + 1;
        let xs: Vec<f64> = (0..n).map(|i| -half + 0.4 * i as f64).collect();
        let w = if taper { scan_taper(&xs, 0.3) } else { vec![1.0; n] };
        let mut ex = Array2::<C64>::zeros((n, n));
        let mut ey = Array2::<C64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let e = x_dipoles([xs[i], xs[j], z0], &sources, f);
                ex[[i, j]] = e[0] * w[i] * w[j];
                ey[[i, j]] = e[1] * w[i] * w[j];
            }
        }
        let ff = planar_near_to_far(&xs, &xs, &ex, &ey, z0, f, &dirs, CONV).unwrap();
        let got: Vec<f64> = ff.iter().map(|v| v[0].norm()).collect();
        plot.series.push(Series::new(name, dirs.iter().zip(&got).map(|(&(t, p), &v)| (if p == 0.0 { t.to_degrees() } else { -t.to_degrees() }, power_to_db((v / pk).powi(2).max(1e-8)))).collect(), Style::Line, ci));
        let err_within = |deg: f64| dirs.iter().zip(got.iter().zip(&exact)).filter(|(d, _)| d.0.to_degrees() <= deg).map(|(_, (g, e))| (g - e).abs() / pk).fold(0.0, f64::max);
        rows.push(vec![name.to_string(), format!("{:.3}", err_within(30.0)), format!("{:.3}", err_within(60.0))]);
    }
    plot.y_range = Some((-40.0, 2.0));
    plot.legend = Legend::BottomLeft;
    write(CHAPTER, "planar_nf_ff.svg", &plot.render());
    write(CHAPTER, "planar_table.md", &md_table(&["scan", "largest ‖E_θ‖ error ÷ peak, within 30°", "within 60°"], &rows));
}

// ------------------------------------------------------------------ method of moments

fn mom_figures() {
    let f = 3e8; // λ = 1 m
    let radius = 1e-3;
    let mut plot = Plot::new("Hallén moment-method currents (a = 0.001λ) against the sinusoidal assumption", "position along the wire, z ÷ L", "|I| ÷ max |I|");
    for (k, l) in [0.5, 1.0, 1.5].into_iter().enumerate() {
        let sol = WireDipole::new(l, radius).unwrap().solve(f, 81).unwrap();
        let peak = sol.currents.iter().map(|v| v.norm()).fold(0.0, f64::max);
        let mut pts: Vec<(f64, f64)> = vec![(-0.5, 0.0)];
        pts.extend(sol.nodes.iter().zip(&sol.currents).map(|(z, i)| (z / l, i.norm() / peak)));
        pts.push((0.5, 0.0));
        plot.series.push(Series::new(&format!("L = {l}λ, moment method"), pts, Style::Line, k));
        let sin: Vec<(f64, f64)> = (0..=200).map(|i| -0.5 + i as f64 / 200.0).map(|u| (u, (TAU * (l / 2.0 - (u * l).abs())).sin().abs())).collect();
        plot.series.push(Series::new(&format!("L = {l}λ, sinusoidal"), sin, Style::Dashed, k));
    }
    plot.legend = Legend::TopRight;
    plot.y_range = Some((0.0, 1.35));
    write(CHAPTER, "mom_currents.svg", &plot.render());
    // Input impedance against length.
    let ls: Vec<f64> = (0..=60).map(|i| 0.3 + 0.012 * i as f64).collect();
    let z: Vec<C64> = ls.iter().map(|&l| WireDipole::new(l, radius).unwrap().solve(f, 61).unwrap().input_impedance).collect();
    let mut plot = Plot::new("Input impedance of a centre-fed dipole (a = 0.001λ)", "length L ÷ λ", "impedance (Ω)");
    plot.series.push(Series::new("resistance R", ls.iter().zip(&z).map(|(l, v)| (*l, v.re)).collect(), Style::Line, 0));
    plot.series.push(Series::new("reactance X", ls.iter().zip(&z).map(|(l, v)| (*l, v.im)).collect(), Style::Line, 1));
    plot.y_range = Some((-600.0, 1200.0));
    plot.legend = Legend::TopLeft;
    write(CHAPTER, "mom_impedance.svg", &plot.render());
    // Convergence for the half-wave dipole, and the resonant length.
    let mut rows = Vec::new();
    for n in [11usize, 21, 41, 81, 161, 321] {
        let z = |a: f64| WireDipole::new(0.5, a).unwrap().solve(f, n).unwrap().input_impedance;
        let (z3, z4) = (z(1e-3), z(1e-4));
        let seg = 0.5 / (n + 1) as f64;
        rows.push(vec![format!("{n}"), format!("{:.1}", seg / 1e-4), format!("{:.2} + {:.2}j", z4.re, z4.im), format!("{:.1}", seg / 1e-3), format!("{:.2} + {:.2}j", z3.re, z3.im)]);
    }
    write(CHAPTER, "mom_convergence.md", &md_table(&["interior nodes", "segment ÷ radius (a = 10⁻⁴λ)", "Z (Ω), a = 10⁻⁴λ", "segment ÷ radius (a = 10⁻³λ)", "Z (Ω), a = 10⁻³λ"], &rows));
    let (mut lo, mut hi) = (0.40, 0.50);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if WireDipole::new(mid, radius).unwrap().solve(f, 61).unwrap().input_impedance.im < 0.0 { lo = mid } else { hi = mid }
    }
    let zr = WireDipole::new(lo, radius).unwrap().solve(f, 61).unwrap().input_impedance;
    write(CHAPTER, "mom_facts.md", &format!("The reactance vanishes at L = {lo:.4}λ, where R = {:.1} Ω: a dipole must be cut about {:.0}% shorter than half a wavelength to resonate.", zr.re, 100.0 * (0.5 - lo) / 0.5));
}

// ------------------------------------------------------------------ FDTD

fn fdtd_figures() {
    // 1D: reflection from a dielectric half-space, measured with a Gaussian pulse.
    let mut rows = Vec::new();
    for eps in [2.0, 4.0, 9.0] {
        let mut measured = Vec::new();
        for s in [1.0, 0.5] {
            let cells = 1200;
            let mut sim = Fdtd1D::new(cells, 1e-3, s).unwrap();
            sim.set_material(700, cells, eps, 0.0, 1.0);
            // Record the field at a probe between the source and the interface.
            let (src, probe) = (100, 400);
            let steps = (2600.0 / s) as usize;
            let mut trace = Vec::with_capacity(steps);
            for n in 0..steps {
                sim.step(Some((src, gaussian_pulse(n, 80.0 / s, 20.0 / s))));
                trace.push(sim.ez[probe]);
            }
            // Incident peak first, reflected peak later (opposite sign for eps > 1).
            let split = ((700.0 - 100.0 + 300.0) / s) as usize;
            let inc = trace[..split].iter().cloned().fold(0.0, f64::max);
            let refl = trace[split..].iter().cloned().fold(0.0f64, |m, v| if v.abs() > m.abs() { v } else { m });
            measured.push(refl / inc);
        }
        let theory = (1.0 - eps.sqrt()) / (1.0 + eps.sqrt());
        rows.push(vec![format!("{eps}"), format!("{theory:.4}"), format!("{:.4}", measured[0]), format!("{:.4}", measured[1])]);
    }
    write(CHAPTER, "fdtd_reflection.md", &md_table(&["ε_r", "Fresnel Γ = (1 − √ε_r)/(1 + √ε_r)", "FDTD, Courant number 1", "FDTD, Courant number 0.5"], &rows));
    // 2D: a pulse scattering from a PEC cylinder, with a PML.
    let (nx, ny, pml) = (200usize, 160usize, 12usize);
    let mut sim = FdtdTmz::new(nx, ny, 1e-3, 0.7, pml).unwrap();
    let (cx, cy, rad) = (120.0, 80.0, 14.0);
    for j in 0..ny {
        for i in 0..nx {
            if ((i as f64 - cx).powi(2) + (j as f64 - cy).powi(2)).sqrt() <= rad {
                sim.set_pec(i, i + 1, j, j + 1);
            }
        }
    }
    let mut frames = Vec::new();
    let mut energy = Vec::new();
    for n in 0..900 {
        sim.step(Some((50, 80, gaussian_pulse(n, 40.0, 12.0))));
        if [110, 170, 240].contains(&n) {
            frames.push((n, Array2::from_shape_fn((ny, nx), |(j, i)| sim.ez(i, ny - 1 - j))));
        }
        if n % 5 == 0 {
            energy.push((n as f64, sim.interior_energy(pml)));
        }
    }
    let ps: Vec<Panel> = frames.into_iter().map(|(n, a)| {
        let m = a.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-12);
        let scaled = a.mapv(|v| (v / m).clamp(-1.0, 1.0));
        let mut p = Panel::new(&format!("step {n}"), scaled, Colours::Ramp(-1.0, 1.0));
        p.marks.push(Mark::Circle { x: cx, y: (ny - 1) as f64 - cy, radius: rad, colour: "#ffffff" });
        p
    }).collect();
    write(CHAPTER, "fdtd_cylinder.svg", &panels("E_z of a pulse scattering from a conducting cylinder (2D FDTD with a PML)", &ps, 3, 240.0));
    let e0 = energy.iter().map(|p| p.1).fold(0.0, f64::max);
    let mut plot = Plot::new("Energy left inside the computational domain", "time step", "energy ÷ its peak");
    plot.series.push(Series::new("interior energy (PML around the edges)", energy.iter().map(|&(n, e)| (n, (e / e0).max(1e-16))).collect(), Style::Line, 0));
    plot.log_y = true;
    plot.legend = Legend::TopRight;
    write(CHAPTER, "fdtd_energy.svg", &plot.render());
    // Numerical dispersion in 1D: measure the phase velocity of a sinusoid between two probes,
    // against the Yee dispersion relation sin(ωΔt/2) = S sin(kΔx/2).
    let mut rows = Vec::new();
    for s in [1.0, 0.5] {
        for ppw in [10.0, 20.0, 40.0] {
            let cells = 2000;
            let mut sim = Fdtd1D::new(cells, 1e-3, s).unwrap();
            let w_dt = TAU / ppw * s; // ω Δt for a free-space wavelength of ppw cells
            let (p1, p2) = (600usize, 600 + 5 * ppw as usize);
            let (mut a1, mut a2) = (c(0.0, 0.0), c(0.0, 0.0));
            let steps = (2600.0 / s) as usize;
            for n in 0..steps {
                let ramp = (n as f64 / (100.0 / s)).min(1.0);
                sim.step(Some((100, ramp * (w_dt * n as f64).sin())));
                if n > steps / 2 {
                    let ph = C64::from_polar(1.0, -w_dt * n as f64);
                    a1 += sim.ez[p1] * ph;
                    a2 += sim.ez[p2] * ph;
                }
            }
            // Phase lag between the probes, unwrapped by the expected number of wavelengths.
            // The free-space lag is exactly 5 wavelengths; the numerical one differs slightly.
            let expected = TAU * (p2 - p1) as f64 / ppw;
            let residual = ((a1 / a2).arg() - expected + PI).rem_euclid(TAU) - PI;
            let lag = expected + residual;
            let k_num_dx = lag / (p2 - p1) as f64;
            let measured = (TAU / ppw) / k_num_dx;
            let theory = (w_dt / 2.0).sin() / s;
            let predicted = (TAU / ppw) / (2.0 * theory.asin());
            rows.push(vec![format!("{s}"), format!("{ppw}"), format!("{:.2e}", (1.0 - predicted).abs()), format!("{:.2e}", (1.0 - measured).abs())]);
        }
    }
    write(CHAPTER, "fdtd_dispersion.md", &md_table(&["Courant number S", "cells per wavelength", "phase velocity error ‖1 − v/c‖, Yee dispersion relation", "measured"], &rows));
}
