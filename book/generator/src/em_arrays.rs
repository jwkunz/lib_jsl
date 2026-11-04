//! Figures and tables for the chapter on electromagnetic fields, antennas, arrays and beamforming.

use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_core::constants::em::{C0, MU_0};
use lib_jsl_electromagnetics::algorithms::{bartlett_spectrum, capon_spectrum, cfm_for_direction, esprit_ula, lcmv_beamformer, max_snr_beamformer, music_spectrum, mvdr_beamformer, response, sinr, spectrum_peaks};
use lib_jsl_electromagnetics::antenna_models::{Antenna, Dipole, Dish, HalfWaveDipole, HertzianDipole, MagneticLoop, PatchAntenna};
use lib_jsl_electromagnetics::arrays::SensorArray;
use lib_jsl_electromagnetics::conventions::{C64, Medium, TimeConvention, amplitude_to_db, power_to_db};
use lib_jsl_electromagnetics::coordinates::{PolarizationBasis, RigidTransform};
use lib_jsl_electromagnetics::fields::Domain;
use lib_jsl_electromagnetics::fields::physics::plane_wave;
use lib_jsl_electromagnetics::fields::Axis;
use lib_jsl_electromagnetics::signal_models::{BrightnessScene, Coherency, PointSource, Scene, SkyPolarization, sample_covariance, synthesize_snapshots};
use ndarray::{Array1, Array2};
use std::f64::consts::{FRAC_PI_2, PI, TAU};
use std::sync::Arc;

const CHAPTER: &str = "em-arrays";
const CONV: TimeConvention = TimeConvention::PlusJ;

pub fn generate() {
    field_figures();
    antenna_figures();
    beamforming_figures();
    doa_figures();
}

// ------------------------------------------------------------------ fields

fn field_figures() {
    // Quadrature on the far-field sphere: the radiated power of a 3λ dipole.
    let f = 3e8;
    let dipole = Dipole::new(3.0);
    let eta = Medium::VACUUM.impedance(CONV).re;
    let intensity = |p: &lib_jsl_electromagnetics::coordinates::Position| {
        let [a, b] = dipole.far_field(p.coords[1], p.coords[2], f, CONV);
        (a.norm_sqr() + b.norm_sqr()) / (2.0 * eta)
    };
    let exact = Domain::far_field_gauss(400, 4).unwrap().integrate_fn(intensity);
    let ns: Vec<usize> = vec![6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128];
    let mut plot = Plot::new("Radiated power of a 3λ dipole by quadrature over the sphere", "samples in θ", "relative error");
    for (k, (name, gauss)) in [("Gauss-Legendre in cos θ", true), ("uniform in θ", false)].into_iter().enumerate() {
        let pts: Vec<(f64, f64)> = ns
            .iter()
            .map(|&n| {
                let d = if gauss { Domain::far_field_gauss(n, 4).unwrap() } else { Domain::far_field_uniform(n, 4).unwrap() };
                (n as f64, ((d.integrate_fn(intensity) - exact) / exact).abs().max(1e-16))
            })
            .collect();
        plot.series.push(Series::new(name, pts, Style::LineMarkers, k));
    }
    plot.log_x = true;
    plot.log_y = true;
    plot.legend = Legend::BottomLeft;
    write(CHAPTER, "sphere_quadrature.svg", &plot.render());
    // Fourth-order derivatives: Faraday's law for a plane wave in a half-wavelength cube.
    let mut rows = Vec::new();
    for n in [6usize, 11, 21, 41] {
        let axis = || Axis::uniform(-0.25, 0.25, n).unwrap();
        let cube = Arc::new(Domain::cartesian_box(axis(), axis(), axis()).unwrap());
        let (e, h) = plane_wave(cube.clone(), [1.0, 1.0, 0.0], [C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0)], C64::new(1.0, 0.0), f, &Medium::VACUUM, CONV).unwrap();
        let curl = e.curl().unwrap().to_basis(PolarizationBasis::CARTESIAN).unwrap();
        let faraday = h.scaled(-CONV.time_derivative(TAU * f) * MU_0);
        let err = (0..cube.len()).map(|i| (0..3).map(|k| (curl.vector(i)[k] - faraday.vector(i)[k]).norm_sqr()).sum::<f64>().sqrt()).fold(0.0, f64::max) / (TAU * f / C0);
        let div = e.divergence().unwrap().values().iter().map(|v| v.norm()).fold(0.0, f64::max);
        let spacing = 0.5 / (n - 1) as f64;
        rows.push(vec![format!("{n}³"), format!("{:.0}", 1.0 / spacing), format!("{err:.1e}"), String::new(), format!("{div:.0e}")]);
    }
    // Simplify the order column: halving the spacing divides the error by 2^order.
    for k in 1..rows.len() {
        let (a, b): (f64, f64) = (rows[k - 1][2].parse().unwrap(), rows[k][2].parse().unwrap());
        rows[k][3] = format!("{:.1}", (a / b).log2());
    }
    write(CHAPTER, "faraday_table.md", &md_table(&["samples", "samples per wavelength", "max ‖∇×E + jωμ₀H‖ ÷ k‖E‖", "observed order", "max ‖∇·E‖ (V/m²)"], &rows));
}

// ------------------------------------------------------------------ antennas

fn antenna_figures() {
    let f = 3e8; // λ = 1 m
    let mut plot = Plot::new("Thin-wire dipole patterns (E-plane)", "θ (degrees)", "relative power (dB)");
    for (k, l) in [0.02, 0.5, 1.0, 1.5].into_iter().enumerate() {
        let d = Dipole::new(l);
        let peak = (0..=1800).map(|i| d.radiation_intensity(PI * i as f64 / 1800.0, 0.0, f)).fold(0.0, f64::max);
        let pts: Vec<(f64, f64)> = (1..360).map(|i| PI * i as f64 / 360.0).map(|t| (t.to_degrees(), power_to_db((d.radiation_intensity(t, 0.0, f) / peak).max(1e-5)))).collect();
        plot.series.push(Series::new(&format!("L = {l}λ"), pts, Style::Line, k));
    }
    plot.y_range = Some((-40.0, 3.0));
    plot.legend = Legend::BottomRight;
    write(CHAPTER, "dipole_patterns.svg", &plot.render());
    // Dish patterns against the normalised angle u = ka sin θ.
    let diameter = 10.0;
    let ka = PI * diameter;
    let mut plot = Plot::new("Circular aperture, D = 10λ: taper (1 − ρ²/a²)^p", "ka sin θ", "relative power (dB)");
    let mut rows = Vec::new();
    for (k, p) in [0u32, 1, 2].into_iter().enumerate() {
        let dish = Dish::with_taper(diameter, p);
        let peak = dish.radiation_intensity(0.0, 0.0, f);
        let pts: Vec<(f64, f64)> = (0..=1200).map(|i| (i as f64 / 1200.0 * 20.0 / ka).asin()).map(|t| (ka * t.sin(), power_to_db((dish.radiation_intensity(t, 0.0, f) / peak).max(1e-8)))).collect();
        // First sidelobe: the largest value after the first local minimum.
        let vals: Vec<f64> = pts.iter().map(|p| p.1).collect();
        let first_min = (1..vals.len() - 1).find(|&i| vals[i] < vals[i - 1] && vals[i] <= vals[i + 1]).unwrap();
        let sidelobe = vals[first_min..].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        rows.push(vec![format!("{p}"), format!("{:.3}", dish.aperture_efficiency()), format!("{:.1}", power_to_db(dish.peak_directivity(f))), format!("{:.2}", dish.half_power_beamwidth(f).to_degrees()), format!("{sidelobe:.1}")]);
        plot.series.push(Series::new(&format!("p = {p}"), pts, Style::Line, k));
    }
    plot.y_range = Some((-60.0, 3.0));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "dish_patterns.svg", &plot.render());
    write(CHAPTER, "dish_table.md", &md_table(&["taper p", "aperture efficiency (2p+1)/(p+1)²", "directivity (dBi)", "half-power beamwidth (°)", "first sidelobe (dB)"], &rows));
    // Patch E- and H-plane.
    let f10 = 10e9;
    let patch = PatchAntenna::design(f10, 2.2, 1.57e-3).unwrap();
    let mut plot = Plot::new(&format!("Microstrip patch at 10 GHz (ε_r = 2.2, W = {:.2} mm, L = {:.2} mm)", 1e3 * patch.width, 1e3 * patch.length), "θ (degrees, measured from broadside)", "relative power (dB)");
    let peak = patch.radiation_intensity(0.0, 0.0, f10);
    for (k, (name, phi)) in [("E-plane (φ = 0)", 0.0), ("H-plane (φ = 90°)", FRAC_PI_2)].into_iter().enumerate() {
        let pts: Vec<(f64, f64)> = (-89..=89).map(|d| {
            let t = (d as f64).to_radians();
            let (theta, ph) = if t >= 0.0 { (t, phi) } else { (-t, phi + PI) };
            (d as f64, power_to_db((patch.radiation_intensity(theta, ph, f10) / peak).max(1e-5)))
        }).collect();
        plot.series.push(Series::new(name, pts, Style::Line, k));
    }
    plot.y_range = Some((-30.0, 2.0));
    plot.legend = Legend::BottomRight;
    write(CHAPTER, "patch_patterns.svg", &plot.render());
    // Summary against closed forms.
    let fz = 10e9;
    let lambda = C0 / fz;
    let hz = HertzianDipole::new(lambda / 50.0);
    let lp = MagneticLoop::new(lambda / 30.0, 1);
    let hw = HalfWaveDipole::new(fz);
    let fw = Dipole::new(lambda);
    let dish = Dish::new(10.0 * lambda);
    let k_a = TAU / 30.0;
    let row = |name: &str, d: f64, d_ref: &str, r: Option<f64>, r_ref: &str| vec![name.to_string(), format!("{d:.4}"), d_ref.to_string(), r.map_or("".to_string(), |v| format!("{v:.4}")), r_ref.to_string()];
    let rows = vec![
        row("Hertzian dipole, l = λ/50", hz.peak_directivity(fz), "1.5", hz.radiation_resistance(fz), &format!("80π²(l/λ)² = {:.4}", 80.0 * PI * PI / 2500.0)),
        row("small loop, a = λ/30", lp.peak_directivity(fz), "1.5", lp.radiation_resistance(fz), &format!("20π²(ka)⁴ = {:.4}", 20.0 * PI * PI * k_a.powi(4))),
        row("half-wave dipole", hw.peak_directivity(fz), "1.6409", hw.radiation_resistance(fz), "30 Cin(2π) = 73.08"),
        row("full-wave dipole", fw.peak_directivity(fz), "2.41", fw.radiation_resistance(fz), "199.1 (at the current maximum)"),
        row("uniform dish, D = 10λ", dish.peak_directivity(fz), &format!("(πD/λ)² = {:.2}", (10.0 * PI).powi(2)), None, ""),
        row(&format!("patch, ε_r = 2.2 (edge resistance {:.0} Ω)", PatchAntenna::design(fz, 2.2, 1.57e-3).unwrap().edge_resistance(fz)), PatchAntenna::design(fz, 2.2, 1.57e-3).unwrap().peak_directivity(fz), "≈ 5–7 (typical)", None, ""),
    ];
    write(CHAPTER, "antenna_table.md", &md_table(&["model", "directivity (computed)", "closed form", "radiation resistance (Ω, computed)", "closed form"], &rows));
}

// ------------------------------------------------------------------ beamforming

struct Demo {
    array: SensorArray,
    rs: Array2<C64>,
    rn: Array2<C64>,
    look: f64,
    jammers: [f64; 2],
}

fn demo_scene() -> Demo {
    let f = 1.5e9;
    let lambda = C0 / f;
    let array = SensorArray::from_antenna(&SensorArray::linear_positions(16, lambda / 2.0, [1.0, 0.0, 0.0]), Arc::new(HalfWaveDipole::new(f)), &RigidTransform::identity(), f, CONV, 1.0).unwrap();
    let (look, jammers) = (1.2, [0.75, 2.4]);
    let bw = 1e6;
    let desired = Scene::new(bw).unwrap().with_point_source(PointSource::new(look, 0.0, 3e-13, Coherency::theta()).unwrap());
    let environment = BrightnessScene::from_fn(Arc::new(Domain::far_field_gauss(40, 80).unwrap()), f, SkyPolarization::Unpolarized, |t, _| if t > FRAC_PI_2 { 290.0 } else { 10.0 }).unwrap();
    let mut interference = Scene::new(bw).unwrap().with_distributed(environment).with_receiver_temperature(75.0);
    for &j in &jammers {
        interference = interference.with_point_source(PointSource::new(j, 0.0, 3e-8, Coherency::theta()).unwrap());
    }
    Demo { rs: desired.covariance(&array).unwrap(), rn: interference.covariance(&array).unwrap(), array, look, jammers }
}

fn beamforming_figures() {
    let d = demo_scene();
    let pol = [C64::new(1.0, 0.0), C64::new(0.0, 0.0)];
    let a_look = d.array.steering_vector(d.look, 0.0, pol).unwrap();
    let r = &d.rs + &d.rn;
    let cfm = cfm_for_direction(&d.array, d.look, 0.0, pol).unwrap();
    let mvdr = mvdr_beamformer(&r, &a_look).unwrap();
    let (opt, _) = max_snr_beamformer(&d.rs, &d.rn).unwrap();
    let dirs = [d.look, d.jammers[0], d.jammers[1]];
    let c = Array2::from_shape_fn((16, 3), |(i, k)| d.array.steering_vector(dirs[k], 0.0, pol).unwrap()[i]);
    let lcmv = lcmv_beamformer(&r, &c, &Array1::from(vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)])).unwrap();
    let mut plot = Plot::new("16 half-wave dipoles, λ/2 apart: pattern in the xz plane", "θ (degrees)", "response re look direction (dB)");
    for (k, (name, w)) in [("CFM (conventional)", &cfm), ("MVDR (adaptive)", &mvdr)].into_iter().enumerate() {
        let g = response(w, &a_look).unwrap().norm();
        let pts: Vec<(f64, f64)> = (1..720).map(|i| PI * i as f64 / 720.0).map(|t| (t.to_degrees(), amplitude_to_db((d.array.pattern_at(w, t, 0.0).unwrap()[0].norm() / g).max(1e-5)))).collect();
        plot.series.push(Series::new(name, pts, Style::Line, k));
    }
    plot.series.push(Series::new("jammers", d.jammers.iter().map(|j| (j.to_degrees(), 0.0)).collect(), Style::Markers, 3));
    plot.series.push(Series::new("look direction", vec![(d.look.to_degrees(), 0.0)], Style::Markers, 2));
    plot.y_range = Some((-80.0, 5.0));
    plot.legend = Legend::BottomLeft;
    write(CHAPTER, "beam_patterns.svg", &plot.render());
    // Sample-matrix-inversion MVDR: SINR loss against the number of training snapshots.
    let opt_sinr = sinr(&opt, &d.rs, &d.rn).unwrap();
    let n = 16.0;
    let mut sim = Vec::new();
    let mut theory = Vec::new();
    for &k in &[17usize, 20, 24, 32, 48, 64, 96, 128, 256, 512] {
        let trials = 200;
        let mean: f64 = (0..trials)
            .map(|t| {
                let snaps = synthesize_snapshots(&d.rn, k, 1000 + t as u64).unwrap();
                let w = mvdr_beamformer(&sample_covariance(&snaps).unwrap(), &a_look).unwrap();
                sinr(&w, &d.rs, &d.rn).unwrap() / opt_sinr
            })
            .sum::<f64>()
            / trials as f64;
        sim.push((k as f64, power_to_db(mean)));
        theory.push((k as f64, power_to_db((k as f64 + 2.0 - n) / (k as f64 + 1.0))));
    }
    let mut plot = Plot::new("MVDR from a sample covariance: SINR relative to the optimum", "training snapshots K (16 elements)", "SINR loss (dB)");
    plot.series.push(Series::new("simulated (mean of 200 trials)", sim, Style::Markers, 0));
    plot.series.push(Series::new("Reed-Mallett-Brennan: (K + 2 − N)/(K + 1)", theory, Style::Line, 1));
    plot.log_x = true;
    plot.legend = Legend::BottomRight;
    write(CHAPTER, "smi_loss.svg", &plot.render());
    let mut rows = Vec::new();
    for (name, w) in [("CFM", &cfm), ("max-SNR", &opt), ("MVDR", &mvdr), ("LCMV (look + 2 nulls)", &lcmv)] {
        let g = response(w, &a_look).unwrap().norm();
        let rel = |j: f64| amplitude_to_db(response(w, &d.array.steering_vector(j, 0.0, pol).unwrap()).unwrap().norm().max(1e-15) / g);
        rows.push(vec![name.to_string(), format!("{:.2}", power_to_db(sinr(w, &d.rs, &d.rn).unwrap())), format!("{:.1}", rel(d.jammers[0])), format!("{:.1}", rel(d.jammers[1]))]);
    }
    write(CHAPTER, "beamformer_table.md", &md_table(&["beamformer", "SINR (dB)", "jammer 1 response (dB)", "jammer 2 response (dB)"], &rows));
}

// ------------------------------------------------------------------ direction finding

fn doa_figures() {
    let f = 1e9;
    let lambda = C0 / f;
    let positions = SensorArray::linear_positions(8, lambda / 2.0, [1.0, 0.0, 0.0]);
    let array = SensorArray::from_antenna(&positions, Arc::new(HertzianDipole::new(0.01)), &RigidTransform::identity(), f, CONV, 1.0).unwrap();
    let pol = [C64::new(1.0, 0.0), C64::new(0.0, 0.0)];
    let covariance = |az: &[f64], snr_db: f64, k: usize, seed: u64| {
        let noise = 10f64.powf(-snr_db / 10.0);
        let mut r = Array2::<C64>::eye(8) * C64::new(noise, 0.0);
        for &a in az {
            let s = array.steering_vector(FRAC_PI_2, a.to_radians(), pol).unwrap();
            let norm = (s.iter().map(|v| v.norm_sqr()).sum::<f64>() / 8.0).sqrt();
            for i in 0..8 {
                for j in 0..8 {
                    r[[i, j]] += s[i] * s[j].conj() / (norm * norm);
                }
            }
        }
        sample_covariance(&synthesize_snapshots(&r, k, seed).unwrap()).unwrap()
    };
    let truth = [85.0, 93.0];
    let r = covariance(&truth, 0.0, 200, 7);
    let grid: Vec<f64> = (0..=1800).map(|i| 45.0 + 90.0 * i as f64 / 1800.0).collect();
    let dirs: Vec<(f64, f64)> = grid.iter().map(|&a| (FRAC_PI_2, a.to_radians())).collect();
    let spectra = [
        ("Bartlett", bartlett_spectrum(&array, &r, &dirs, pol).unwrap()),
        ("Capon", capon_spectrum(&array, &r, &dirs, pol).unwrap()),
        ("MUSIC", music_spectrum(&array, &r, 2, &dirs, pol).unwrap()),
    ];
    let mut plot = Plot::new("Two sources 8° apart, 8 elements λ/2 apart, 0 dB SNR, 200 snapshots", "azimuth φ (degrees)", "spectrum (dB re its peak)");
    for (k, (name, s)) in spectra.iter().enumerate() {
        let peak = s.iter().cloned().fold(0.0, f64::max);
        plot.series.push(Series::new(name, grid.iter().zip(s).map(|(&a, &v)| (a, power_to_db(v / peak))).collect(), Style::Line, k));
    }
    plot.series.push(Series::new("true directions", truth.iter().map(|&a| (a, 0.0)).collect(), Style::Markers, 3));
    plot.y_range = Some((-40.0, 3.0));
    plot.legend = Legend::BottomLeft;
    write(CHAPTER, "doa_spectra.svg", &plot.render());
    let esprit: Vec<f64> = esprit_ula(&r, 2, lambda / 2.0, TAU / lambda, CONV).unwrap().iter().map(|u| u.acos().to_degrees()).collect();
    let mut found: Vec<String> = Vec::new();
    for (name, s) in &spectra {
        let mut p: Vec<f64> = spectrum_peaks(s, 2).iter().map(|&i| grid[i]).collect();
        p.sort_by(f64::total_cmp);
        found.push(format!("{name} {}", p.iter().map(|v| format!("{v:.1}°")).collect::<Vec<_>>().join(" and ")));
    }
    let mut e = esprit.clone();
    e.sort_by(f64::total_cmp);
    found.push(format!("ESPRIT {}", e.iter().map(|v| format!("{v:.2}°")).collect::<Vec<_>>().join(" and ")));
    write(CHAPTER, "doa_facts.md", &format!("Estimates (true 85° and 93°): {}. The Bartlett spectrum shows a single merged peak.", found.join("; ")));
    // RMS error of a single source against SNR.
    let mut rows = Vec::new();
    for snr in [-10.0, -5.0, 0.0, 10.0, 20.0] {
        let trials = 40;
        let (mut se_music, mut se_esprit, mut se_capon) = (0.0, 0.0, 0.0);
        for t in 0..trials {
            let r = covariance(&[80.0], snr, 100, 100 + t);
            let fine: Vec<f64> = (0..=600).map(|i| 75.0 + 10.0 * i as f64 / 600.0).collect();
            let fd: Vec<(f64, f64)> = fine.iter().map(|&a| (FRAC_PI_2, a.to_radians())).collect();
            let m = music_spectrum(&array, &r, 1, &fd, pol).unwrap();
            let c = capon_spectrum(&array, &r, &fd, pol).unwrap();
            let argmax = |s: &[f64]| fine[(0..s.len()).max_by(|&a, &b| s[a].total_cmp(&s[b])).unwrap()];
            se_music += (argmax(&m) - 80.0).powi(2);
            se_capon += (argmax(&c) - 80.0).powi(2);
            se_esprit += (esprit_ula(&r, 1, lambda / 2.0, TAU / lambda, CONV).unwrap()[0].acos().to_degrees() - 80.0).powi(2);
        }
        let rms = |s: f64| format!("{:.3}", (s / trials as f64).sqrt());
        rows.push(vec![format!("{snr}"), rms(se_capon), rms(se_music), rms(se_esprit)]);
    }
    write(CHAPTER, "doa_rmse.md", &md_table(&["SNR per element (dB)", "Capon RMS error (°)", "MUSIC RMS error (°)", "ESPRIT RMS error (°)"], &rows));
}
