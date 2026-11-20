//! Tables and figures for release 2.13.0 (linear algebra chapter): the real QZ algorithm,
//! `ordqz` and sorted Schur forms, and `hfft`, `ihfft` and the fast Hankel transform, compared
//! with SciPy.

use crate::refs::{Refs, fmt_diff, max_matched_distance};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_ffts::{fht, hfft, ifht, ihfft};
use lib_jsl_linalg::{EigenSort, ordqz, ordqz_complex, qz_real, schur_sorted, schur_sorted_complex};
use ndarray::Array2;
use num_complex::Complex;

const CHAPTER: &str = "linear_algebra";
type C = Complex<f64>;

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/qz213_reference.txt"));
    qz_table(&refs);
    transform_table(&refs);
    figures(&refs);
}

fn square(v: &[f64]) -> Array2<f64> {
    let n = (v.len() as f64).sqrt() as usize;
    Array2::from_shape_vec((n, n), v.to_vec()).unwrap()
}

fn projector(z: &Array2<f64>, s: usize) -> Array2<f64> {
    let zs = z.slice(ndarray::s![.., ..s]);
    zs.dot(&zs.t())
}

fn max_diff(a: impl IntoIterator<Item = f64>, b: &[f64]) -> f64 {
    a.into_iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn sort_of(name: &str) -> EigenSort<'static> {
    match name {
        "lhp" => EigenSort::LeftHalfPlane,
        "rhp" => EigenSort::RightHalfPlane,
        "iuc" => EigenSort::InsideUnitCircle,
        _ => EigenSort::OutsideUnitCircle,
    }
}

fn qz_table(refs: &Refs) {
    let mut rows = Vec::new();
    for (name, label) in [("rand6", "6 × 6"), ("rand12", "12 × 12"), ("rand30", "30 × 30"), ("singular8", "8 × 8, B with two zero rows (rhp, iuc)")] {
        let a = square(refs.values(&format!("{name} a")));
        let b = square(refs.values(&format!("{name} b")));
        let f = qz_real(&a, &b).unwrap();
        let backward = (f.q.dot(&f.aa).dot(&f.z.t()) - &a).iter().fold(0.0f64, |m, v| m.max(v.abs())).max((f.q.dot(&f.bb).dot(&f.z.t()) - &b).iter().fold(0.0f64, |m, v| m.max(v.abs())));
        // Eigenvalues: finite ones matched, infinite ones counted.
        let o = ordqz(&a, &b, &EigenSort::Custom(&|_| false)).unwrap();
        let w = refs.get(&format!("{name} eig"));
        let lib: Vec<C> = o.alpha.iter().zip(&o.beta).filter(|(_, b)| **b != 0.0).map(|(al, be)| al / *be).collect();
        let (mut re, mut im) = (Vec::new(), Vec::new());
        for ((r, i), b) in w[0].iter().zip(&w[1]).zip(&w[2]) {
            if *b != 0.0 {
                re.push(r / b);
                im.push(i / b);
            }
        }
        let eig = if lib.len() == re.len() {
            let scale = re.iter().zip(&im).map(|(x, y)| x.hypot(*y)).fold(1.0f64, f64::max);
            fmt_diff(max_matched_distance(&lib, &re, &im) / scale)
        } else {
            format!("{} vs {} finite", lib.len(), re.len())
        };
        let (mut dq, mut same_sdim) = (0.0f64, true);
        // With a singular B, LAPACK leaves "infinite" eigenvalues with β of rounding size whose
        // sign decides 'lhp' and 'ouc'; only the other two sorts are comparable.
        let sorts: &[&str] = if name == "singular8" { &["rhp", "iuc"] } else { &["lhp", "rhp", "iuc", "ouc"] };
        for &s in sorts {
            let r = ordqz(&a, &b, &sort_of(s)).unwrap();
            let want = refs.get(&format!("{name} ordqz {s}"));
            same_sdim &= r.sdim as f64 == want[0][0];
            dq = dq.max(max_diff(projector(&r.z, r.sdim).iter().copied(), &want[1])).max(max_diff(projector(&r.q, r.sdim).iter().copied(), &want[2]));
            let rc = ordqz_complex(&a, &b, &sort_of(s)).unwrap();
            same_sdim &= rc.sdim as f64 == want[0][0];
        }
        let mut ds = 0.0f64;
        if name != "singular8" {
            for s in ["lhp", "rhp", "iuc", "ouc"] {
                let (_, z, sdim) = schur_sorted(&a, &sort_of(s)).unwrap();
                let want = refs.get(&format!("{name} schur {s}"));
                same_sdim &= sdim as f64 == want[0][0];
                ds = ds.max(max_diff(projector(&z, sdim).iter().copied(), &want[1]));
                let (_, zc, sc) = schur_sorted_complex(&a, &sort_of(s)).unwrap();
                let wc = refs.get(&format!("{name} schurc {s}"));
                same_sdim &= sc as f64 == wc[0][0];
                let zs = zc.slice(ndarray::s![.., ..sc]);
                let pc = zs.dot(&zs.t().mapv(|v| v.conj()));
                ds = ds.max(max_diff(pc.iter().map(|v| v.re), &wc[1])).max(max_diff(pc.iter().map(|v| v.im), &wc[2]));
            }
        }
        rows.push(vec![
            label.into(),
            fmt_diff(backward),
            eig,
            if same_sdim { "same".into() } else { "differ".into() },
            fmt_diff(dq),
            if name == "singular8" { "—".into() } else { fmt_diff(ds) },
        ]);
    }
    write(
        CHAPTER,
        "qz213_qz.md",
        &md_table(&["Pencil", "‖Q AA Zᵀ − A‖, ‖Q BB Zᵀ − B‖", "Eigenvalues vs SciPy", "sdim (real and complex)", "`ordqz` subspaces", "`schur(sort)` subspaces"], &rows),
    );
}

fn transform_table(refs: &Refs) {
    let mut rows = Vec::new();
    let x = refs.get("hfft x");
    let xc: Vec<C> = x[0].iter().zip(&x[1]).map(|(a, b)| C::new(*a, *b)).collect();
    let h = refs.get("hfft");
    let d = max_diff(hfft(&xc, None).unwrap(), &h[0]).max(max_diff(hfft(&xc, Some(13)).unwrap(), &h[1])).max(max_diff(hfft(&xc, Some(6)).unwrap(), &h[2]));
    rows.push(vec!["`hfft`, 9 inputs; n = 16, 13, 6".into(), fmt_diff(d)]);
    let ih = ihfft(refs.values("ihfft y")).unwrap();
    let w = refs.get("ihfft");
    rows.push(vec!["`ihfft`, 11 inputs".into(), fmt_diff(max_diff(ih.iter().map(|v| v.re), &w[0]).max(max_diff(ih.iter().map(|v| v.im), &w[1])))]);
    for (name, label) in [("gauss0", "μ = 0, r e^{−r²/2}"), ("pow1", "μ = 1, r² e^{−r}"), ("bias", "μ = ½, bias 0.4, log-normal"), ("neg", "μ = −½, bias −0.3, 1/(1 + r²)")] {
        let w = refs.get(&format!("fht {name}"));
        let (mu, bias, off) = (w[0][0], w[0][1], w[0][2]);
        let off_lib = lib_jsl_ffts::fhtoffset(0.05, mu, 0.3, bias);
        let big = fht(&w[1], 0.05, mu, off, bias).unwrap();
        let scale = w[2].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let back = ifht(&w[2], 0.05, mu, off, bias).unwrap();
        let bscale = w[3].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        rows.push(vec![
            format!("`fht`, `ifht`, `fhtoffset`: {label}, n = 128"),
            format!("{} / {} / {}", fmt_diff(max_diff(big, &w[2]) / scale), fmt_diff(max_diff(back, &w[3]) / bscale), fmt_diff((off_lib - off).abs())),
        ]);
    }
    write(CHAPTER, "qz213_transforms.md", &md_table(&["Transform", "Largest difference from SciPy (relative to the largest value)"], &rows));
}

fn figures(refs: &Refs) {
    let a = square(refs.values("rand30 a"));
    let b = square(refs.values("rand30 b"));
    let o = ordqz(&a, &b, &EigenSort::InsideUnitCircle).unwrap();
    let lam: Vec<C> = o.alpha.iter().zip(&o.beta).map(|(al, be)| al / *be).collect();
    let mut plot = Plot::new(&format!("ordqz: the {} eigenvalues inside the unit circle come first", o.sdim), "Re λ", "Im λ");
    plot.legend = Legend::TopRight;
    plot.x_range = Some((-5.5, 5.5));
    plot.y_range = Some((-3.0, 3.0));
    plot.series.push(Series::new("unit circle", (0..=200).map(|i| { let t = i as f64 * std::f64::consts::TAU / 200.0; (t.cos(), t.sin()) }).collect(), Style::Dashed, 7));
    plot.series.push(Series::new("selected (leading block)", lam[..o.sdim].iter().map(|l| (l.re, l.im)).collect(), Style::Markers, 0));
    plot.series.push(Series::new("others", lam[o.sdim..].iter().filter(|l| l.norm() < 6.0).map(|l| (l.re, l.im)).collect(), Style::Markers, 1));
    write(CHAPTER, "ordqz_eigenvalues.svg", &plot.render());

    let w = refs.get("fht gauss0");
    let (n, dln, off) = (128usize, 0.05, w[0][2]);
    let jc = (n as f64 - 1.0) / 2.0;
    let big = fht(&w[1], dln, 0.0, off, 0.0).unwrap();
    let k: Vec<f64> = (0..n).map(|j| off.exp() * ((j as f64 - jc) * dln).exp()).collect();
    let mut plot = Plot::new("FFTLog: the Hankel transform of r e^(−r²/2) is k e^(−k²/2)", "k", "A(k)");
    plot.log_x = true;
    plot.legend = Legend::TopLeft;
    plot.series.push(Series::new("exact", k.iter().map(|&x| (x, x * (-x * x / 2.0).exp())).collect(), Style::Line, 7));
    plot.series.push(Series::new("fht, 128 log-spaced samples", k.iter().zip(&big).map(|(&x, &v)| (x, v)).collect(), Style::Markers, 0));
    write(CHAPTER, "fht_gaussian.svg", &plot.render());
}

