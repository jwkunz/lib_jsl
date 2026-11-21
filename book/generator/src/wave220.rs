//! Tables and figures for release 2.20.0 (special functions chapter): Mathieu, parabolic
//! cylinder and spheroidal wave functions, compared with SciPy.

use crate::refs::{Refs, fmt_diff};
use crate::spectral::md_table;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_special_functions::mathieu::*;
use lib_jsl_special_functions::parabolic_cylinder::{pbdv, pbvv, pbwa};
use lib_jsl_special_functions::spheroidal::*;

const CHAPTER: &str = "special_functions";

pub fn generate() {
    let refs = Refs::parse(include_str!("../data/wave220_reference.txt"));
    let mut rows = Vec::new();
    mathieu_rows(&refs, &mut rows);
    for (name, f) in [("pbdv", &pbdv as &dyn Fn(f64, f64) -> (f64, f64)), ("pbvv", &pbvv), ("pbwa", &pbwa)] {
        compare7(&mut rows, name, refs.values(name), f);
    }
    spheroidal_rows(&refs, &mut rows);
    figure();
    write(CHAPTER, "wave220.md", &md_table(&["Function", "Cases", "Largest relative difference (value; derivative)"], &rows));
}

/// The relative difference `|a − b| / max(|b|, floor)`.
fn rel(a: f64, b: f64, floor: f64) -> f64 {
    if a.is_nan() && b.is_nan() {
        return 0.0;
    }
    (a - b).abs() / b.abs().max(floor)
}

/// Compares a function of `(m, p, x)` returning (value, derivative) with the reference rows
/// `[m, p, x, y, dy]`, relative to the largest value of the row's case.
fn compare(rows: &mut Vec<Vec<String>>, label: &str, data: &[f64], f: &dyn Fn(usize, f64, f64) -> (f64, f64)) {
    let (mut wv, mut wd, mut worst) = (0.0f64, 0.0f64, String::new());
    let mut n = 0;
    for r in data.chunks(5) {
        let (y, dy) = f(r[0] as usize, r[1], r[2]);
        let (ev, ed) = (rel(y, r[3], 1e-3), rel(dy, r[4], 1e-3));
        if ev.max(ed) > wv.max(wd) {
            worst = format!("m = {}, {} , {}: {:.6e} vs {:.6e}", r[0], r[1], r[2], y, r[3]);
        }
        wv = wv.max(ev);
        wd = wd.max(ed);
        n += 1;
    }
    let _ = worst;
    rows.push(vec![label.into(), n.to_string(), format!("{}; {}", fmt_diff(wv), fmt_diff(wd))]);
}

/// Rows `[p, _, x, scipy y, scipy dy, mpmath y, mpmath dy]`: the library and SciPy against
/// mpmath, relative to `max(|value|, 1e-3)`.
fn compare7(rows: &mut Vec<Vec<String>>, name: &str, data: &[f64], f: &dyn Fn(f64, f64) -> (f64, f64)) {
    let (mut lib, mut sci) = (0.0f64, 0.0f64);
    let mut worst = String::new();
    let mut n = 0;
    for r in data.chunks(7) {
        if r[5].is_nan() {
            continue;
        }
        let (y, dy) = f(r[0], r[2]);
        let e = rel(y, r[5], 1e-3).max(rel(dy, r[6], 1e-3));
        if e > lib {
            worst = format!("({}, {}): {:.6e} vs {:.6e}", r[0], r[2], y, r[5]);
        }
        lib = lib.max(e);
        sci = sci.max(rel(r[3], r[5], 1e-3).max(rel(r[4], r[6], 1e-3)));
        n += 1;
    }
    let _ = worst;
    rows.push(vec![format!("`{name}` against mpmath (50 digits)"), n.to_string(), format!("library {}; SciPy {}", fmt_diff(lib), fmt_diff(sci))]);
}

fn mathieu_rows(refs: &Refs, rows: &mut Vec<Vec<String>>) {
    let ab = refs.values("mathieu ab");
    let (mut da, mut db) = (0.0f64, 0.0f64);
    for r in ab.chunks(4) {
        da = da.max(rel(mathieu_a(r[0] as usize, r[1]), r[2], 1.0));
        if r[0] > 0.0 {
            db = db.max(rel(mathieu_b(r[0] as usize, r[1]), r[3], 1.0));
        }
    }
    rows.push(vec!["`mathieu_a`, `mathieu_b`".into(), (ab.len() / 4).to_string(), format!("{}; {}", fmt_diff(da), fmt_diff(db))]);
    compare(rows, "`mathieu_cem`", refs.values("mathieu cem"), &mathieu_cem);
    compare(rows, "`mathieu_sem`", refs.values("mathieu sem"), &mathieu_sem);
    compare(rows, "`mathieu_modcem1`", refs.values("mathieu modcem1"), &mathieu_modcem1);
    compare(rows, "`mathieu_modcem2`", refs.values("mathieu modcem2"), &mathieu_modcem2);
    compare(rows, "`mathieu_modsem1`", refs.values("mathieu modsem1"), &mathieu_modsem1);
    compare(rows, "`mathieu_modsem2`", refs.values("mathieu modsem2"), &mathieu_modsem2);
}

fn spheroidal_rows(refs: &Refs, rows: &mut Vec<Vec<String>>) {
    let cv = refs.values("spheroidal cv");
    let (mut dp, mut dob) = (0.0f64, 0.0f64);
    for r in cv.chunks(5) {
        dp = dp.max(rel(pro_cv(r[0] as usize, r[1] as usize, r[2]), r[3], 1.0));
        dob = dob.max(rel(obl_cv(r[0] as usize, r[1] as usize, r[2]), r[4], 1.0));
    }
    rows.push(vec!["`pro_cv`, `obl_cv`".into(), (cv.len() / 5).to_string(), format!("{}; {}", fmt_diff(dp), fmt_diff(dob))]);
    for (name, f) in [("pro_ang1", &pro_ang1 as &dyn Fn(usize, usize, f64, f64) -> (f64, f64)), ("obl_ang1", &obl_ang1), ("pro_rad1", &pro_rad1), ("obl_rad1", &obl_rad1)] {
        let data = refs.values(name);
        let (mut wv, mut wd, mut worst) = (0.0f64, 0.0f64, String::new());
        for r in data.chunks(6) {
            let (y, dy) = f(r[0] as usize, r[1] as usize, r[2], r[3]);
            let (ev, ed) = (rel(y, r[4], 1e-3), rel(dy, r[5], 1e-3));
            if ev.max(ed) > wv.max(wd) {
                worst = format!("(m, n, c, x) = ({}, {}, {}, {}): {:.6e} vs {:.6e}", r[0], r[1], r[2], r[3], y, r[4]);
            }
            wv = wv.max(ev);
            wd = wd.max(ed);
        }
        let _ = worst;
        rows.push(vec![format!("`{name}`"), (data.len() / 6).to_string(), format!("{}; {}", fmt_diff(wv), fmt_diff(wd))]);
    }
}

fn figure() {
    let qs: Vec<f64> = (0..=120).map(|i| i as f64 * 0.25).collect();
    let mut plot = Plot::new("Mathieu characteristic values: a_m(q) (solid) and b_m(q) (dashed)", "q", "characteristic value");
    plot.legend = Legend::TopLeft;
    plot.y_range = Some((-40.0, 45.0));
    for m in 0..6usize {
        plot.series.push(Series::new(if m == 0 { "a_m, m = 0…5" } else { "" }, qs.iter().map(|&q| (q, mathieu_a(m, q))).collect(), Style::Line, m));
        if m > 0 {
            plot.series.push(Series::new(if m == 1 { "b_m, m = 1…5" } else { "" }, qs.iter().map(|&q| (q, mathieu_b(m, q))).collect(), Style::Dashed, m));
        }
    }
    write(CHAPTER, "mathieu_chart.svg", &plot.render());
}
