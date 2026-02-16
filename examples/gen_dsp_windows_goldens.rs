use std::{fs, path::Path};

use lib_jsl::dsp::windows;

fn write_csv(path: &Path, data: &[f64]) {
    let mut out = String::new();
    for v in data {
        out.push_str(&format!("{:.17e}\n", v));
    }
    fs::write(path, out).unwrap();
}

fn main() {
    let dir = Path::new("src/dsp/test_data");
    fs::create_dir_all(dir).unwrap();

    let items: Vec<(&str, Vec<f64>)> = vec![
        ("boxcar", windows::boxcar(7, true)),
        ("triang", windows::triang(7, true)),
        ("parzen", windows::parzen(7, true)),
        ("bohman", windows::bohman(7, true)),
        ("blackman", windows::blackman(7, true)),
        ("nuttall", windows::nuttall(7, true)),
        ("blackmanharris", windows::blackmanharris(7, true)),
        ("flattop", windows::flattop(7, true)),
        ("bartlett", windows::bartlett(7, true)),
        ("barthann", windows::barthann(7, true)),
        ("hamming", windows::hamming(7, true)),
        ("kaiser", windows::kaiser(7, 14.0, true)),
        ("kaiser_bessel_derived", windows::kaiser_bessel_derived(7, 14.0, true)),
        ("gaussian", windows::gaussian(7, 1.0, true)),
        ("general_cosine", windows::general_cosine(7, &[1.0, 1.942604, 1.340318, 0.440811, 0.043097], true)),
        ("general_gaussian", windows::general_gaussian(7, 1.5, 1.0, true)),
        ("general_hamming", windows::general_hamming(7, 0.54, true)),
        ("chebwin", windows::chebwin(7, 100.0, true)),
        ("cosine", windows::cosine(7, true)),
        ("hann", windows::hann(7, true)),
        ("exponential", windows::exponential(7, None, 1.0, true)),
        ("tukey", windows::tukey(7, 0.5, true)),
        ("taylor", windows::taylor(7, 4, 30.0, true, true)),
        ("dpss", windows::dpss(7, 2.5, true)),
        ("lanczos", windows::lanczos(7, true)),
    ];

    for (name, data) in items {
        write_csv(&dir.join(format!("{name}.csv")), &data);
    }
}
