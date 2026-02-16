use std::{fs::File, io::Write, path::PathBuf};

use lib_jsl::dsp::filters::remez::{remez, RemezFilterType};

fn write_csv(path: PathBuf, data: &[f64]) {
    let mut f = File::create(path).expect("create csv");
    for &v in data {
        writeln!(f, "{v:.17}").expect("write line");
    }
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/dsp/test_data");
    std::fs::create_dir_all(&root).expect("create test_data");

    let low = remez(
        17,
        &[0.0, 0.2, 0.3, 0.5],
        &[1.0, 0.0],
        None,
        RemezFilterType::Bandpass,
        25,
        16,
        1.0,
    )
    .expect("design lowpass");
    write_csv(root.join("remez_lowpass_numtaps17.csv"), &low);

    let high = remez(
        21,
        &[0.0, 0.14, 0.2, 0.5],
        &[0.0, 1.0],
        Some(&[2.0, 1.0]),
        RemezFilterType::Bandpass,
        25,
        16,
        1.0,
    )
    .expect("design highpass");
    write_csv(root.join("remez_highpass_numtaps21_weighted.csv"), &high);
}

