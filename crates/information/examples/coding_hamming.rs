//! The (7,4) Hamming code and its extended (8,4) SECDED form.

use lib_jsl_information::forward_error_correction::HammingCode;

fn main() {
    let code = HammingCode::new(3).unwrap();
    let data = [1, 0, 1, 1];
    let cw = code.encode(&data).unwrap();
    println!("(7,4): data {data:?} -> codeword {cw:?}");
    let mut rx = cw.clone();
    rx[4] ^= 1;
    let d = code.decode(&rx).unwrap();
    println!("one error at index 4: {:?}, data {:?}", d.status, d.data);
    let secded = HammingCode::new_extended(3).unwrap();
    let cw = secded.encode(&data).unwrap();
    let mut rx = cw.clone();
    rx[1] ^= 1;
    rx[6] ^= 1;
    println!("(8,4) with two errors: {:?}", secded.decode(&rx).unwrap().status);
    let big = HammingCode::new(6).unwrap();
    println!("r = 6: ({}, {}) code, rate {:.3}", big.n(), big.k(), big.k() as f64 / big.n() as f64);
}
