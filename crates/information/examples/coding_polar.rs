//! A polar code with successive-cancellation decoding.

use lib_jsl_information::forward_error_correction::PolarCode;

fn main() {
    let code = PolarCode::new(3, 4, 0.5).unwrap();
    println!("N = {}, k = {}: information on channels {:?}", code.n(), code.k(), code.info_positions());
    let info = [1, 0, 1, 1];
    let cw = code.encode(&info).unwrap();
    println!("codeword {cw:?}");
    let mut rx = cw.clone();
    rx[0] ^= 1;
    println!("one flipped bit, SC decoding: {:?}", code.decode_hard(&rx).unwrap());
    let big = PolarCode::new(10, 512, 0.32).unwrap();
    let info: Vec<u8> = (0..512).map(|i| (i % 3 == 0) as u8).collect();
    let cw = big.encode(&info).unwrap();
    // Soft input: every received value reliable except 40 erased (LLR 0) positions.
    let llr: Vec<f64> = cw.iter().enumerate().map(|(i, &b)| if i % 25 == 7 { 0.0 } else if b == 0 { 3.0 } else { -3.0 }).collect();
    println!("N = 1024, k = 512, 41 erasures: recovered {}", big.decode(&llr).unwrap() == info);
}
