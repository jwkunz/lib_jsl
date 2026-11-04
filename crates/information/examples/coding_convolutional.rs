//! Convolutional encoding and Viterbi decoding, with hard and soft decisions.

use lib_jsl_information::forward_error_correction::ConvolutionalCode;

fn main() {
    let code = ConvolutionalCode::new(3, &[0o7, 0o5]).unwrap();
    let data = [1, 0, 1, 1, 0, 0, 1];
    let coded = code.encode(&data, true);
    println!("K = 3 (7, 5): {} bits -> {} coded bits {:?}", data.len(), coded.len(), coded);
    // Flip two well-separated coded bits: the free distance 5 corrects them.
    let mut rx = coded.clone();
    rx[2] ^= 1;
    rx[11] ^= 1;
    println!("hard Viterbi with 2 bit errors: {:?}", code.decode_hard(&rx, true).unwrap());
    // Soft decisions: unreliable bits (small |LLR|) count for less.
    let mut llr: Vec<f64> = coded.iter().map(|&b| if b == 0 { 4.0 } else { -4.0 }).collect();
    for i in [0, 1, 2, 5] {
        llr[i] = -0.3 * llr[i].signum(); // four weakly wrong values
    }
    println!("soft Viterbi with 4 weak errors: {:?}", code.decode_soft(&llr, true).unwrap());
    let nasa = ConvolutionalCode::nasa_k7();
    println!("NASA K = 7: {} outputs per bit, {} tail bits", nasa.outputs(), nasa.tail_len());
}
