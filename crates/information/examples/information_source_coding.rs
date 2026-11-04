//! Lossless compression with the six source coders, through the shared `SourceCoder` trait.

use lib_jsl_information::operators::entropy;
use lib_jsl_information::source_coder::{
    ArithmeticSourceCoder, HuffmanSourceCoder, Lz77SourceCoder, Lz78SourceCoder, LzwSourceCoder, ShannonFanoSourceCoder, SourceCoder,
};

fn report<C: SourceCoder>(name: &str, data: &[u8]) {
    let packed = C::try_compress(data).unwrap();
    let unpacked = C::try_decompress(&packed).unwrap();
    assert_eq!(unpacked, data);
    println!("{name:>13}: {:5} bytes ({:.2} bits per input byte)", packed.len(), 8.0 * packed.len() as f64 / data.len() as f64);
}

fn main() {
    let text = "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, \
                it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness. "
        .repeat(8);
    let data = text.as_bytes();
    println!("{} bytes, order-0 entropy {:.3} bits per byte", data.len(), entropy(data));
    report::<HuffmanSourceCoder>("Huffman", data);
    report::<ShannonFanoSourceCoder>("Shannon-Fano", data);
    report::<ArithmeticSourceCoder>("arithmetic", data);
    report::<Lz77SourceCoder>("LZ77", data);
    report::<Lz78SourceCoder>("LZ78", data);
    report::<LzwSourceCoder>("LZW", data);
}
