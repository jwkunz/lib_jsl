/// Benchmarking code to compare the performance of different FFT implementations on the same input data.
/// This code uses the Criterion crate for benchmarking, and includes a variety of FFT implementations, including the simple Cooley-Tukey algorithm, an optimized radix-2 implementation, an optimized split-radix implementation, a SIMD-optimized implementation, and the RustFFT library. 
/// The benchmarks are designed to compare the execution time of each FFT implementation on the same input data, allowing us to evaluate the performance of each implementation and identify any differences in speed or efficiency. 
/// The results of these benchmarks can be used to inform decisions about which FFT implementation to use for different applications, based on factors such as input size, performance requirements, and numerical accuracy.
/// The benchmark data is loaded from binary files containing precomputed input data, and the benchmarks are organized into groups based on the input size and ordering (standard vs. bit-reversed). 
/// The results of the benchmarks can be analyzed to identify trends and performance characteristics of each FFT implementation, providing valuable insights for optimizing FFT computations in various applications.
/// The benchmarking code is structured to be easily extensible, allowing for additional FFT implementations to be added in the future for comparison. 
/// The use of Criterion provides a robust framework for conducting and analyzing benchmarks, making it easier to draw meaningful conclusions from the results.
/// Overall, this benchmarking code serves as a valuable tool for evaluating the performance of different FFT implementations and guiding decisions about which implementation to use in different contexts.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
mod rust_fft_wrapper;
use lib_jsl::ffts::{
    best_fft::BestFft,
    bluestein_fft::BluesteinFft,
    simd_fft::SimdFft,
    fft_enginer_trait::{FfftEngine1D, FftDirection, FftOrdering, FftScaleFactor},
    optimized_radix2::OptimizedRadix2FFT,
    optimized_split_radix::OptimizedSplitRadixFFT,
    simple_cooley_tukey::SimpleCooleyTukeyFFT,
};
use num::Complex;
use rust_fft_wrapper::RustFftWrapper;

fn parse_complex_bin(bytes: &[u8]) -> Vec<Complex<f64>> {
    assert!(bytes.len() % 16 == 0, "binary complex data must be 16-byte aligned");
    bytes
        .chunks_exact(16)
        .map(|chunk| {
            let mut re = [0_u8; 8];
            let mut im = [0_u8; 8];
            re.copy_from_slice(&chunk[0..8]);
            im.copy_from_slice(&chunk[8..16]);
            Complex::new(f64::from_le_bytes(re), f64::from_le_bytes(im))
        })
        .collect()
}

fn fft_gaussian_32768_input() -> Vec<Complex<f64>> {
    parse_complex_bin(include_bytes!("../src/ffts/test_data/fft_gaussian_32768_input.bin"))
}

fn fft_gaussian_63_input() -> Vec<Complex<f64>> {
    parse_complex_bin(include_bytes!("../src/ffts/test_data/fft_gaussian_63_input.bin"))
}

fn bit_reverse_index(mut n: usize, bits: usize) -> usize {
    let mut reversed = 0usize;
    for _ in 0..bits {
        reversed = (reversed << 1) | (n & 1);
        n >>= 1;
    }
    reversed
}

fn make_bit_reversed_input(input: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let bits = input.len().trailing_zeros() as usize;
    let mut out = vec![Complex::new(0.0, 0.0); input.len()];
    for (i, value) in input.iter().enumerate() {
        out[bit_reverse_index(i, bits)] = *value;
    }
    out
}

fn bench_engine_execute<E: FfftEngine1D>(
    c: &mut Criterion,
    group_name: &str,
    engine_name: &str,
    mut engine: E,
    input: &[Complex<f64>],
    ordering: FftOrdering,
) {
    engine
        .plan(
            input.len(),
            FftScaleFactor::None,
            FftDirection::Forward,
            ordering,
        )
        .unwrap();

    let mut group = c.benchmark_group(group_name);
    group.throughput(Throughput::Elements(input.len() as u64));
    group.bench_with_input(BenchmarkId::from_parameter(engine_name), input, |b, data| {
        b.iter(|| {
            let out = engine.execute(black_box(data)).unwrap();
            black_box(out);
        })
    });
    group.finish();
}

fn fft_race(c: &mut Criterion) {
    let input = fft_gaussian_32768_input();
    let bit_reversed_input = make_bit_reversed_input(&input);
    let input63 = fft_gaussian_63_input();

    bench_engine_execute(
        c,
        "fft_execute_standard_32768",
        "simple_cooley_tukey",
        SimpleCooleyTukeyFFT::new(),
        &input,
        FftOrdering::Standard,
    );
    bench_engine_execute(
        c,
        "fft_execute_standard_32768",
        "optimized_radix2",
        OptimizedRadix2FFT::new(),
        &input,
        FftOrdering::Standard,
    );
    bench_engine_execute(
        c,
        "fft_execute_standard_32768",
        "optimized_split_radix",
        OptimizedSplitRadixFFT::new(),
        &input,
        FftOrdering::Standard,
    );
    bench_engine_execute(
        c,
        "fft_execute_standard_32768",
        "simd_fft",
        SimdFft::new(),
        &input,
        FftOrdering::Standard,
    );
    bench_engine_execute(
        c,
        "fft_execute_standard_32768",
        "best_fft",
        BestFft::new(),
        &input,
        FftOrdering::Standard,
    );
    bench_engine_execute(
        c,
        "fft_execute_standard_32768",
        "rustfft_wrapper",
        RustFftWrapper::new(),
        &input,
        FftOrdering::Standard,
    );
    bench_engine_execute(
        c,
        "fft_execute_bit_reversed_32768",
        "simple_cooley_tukey",
        SimpleCooleyTukeyFFT::new(),
        &bit_reversed_input,
        FftOrdering::BitReversed,
    );
    bench_engine_execute(
        c,
        "fft_execute_bit_reversed_32768",
        "optimized_radix2",
        OptimizedRadix2FFT::new(),
        &bit_reversed_input,
        FftOrdering::BitReversed,
    );
    bench_engine_execute(
        c,
        "fft_execute_bit_reversed_32768",
        "optimized_split_radix",
        OptimizedSplitRadixFFT::new(),
        &bit_reversed_input,
        FftOrdering::BitReversed,
    );
    bench_engine_execute(
        c,
        "fft_execute_bit_reversed_32768",
        "simd_fft",
        SimdFft::new(),
        &bit_reversed_input,
        FftOrdering::BitReversed,
    );
    bench_engine_execute(
        c,
        "fft_execute_bit_reversed_32768",
        "best_fft",
        BestFft::new(),
        &bit_reversed_input,
        FftOrdering::BitReversed,
    );
    bench_engine_execute(
        c,
        "fft_execute_bit_reversed_32768",
        "rustfft_wrapper",
        RustFftWrapper::new(),
        &bit_reversed_input,
        FftOrdering::BitReversed,
    );

    bench_engine_execute(
        c,
        "fft_execute_standard_63",
        "bluestein_fft",
        BluesteinFft::new(),
        &input63,
        FftOrdering::Standard,
    );
    bench_engine_execute(
        c,
        "fft_execute_standard_63",
        "rustfft_wrapper",
        RustFftWrapper::new(),
        &input63,
        FftOrdering::Standard,
    );
}

criterion_group!(benches, fft_race);
criterion_main!(benches);
