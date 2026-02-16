use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lib_jsl::ffts::{
    best_fft::BestFft,
    simd_fft::SimdFft,
    fft_enginer_trait::{FfftEngine1D, FftDirection, FftOrdering, FftScaleFactor},
    optimized_radix2::OptimizedRadix2FFT,
    optimized_split_radix::OptimizedSplitRadixFFT,
    rust_fft_wrapper::RustFftWrapper,
    simple_cooley_tukey::SimpleCooleyTukeyFFT,
};
use num::Complex;

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
}

criterion_group!(benches, fft_race);
criterion_main!(benches);
