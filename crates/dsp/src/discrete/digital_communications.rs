//! Digital communications blocks and algorithms.

pub mod constellations {
    pub fn qam16() {
        todo!("qam16 is not yet implemented")
    }
}

pub fn barker_code(_length: usize) -> Vec<i8> {
    todo!("barker_code is not yet implemented")
}

pub fn decision_feedback_equalizer() {
    todo!("decision_feedback_equalizer is not yet implemented")
}

pub fn farro_filter() {
    todo!("farro_filter is not yet implemented")
}

pub fn mueller_timing_detector() {
    todo!("mueller_timing_detector is not yet implemented")
}

pub fn ofdm_demodulate() {
    todo!("ofdm_demodulate is not yet implemented")
}

pub fn fm_demodulate() {
    todo!("fm_demodulate is not yet implemented")
}

pub fn pulse_correlation_tracker() {
    todo!("pulse_correlation_tracker is not yet implemented")
}

pub struct AddressableShiftRegister<T> {
    _data: Vec<T>,
}

pub struct PingPongBuffer<T> {
    _buf_a: Vec<T>,
    _buf_b: Vec<T>,
}
