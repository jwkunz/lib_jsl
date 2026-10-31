//! Digital communications blocks and algorithms.

pub mod constellations;
pub mod barker_codes;
pub mod decision_feedback_equalizer;
pub mod farrow_filter;
pub mod mueller_timing_detector;
pub mod ofdm;
pub mod fm;
pub mod pulse_correlation_tracker;
pub mod addressable_shift_register;
pub mod ping_pong_buffer;

pub use constellations::{qam16, Constellation};
pub use barker_codes::{barker_autocorrelation, barker_code, BARKER_LENGTHS};
pub use decision_feedback_equalizer::{decision_feedback_equalizer, DecisionFeedbackEqualizer};
pub use farrow_filter::{farrow_filter, FarrowFilter};
pub use mueller_timing_detector::{mueller_timing_detector, MuellerTimingRecovery};
pub use ofdm::ofdm_demodulate;
pub use fm::{carson_bandwidth, fm_demodulate, fm_modulate, FmDemodulator, FmModulator};
pub use pulse_correlation_tracker::pulse_correlation_tracker;
pub use addressable_shift_register::AddressableShiftRegister;
pub use ping_pong_buffer::{PingPongBuffer, PushOutcome};
