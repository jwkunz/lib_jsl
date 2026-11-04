# DSP: synchronisation, communications, radar and controls

The previous chapters designed [filters](../filters/index.md) and [measured spectra](../spectral/index.md). This one builds systems from them. A software radio generates carriers and test signals, maps
bits to symbols and back, passes them through a simulated channel, recovers the carrier and the symbol clock, and equalizes what the channel did. A radar
compresses long pulses into sharp echoes and tracks them. A controller drives a plant to a setpoint, and a Kalman filter estimates what cannot be measured directly. Most of these are
[stream operators](../../api/lib_jsl_dsp/discrete/stream_operator/trait.StreamOperator.html): they keep state between blocks, so a signal can be processed as it arrives.

## The central questions

- **What does the receiver not know?** Carrier frequency and phase, symbol timing, the channel's response and the noise level must all be estimated from the signal, usually by feedback loops
  that trade acquisition speed against noise.
- **Hard or soft?** Decisions made early are simple; likelihoods kept until the decoder are worth about 2 dB.
- **How much bandwidth, how much power?** Denser constellations carry more bits per symbol at a higher signal-to-noise ratio; chirps buy range resolution with bandwidth instead of peak power.
- **Is the model right?** Equalizers, Kalman filters and PID loops all perform as designed only if the system they assume resembles the real one; each has a
  diagnostic (the error vector magnitude, the normalised estimation error, the step response) that shows when it does not.

## Streaming buffers

Two small containers support the streaming blocks.

### Addressable shift register

[`AddressableShiftRegister`](../../api/lib_jsl_dsp/discrete/digital_communications/addressable_shift_register/struct.AddressableShiftRegister.html) is a delay line of fixed length whose every stage can be read or written: address 0 holds the newest sample and
`capacity - 1` the oldest. `push` shifts every stage by one and returns the value that falls off the end; `taps` reads a set of addresses at once, as a tapped delay line or an FIR
filter needs. It is a circular buffer, so a push costs $O(1)$ whatever the length. As a stream operator it is a pure delay of `capacity` samples.

### Ping-pong buffer

[`PingPongBuffer`](../../api/lib_jsl_dsp/discrete/digital_communications/ping_pong_buffer/struct.PingPongBuffer.html) is a double buffer: one block fills with incoming samples while the other, already full, is processed, so a producer and a consumer never
touch the same memory. When the filling block is complete it becomes ready for `take`. If the producer completes another block before the consumer has taken the first, the new samples
are dropped and counted as an **overrun**, not written over the block being read. As a stream operator it re-blocks a stream into whole blocks, and `flush` emits the final partial one.

## Fixes

Writing this chapter fixed one documentation problem: the [`KalmanFilter`](../../api/lib_jsl_dsp/discrete/controls/kalman/struct.KalmanFilter.html)'s description was attached to its result type, so the filter itself appeared undocumented.

## In the library

All of these are in `lib_jsl_dsp` ([API reference](../../api/lib_jsl_dsp/index.html)): sources in `discrete::sources`, communications blocks in `discrete::digital_communications`,
loops in `discrete::synchronization`, channel models in `discrete::transformations`, pulse compression in `discrete::radar`, and controllers in `discrete::controls`.

| You need | Use | Entry |
|---|---|---|
| A carrier, test waveform or noise; a sine without a multiplier; a frequency shift | `Nco`, `NcoSource`, `SineWave`, `QuadratureWave`, `Cordic32`, `Cordic64`, `FrequencyMixer` | [Signal sources](sources.md) |
| Bits to symbols and back; OFDM; FM | `Constellation`, `Ofdm`, `FmModulator`, `FmDemodulator` | [Modulation](modulation.md) |
| Carrier and symbol-timing recovery; a fractional delay | `PhaseLockedLoop`, `CostasLoop`, `MuellerTimingRecovery`, `FarrowFilter` | [Synchronisation](synchronisation.md) |
| A simulated channel; equalization | `ChannelImpairment`, `ChannelTransmitter`, `ChannelReceiver`, `NonLinearityTransformer`, `DecisionFeedbackEqualizer` | [Channels and equalisation](equalisation.md) |
| Range resolution from a long pulse; a pulse found and tracked in noise | `linear_chirp`, `pulse_compress`, `barker_code`, `locate_pulse`, `PulseCorrelationTracker` | [Pulse compression and tracking](radar.md) |
| Stability and structure of a linear system; feedback control; state estimation | `StateSpace`, `PidController`, `KalmanFilter` | [Control](controls.md) |
| A delay line with random access; double buffering | `AddressableShiftRegister`, `PingPongBuffer` | [Streaming buffers](#streaming-buffers) |
