pub use crate as dsp;
pub use lib_jsl_core as prelude;
pub use lib_jsl_ffts as ffts;
pub use lib_jsl_number_theory as number_theory;
pub use lib_jsl_random as random;

pub mod continuous;

pub mod discrete {
    pub mod convolve;
    pub mod cross_correlation;
    pub mod sinc;
    pub mod stream_operator;
    pub mod windows;

    pub mod filters {
        pub mod fir {
            pub mod boxcar_integrator;
            pub mod derivative_filter;
            pub mod firwin;
            pub mod firwin2;
            pub mod overlap_and_add_fir;
            pub mod remez;
        }

        pub mod iir {
            pub mod bessel;
            pub mod biquad;
            pub mod butterworth;
            pub mod chebyshev1;
            pub mod chebyshev2;
            pub mod discrete_linear_filter;
            pub mod elliptic;
            pub mod iir_comb;
            pub mod iir_notch;
            pub mod iir_peak;
        }

        pub use fir::boxcar_integrator;
        pub use fir::derivative_filter;
        pub use fir::firwin;
        pub use fir::firwin2;
        pub use fir::overlap_and_add_fir;
        pub use fir::remez;
        pub use iir::bessel;
        pub use iir::biquad;
        pub use iir::butterworth;
        pub use iir::chebyshev1;
        pub use iir::chebyshev2;
        pub use iir::discrete_linear_filter;
        pub use iir::elliptic;
        pub use iir::iir_comb;
        pub use iir::iir_notch;
        pub use iir::iir_peak;
    }

    pub mod transformations {
        pub mod channel_impairment;
        pub mod channel_receiver;
        pub mod channel_transmitter;
        pub mod frequency_mixer;
        pub mod non_linearity_transformers;
    }

    pub mod resampling {
        pub mod polyphase_arbitrary_resampling;
        pub mod polyphase_integer_resampling;
    }

    pub mod controls {
        pub mod kalman;
        pub mod pid;
        pub mod state_space;
    }

    pub mod spectral {
        pub mod chirpz;
        pub mod critically_sampled_polyphase_filter_bank;
        pub mod freqz;
        pub mod oversampled_polyphase_filter_bank;
    }
}
