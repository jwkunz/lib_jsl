//! Link budgets, path loss models, the two-ray model, atmospheric attenuation and the radar
//! equation.
//!
//! Run with `cargo run --release -p lib_jsl_electromagnetics --example propagation`.

use lib_jsl_electromagnetics::propagation::atmosphere::{gaseous_slant_attenuation, rain_specific_attenuation};
use lib_jsl_electromagnetics::propagation::multipath::{fresnel_zone_radius, two_ray_crossover_distance};
use lib_jsl_electromagnetics::propagation::path_loss::{HataEnvironment, LinkBudget, free_space_path_loss_db, okumura_hata_db};
use lib_jsl_electromagnetics::propagation::radar_equation::{Radar, range_resolution, unambiguous_range};

fn main() {
    // A 2.4 GHz link: 20 dBm, 6 dBi antennas at both ends, 3 dB of cable loss.
    let link = LinkBudget { transmit_power_dbm: 20.0, transmit_gain_dbi: 6.0, receive_gain_dbi: 6.0, other_losses_db: 3.0 };
    println!("2.4 GHz link: free-space loss at 1 km {:.1} dB, received {:.1} dBm", free_space_path_loss_db(1e3, 2.4e9), link.received_power_dbm(free_space_path_loss_db(1e3, 2.4e9)));
    println!("  range for a -85 dBm receiver: {:.1} km", link.free_space_range(-85.0, 2.4e9) / 1e3);
    println!("  first Fresnel zone radius at mid-path: {:.2} m", fresnel_zone_radius(1, 500.0, 500.0, 2.4e9));
    println!("cellular, 900 MHz, 5 km: Hata urban {:.1} dB, suburban {:.1} dB; two-ray crossover {:.1} km",
        okumura_hata_db(900.0, 40.0, 1.5, 5.0, HataEnvironment::Urban).unwrap(),
        okumura_hata_db(900.0, 40.0, 1.5, 5.0, HataEnvironment::Suburban).unwrap(),
        two_ray_crossover_distance(40.0, 1.5, 900e6) / 1e3);
    // A 20 GHz satellite downlink at 30° elevation, through 7.5 g/m³ of water vapour, and in rain.
    println!("20 GHz at 30° elevation: gases {:.2} dB; 25 mm/h rain (k = 0.0751, α = 1.099) adds {:.2} dB/km",
        gaseous_slant_attenuation(20.0, 7.5, 30f64.to_radians()).unwrap(), rain_specific_attenuation(25.0, 0.0751, 1.099));
    let radar = Radar { transmit_power: 1e6, gain: 10f64.powf(3.5), frequency: 3e9, bandwidth: 1e6, noise_figure: 2.0, losses: 3.16, noise_temperature: 290.0 };
    println!("S-band radar: 1 m² detected (13 dB) to {:.0} km; range resolution {:.0} m; unambiguous range at 1 kHz PRF {:.0} km",
        radar.maximum_range(1.0, 20.0, 1.0).unwrap() / 1e3, range_resolution(radar.bandwidth), unambiguous_range(1e3) / 1e3);
}
