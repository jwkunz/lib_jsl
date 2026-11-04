//! Finite-difference time-domain (FDTD) solvers on Yee grids.
//!
//! * [`Fdtd1D`]: `E_z`, `H_y` along x with per-cell permittivity, permeability and conductivity,
//!   and first-order Mur absorbing ends (exact at the magic time step, Courant number 1).
//! * [`FdtdTmz`]: the 2D transverse-magnetic mode (`E_z`, `H_x`, `H_y`) with a graded Berenger
//!   split-field perfectly matched layer around the domain, per-cell permittivity and perfect
//!   electric conductors.
//!
//! Fields use the normalised magnetic field `H̃ = η₀ H` so `E` and `H̃` share units, and the time
//! step is `Δt = S Δ / c` for Courant number `S` (`S ≤ 1` in 1D, `S ≤ 1/√2` in 2D).
//!
//! See the book entry [The finite-difference time-domain method](https://jwkunz.github.io/lib_jsl/book/em-waves/fdtd.html).

use lib_jsl_core::ErrorsJSL;
use lib_jsl_core::constants::em::{C0, EPS_0};

/// A one-dimensional FDTD grid.
#[derive(Debug, Clone)]
pub struct Fdtd1D {
    pub ez: Vec<f64>,
    /// `η₀ H_y`, located half a cell to the right of `ez[i]`.
    pub hy: Vec<f64>,
    eps_r: Vec<f64>,
    mu_r: Vec<f64>,
    sigma: Vec<f64>,
    courant: f64,
    dx: f64,
    steps: usize,
}

impl Fdtd1D {
    pub fn new(cells: usize, dx: f64, courant: f64) -> Result<Self, ErrorsJSL> {
        if cells < 3 || dx.is_nan() || dx <= 0.0 || !(courant > 0.0 && courant <= 1.0) {
            return Err(ErrorsJSL::InvalidInputRange("Need at least 3 cells, dx > 0 and 0 < S <= 1."));
        }
        Ok(Fdtd1D { ez: vec![0.0; cells], hy: vec![0.0; cells - 1], eps_r: vec![1.0; cells], mu_r: vec![1.0; cells - 1], sigma: vec![0.0; cells], courant, dx, steps: 0 })
    }

    pub fn time_step(&self) -> f64 {
        self.courant * self.dx / C0
    }

    pub fn time(&self) -> f64 {
        self.steps as f64 * self.time_step()
    }

    /// Sets the material of cells `start..end`: relative permittivity, conductivity (S/m) and
    /// relative permeability.
    pub fn set_material(&mut self, start: usize, end: usize, eps_r: f64, sigma: f64, mu_r: f64) {
        let end = end.min(self.ez.len());
        for i in start..end {
            self.eps_r[i] = eps_r;
            self.sigma[i] = sigma;
            if i < self.mu_r.len() {
                self.mu_r[i] = mu_r;
            }
        }
    }

    /// Advances one time step; `source` is added to `E_z` at its index (a soft source).
    pub fn step(&mut self, source: Option<(usize, f64)>) {
        // ANCHOR: fdtd1d
        let s = self.courant;
        let n = self.ez.len();
        for i in 0..n - 1 {
            self.hy[i] += s / self.mu_r[i] * (self.ez[i + 1] - self.ez[i]);
        }
        let (old_left, old_right) = (self.ez[1], self.ez[n - 2]);
        let dt = self.time_step();
        for i in 1..n - 1 {
            let loss = self.sigma[i] * dt / (2.0 * EPS_0 * self.eps_r[i]);
            self.ez[i] = (1.0 - loss) / (1.0 + loss) * self.ez[i] + s / (self.eps_r[i] * (1.0 + loss)) * (self.hy[i] - self.hy[i - 1]);
        }
        if let Some((i, value)) = source {
            self.ez[i] += value;
        }
        // First-order Mur boundaries with the local Courant number.
        let mur = |s_local: f64| (s_local - 1.0) / (s_local + 1.0);
        let (sl, sr) = (s / self.eps_r[0].sqrt(), s / self.eps_r[n - 1].sqrt());
        self.ez[0] = old_left + mur(sl) * (self.ez[1] - self.ez[0]);
        self.ez[n - 1] = old_right + mur(sr) * (self.ez[n - 2] - self.ez[n - 1]);
        // ANCHOR_END: fdtd1d
        self.steps += 1;
    }
}

/// A two-dimensional TMz FDTD grid with a perfectly matched layer.
#[derive(Debug, Clone)]
pub struct FdtdTmz {
    nx: usize,
    ny: usize,
    courant: f64,
    delta: f64,
    ezx: Vec<f64>,
    ezy: Vec<f64>,
    /// `η₀ H_x` at `(i, j + ½)`.
    pub hx: Vec<f64>,
    /// `η₀ H_y` at `(i + ½, j)`.
    pub hy: Vec<f64>,
    eps_r: Vec<f64>,
    pec: Vec<bool>,
    sigma_x: Vec<f64>,
    sigma_y: Vec<f64>,
    steps: usize,
}

impl FdtdTmz {
    /// An `nx × ny` grid of square cells of side `delta` with a PML of `pml_cells` on every side,
    /// graded polynomially (order 3) for a normal-incidence reflection of about 1e-6.
    pub fn new(nx: usize, ny: usize, delta: f64, courant: f64, pml_cells: usize) -> Result<Self, ErrorsJSL> {
        if nx < 2 * pml_cells + 3 || ny < 2 * pml_cells + 3 || delta.is_nan() || delta <= 0.0 || !(courant > 0.0 && courant <= std::f64::consts::FRAC_1_SQRT_2) {
            return Err(ErrorsJSL::InvalidInputRange("Need room inside the PML, delta > 0 and 0 < S <= 1/√2."));
        }
        // Normalised conductivity σΔt/(2ε₀) at depth d (cells) into the layer.
        let order = 3.0;
        let s_max = -(order + 1.0) * (1e-6f64).ln() * courant / (4.0 * pml_cells.max(1) as f64);
        let profile = |index: f64, cells: usize| -> f64 {
            if pml_cells == 0 {
                return 0.0;
            }
            let depth = if index < pml_cells as f64 { pml_cells as f64 - index } else if index > (cells - 1 - pml_cells) as f64 { index - (cells - 1 - pml_cells) as f64 } else { 0.0 };
            s_max * (depth / pml_cells as f64).powf(order)
        };
        let size = nx * ny;
        let mut sigma_x = vec![0.0; size];
        let mut sigma_y = vec![0.0; size];
        for j in 0..ny {
            for i in 0..nx {
                sigma_x[j * nx + i] = profile(i as f64, nx);
                sigma_y[j * nx + i] = profile(j as f64, ny);
            }
        }
        Ok(FdtdTmz { nx, ny, courant, delta, ezx: vec![0.0; size], ezy: vec![0.0; size], hx: vec![0.0; size], hy: vec![0.0; size], eps_r: vec![1.0; size], pec: vec![false; size], sigma_x, sigma_y, steps: 0 })
    }

    fn id(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }

    pub fn time_step(&self) -> f64 {
        self.courant * self.delta / C0
    }

    pub fn time(&self) -> f64 {
        self.steps as f64 * self.time_step()
    }

    /// `E_z` at node `(i, j)`.
    pub fn ez(&self, i: usize, j: usize) -> f64 {
        let k = self.id(i, j);
        self.ezx[k] + self.ezy[k]
    }

    /// Sets the relative permittivity in the box `[i0, i1) × [j0, j1)`.
    pub fn set_permittivity(&mut self, i0: usize, i1: usize, j0: usize, j1: usize, eps_r: f64) {
        for j in j0..j1.min(self.ny) {
            for i in i0..i1.min(self.nx) {
                let k = self.id(i, j);
                self.eps_r[k] = eps_r;
            }
        }
    }

    /// Makes the box `[i0, i1) × [j0, j1)` a perfect electric conductor (`E_z = 0`).
    pub fn set_pec(&mut self, i0: usize, i1: usize, j0: usize, j1: usize) {
        for j in j0..j1.min(self.ny) {
            for i in i0..i1.min(self.nx) {
                let k = self.id(i, j);
                self.pec[k] = true;
            }
        }
    }

    /// Advances one time step with an optional soft source added to `E_z` at `(i, j)`.
    pub fn step(&mut self, source: Option<(usize, usize, f64)>) {
        let (nx, ny, s) = (self.nx, self.ny, self.courant);
        // Magnetic fields with matched magnetic conductivity.
        for j in 0..ny - 1 {
            for i in 0..nx {
                let k = j * nx + i;
                let sy = 0.5 * (self.sigma_y[k] + self.sigma_y[k + nx]);
                let curl = self.ezx[k + nx] + self.ezy[k + nx] - self.ezx[k] - self.ezy[k];
                self.hx[k] = (1.0 - sy) / (1.0 + sy) * self.hx[k] - s / (1.0 + sy) * curl;
            }
        }
        for j in 0..ny {
            for i in 0..nx - 1 {
                let k = j * nx + i;
                let sx = 0.5 * (self.sigma_x[k] + self.sigma_x[k + 1]);
                let curl = self.ezx[k + 1] + self.ezy[k + 1] - self.ezx[k] - self.ezy[k];
                self.hy[k] = (1.0 - sx) / (1.0 + sx) * self.hy[k] + s / (1.0 + sx) * curl;
            }
        }
        // Split electric field; the outermost nodes stay zero (PEC backing the PML).
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let k = j * nx + i;
                if self.pec[k] {
                    self.ezx[k] = 0.0;
                    self.ezy[k] = 0.0;
                    continue;
                }
                let (sx, sy, e) = (self.sigma_x[k], self.sigma_y[k], self.eps_r[k]);
                self.ezx[k] = (1.0 - sx) / (1.0 + sx) * self.ezx[k] + s / (e * (1.0 + sx)) * (self.hy[k] - self.hy[k - 1]);
                self.ezy[k] = (1.0 - sy) / (1.0 + sy) * self.ezy[k] - s / (e * (1.0 + sy)) * (self.hx[k] - self.hx[k - nx]);
            }
        }
        if let Some((i, j, value)) = source {
            let k = self.id(i, j);
            self.ezx[k] += 0.5 * value;
            self.ezy[k] += 0.5 * value;
        }
        self.steps += 1;
    }

    /// The electromagnetic energy proxy `Σ (ε_r E_z² + H̃²)` over the interior (outside the PML).
    pub fn interior_energy(&self, pml_cells: usize) -> f64 {
        let mut total = 0.0;
        for j in pml_cells..self.ny - pml_cells {
            for i in pml_cells..self.nx - pml_cells {
                let k = self.id(i, j);
                total += self.eps_r[k] * (self.ezx[k] + self.ezy[k]).powi(2) + self.hx[k].powi(2) + self.hy[k].powi(2);
            }
        }
        total
    }
}

/// A Gaussian pulse `exp(−((n − delay)/width)²)` in time steps.
pub fn gaussian_pulse(step: usize, delay: f64, width: f64) -> f64 {
    (-((step as f64 - delay) / width).powi(2)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_dimensional_pulses_travel_at_c_and_leave_through_mur_boundaries() {
        let mut grid = Fdtd1D::new(400, 1e-3, 1.0).unwrap();
        for n in 0..60 {
            grid.step(Some((200, gaussian_pulse(n, 30.0, 8.0))));
        }
        // Two pulses moving one cell per step away from the source (peak injected at step 30).
        for _ in 0..100 {
            grid.step(None);
        }
        let right = (200..400).max_by(|&a, &b| grid.ez[a].abs().total_cmp(&grid.ez[b].abs())).unwrap();
        assert!((right as i64 - (200 + 130)).abs() <= 1, "{right}");
        assert!((grid.time() - 160.0 * 1e-3 / C0).abs() < 1e-18);
        for _ in 0..400 {
            grid.step(None);
        }
        let residual = grid.ez.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        // Mur boundaries are exact at S = 1; a soft source leaves only a tiny residue.
        assert!(residual < 1e-5, "{residual}");
    }

    #[test]
    fn dielectric_interface_reflects_with_the_fresnel_coefficient() {
        // ε_r = 4: reflection (1 − 2)/(1 + 2) = −1/3, transmission 2/3.
        let mut grid = Fdtd1D::new(1200, 1e-3, 0.5).unwrap();
        grid.set_material(600, 1200, 4.0, 0.0, 1.0);
        let (mut incident, mut reflected, mut transmitted) = (0.0f64, 0.0f64, 0.0f64);
        for n in 0..2000 {
            // A total-field source would be cleaner; record at a probe before and after reflection.
            grid.step(Some((300, gaussian_pulse(n, 120.0, 40.0))));
            if n < 700 {
                incident = incident.max(grid.ez[450].abs());
            } else {
                reflected = if grid.ez[450].abs() > reflected.abs() { grid.ez[450] } else { reflected };
            }
            transmitted = transmitted.max(grid.ez[800].abs());
        }
        // Grid dispersion in the slower medium costs a little transmitted peak amplitude.
        assert!((reflected / incident + 1.0 / 3.0).abs() < 0.03, "{}", reflected / incident);
        assert!((transmitted / incident - 2.0 / 3.0).abs() < 0.03, "{}", transmitted / incident);
        assert!(Fdtd1D::new(10, 1e-3, 1.5).is_err());
    }

    #[test]
    fn two_dimensional_pml_absorbs_an_outgoing_cylindrical_wave() {
        let (n, pml) = (120, 12);
        let mut small = FdtdTmz::new(n, n, 1e-3, 0.5, pml).unwrap();
        let big_n = 3 * n;
        let mut big = FdtdTmz::new(big_n, big_n, 1e-3, 0.5, pml).unwrap();
        let (c, bc) = (n / 2, big_n / 2);
        let (mut worst, mut peak, mut energy) = (0.0f64, 0.0f64, 0.0f64);
        for step in 0..260 {
            let pulse = gaussian_pulse(step, 40.0, 12.0);
            small.step(Some((c, c, pulse)));
            big.step(Some((bc, bc, pulse)));
            // A probe 40 cells from the source; the big grid is reflection-free for this window.
            let (a, b) = (small.ez(c + 40, c), big.ez(bc + 40, bc));
            worst = worst.max((a - b).abs());
            peak = peak.max(b.abs());
            energy = energy.max(small.interior_energy(pml));
            // Cylindrical symmetry.
            assert!((small.ez(c + 25, c) - small.ez(c, c + 25)).abs() < 1e-9 * (1.0 + peak));
        }
        assert!(worst < 0.01 * peak, "PML reflection {:.2e} of {:.2e}", worst, peak);
        for _ in 0..600 {
            small.step(None);
        }
        // What remains after the wave has left is a small fraction of the peak interior energy.
        assert!(small.interior_energy(pml) < 1e-3 * energy, "{} of {energy}", small.interior_energy(pml));
    }

    #[test]
    fn a_conducting_wall_reflects_with_a_phase_reversal() {
        let mut grid = FdtdTmz::new(200, 60, 1e-3, 0.5, 10);
        let g = grid.as_mut().unwrap();
        g.set_pec(150, 152, 0, 60);
        g.set_permittivity(0, 1, 0, 1, 1.0);
        let mut most_negative = 0.0f64;
        for step in 0..500 {
            // A line source across the guide approximates a plane wave.
            for j in 11..49 {
                let k = g.id(60, j);
                let pulse = gaussian_pulse(step, 40.0, 10.0);
                g.ezx[k] += 0.5 * pulse;
                g.ezy[k] += 0.5 * pulse;
            }
            g.step(None);
            if step > 200 {
                most_negative = most_negative.min(g.ez(100, 30));
            }
        }
        assert!(most_negative < -0.1, "{most_negative}");
        assert_eq!(g.ez(150, 30), 0.0);
        assert_eq!(g.dimensions(), (200, 60));
    }
}
