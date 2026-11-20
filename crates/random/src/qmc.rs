//! Tools for quasi-Monte Carlo sampling, following `scipy.stats.qmc`: Latin hypercube samples,
//! the discrepancy of a point set (how far it is from uniform), and scaling between the unit cube
//! and a box. The low-discrepancy sequences themselves, plain or scrambled, are
//! [`SobolSequence`](crate::quasi_random::SobolSequence) and
//! [`HaltonSequence`](crate::quasi_random::HaltonSequence).
//!
//! See the book entry [Quasi-random sequences](https://jwkunz.github.io/lib_jsl/book/random/quasi-random.html).

use crate::uniform_generator::UniformGenerator;
use lib_jsl_core::ErrorsJSL;

/// A discrepancy measure (SciPy's `method`). All are `L²` discrepancies with closed forms that
/// cost `O(n² d)`; each is the worst-case integration error, over a unit ball of functions, of
/// the equal-weight rule on the points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Discrepancy {
    /// Centered `L²` discrepancy (`'CD'`, SciPy's default): boxes with a corner at the nearest
    /// vertex of the cube. Invariant under reflections `x_k → 1 − x_k`.
    Centered,
    /// Wrap-around `L²` discrepancy (`'WD'`): boxes that wrap around the torus. Invariant under
    /// shifts modulo 1.
    WrapAround,
    /// Mixture discrepancy (`'MD'`), which combines the two and avoids some of their blind spots.
    Mixture,
    /// The `L²`-star discrepancy (`'L2-star'`): boxes anchored at the origin, Warnock's formula.
    /// Returned as the discrepancy itself; the others are returned squared, as SciPy does.
    L2Star,
}

// ANCHOR: discrepancy
/// The discrepancy of `points` in `[0, 1]^d` (SciPy's `qmc.discrepancy`): smaller is more uniform.
pub fn discrepancy(points: &[Vec<f64>], method: Discrepancy) -> Result<f64, ErrorsJSL> {
    let n = points.len();
    let d = points.first().map_or(0, |p| p.len());
    if n == 0 || d == 0 || points.iter().any(|p| p.len() != d) {
        return Err(ErrorsJSL::InvalidInputRange("discrepancy: need a non-empty set of points of one dimension"));
    }
    if points.iter().flatten().any(|x| !(0.0..=1.0).contains(x)) {
        return Err(ErrorsJSL::InvalidInputRange("discrepancy: the points must lie in the unit cube"));
    }
    let nf = n as f64;
    let df = d as i32;
    // Σ_i Π_k f(x_ik) and Σ_i Σ_j Π_k g(x_ik, x_jk).
    let single = |f: &dyn Fn(f64) -> f64| points.iter().map(|p| p.iter().map(|&x| f(x)).product::<f64>()).sum::<f64>();
    let pairs = |g: &dyn Fn(f64, f64) -> f64| {
        let mut total = 0.0;
        for p in points {
            for q in points {
                total += p.iter().zip(q).map(|(&x, &y)| g(x, y)).product::<f64>();
            }
        }
        total
    };
    Ok(match method {
        Discrepancy::Centered => {
            let s1 = single(&|x| 1.0 + 0.5 * (x - 0.5).abs() - 0.5 * (x - 0.5).powi(2));
            let s2 = pairs(&|x, y| 1.0 + 0.5 * (x - 0.5).abs() + 0.5 * (y - 0.5).abs() - 0.5 * (x - y).abs());
            (13.0f64 / 12.0).powi(df) - 2.0 / nf * s1 + s2 / (nf * nf)
        }
        Discrepancy::WrapAround => {
            let s2 = pairs(&|x, y| {
                let t = (x - y).abs();
                1.5 - t * (1.0 - t)
            });
            -(4.0f64 / 3.0).powi(df) + s2 / (nf * nf)
        }
        Discrepancy::Mixture => {
            let s1 = single(&|x| 5.0 / 3.0 - 0.25 * (x - 0.5).abs() - 0.25 * (x - 0.5).powi(2));
            let s2 = pairs(&|x, y| 15.0 / 8.0 - 0.25 * (x - 0.5).abs() - 0.25 * (y - 0.5).abs() - 0.75 * (x - y).abs() + 0.5 * (x - y).powi(2));
            (19.0f64 / 12.0).powi(df) - 2.0 / nf * s1 + s2 / (nf * nf)
        }
        Discrepancy::L2Star => {
            let s1 = single(&|x| 1.0 - x * x);
            let s2 = pairs(&|x, y| 1.0 - x.max(y));
            (3.0f64.powi(-df) - 2.0f64.powi(1 - df) / nf * s1 + s2 / (nf * nf)).max(0.0).sqrt()
        }
    })
}
// ANCHOR_END: discrepancy

// ANCHOR: latin_hypercube
/// `n` points of a Latin hypercube in `[0, 1)^d` (SciPy's `qmc.LatinHypercube`): in every
/// coordinate, each of the `n` intervals `[i/n, (i+1)/n)` holds exactly one point. Each coordinate
/// takes an independent random permutation of the intervals; the point is placed uniformly in its
/// cell, or at the cell's centre when `centered` (SciPy's `scramble=False`).
pub fn latin_hypercube<G: UniformGenerator>(n: usize, d: usize, centered: bool, rng: &mut G) -> Vec<Vec<f64>> {
    let mut points = vec![vec![0.0; d]; n];
    for k in 0..d {
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            perm.swap(i, j);
        }
        for (p, &cell) in points.iter_mut().zip(&perm) {
            let offset = if centered { 0.5 } else { rng.next_f64() };
            p[k] = (cell as f64 + offset) / n as f64;
        }
    }
    points
}
// ANCHOR_END: latin_hypercube

/// Maps points from the unit cube to the box `[lower, upper]` (SciPy's `qmc.scale`), or back
/// when `reverse`.
pub fn scale(points: &[Vec<f64>], lower: &[f64], upper: &[f64], reverse: bool) -> Result<Vec<Vec<f64>>, ErrorsJSL> {
    let d = lower.len();
    if upper.len() != d || points.iter().any(|p| p.len() != d) {
        return Err(ErrorsJSL::InvalidInputRange("scale: the bounds and points must have one dimension"));
    }
    if lower.iter().zip(upper).any(|(l, u)| l.partial_cmp(u) != Some(std::cmp::Ordering::Less)) {
        return Err(ErrorsJSL::InvalidInputRange("scale: need lower < upper in every coordinate"));
    }
    Ok(points
        .iter()
        .map(|p| p.iter().zip(lower.iter().zip(upper)).map(|(&x, (&l, &u))| if reverse { (x - l) / (u - l) } else { l + x * (u - l) }).collect())
        .collect())
}


/// How [`PoissonDisk`] draws candidates around a centre (SciPy's `hypersphere`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Hypersphere {
    /// Uniformly in the ball of radius `2r` (the default).
    Volume,
    /// On the sphere of radius `1.001 r`, which packs more densely.
    Surface,
}

/// Poisson-disk sampling in a box (SciPy's `qmc.PoissonDisk`): random points no two of which are
/// closer than `radius`, by Bridson's algorithm. A background grid with cells of side `r/√d`
/// holds at most one point each, so the test for nearby points looks only at the surrounding
/// cells; new points are drawn as candidates around a random active point and kept if no
/// existing point is within `r`.
#[derive(Clone, Debug)]
pub struct PoissonDisk {
    d: usize,
    radius: f64,
    hypersphere: Hypersphere,
    ncandidates: usize,
    lower: Vec<f64>,
    upper: Vec<f64>,
    cell: f64,
    grid_size: Vec<usize>,
    grid: Vec<Option<usize>>,
    points: Vec<Vec<f64>>,
    pool: Vec<usize>,
}

impl PoissonDisk {
    /// A sampler in `[lower, upper]` (the unit cube by default, `None`), with SciPy's defaults
    /// `ncandidates = 30` and [`Hypersphere::Volume`].
    pub fn new(d: usize, radius: f64, hypersphere: Hypersphere, ncandidates: usize, bounds: Option<(&[f64], &[f64])>) -> Result<Self, ErrorsJSL> {
        if d == 0 || radius.is_nan() || radius <= 0.0 || ncandidates == 0 {
            return Err(ErrorsJSL::InvalidInputRange("PoissonDisk: need d ≥ 1, radius > 0 and at least one candidate"));
        }
        let (lower, upper) = match bounds {
            Some((l, u)) if l.len() == d && u.len() == d && l.iter().zip(u).all(|(a, b)| a < b) => (l.to_vec(), u.to_vec()),
            Some(_) => return Err(ErrorsJSL::InvalidInputRange("PoissonDisk: bounds must have d entries with lower < upper")),
            None => (vec![0.0; d], vec![1.0; d]),
        };
        let cell = radius / (d as f64).sqrt();
        let grid_size: Vec<usize> = (0..d).map(|k| ((upper[k] - lower[k]) / cell).ceil().max(1.0) as usize).collect();
        let cells = grid_size.iter().try_fold(1usize, |acc, &g| acc.checked_mul(g)).filter(|&c| c <= 50_000_000).ok_or(ErrorsJSL::InvalidInputRange("PoissonDisk: the radius is too small for the grid"))?;
        Ok(PoissonDisk { d, radius, hypersphere, ncandidates, lower, upper, cell, grid_size, grid: vec![None; cells], points: Vec::new(), pool: Vec::new() })
    }

    fn cell_of(&self, p: &[f64]) -> Vec<usize> {
        (0..self.d).map(|k| (((p[k] - self.lower[k]) / self.cell) as usize).min(self.grid_size[k] - 1)).collect()
    }

    fn flat(&self, idx: &[usize]) -> usize {
        idx.iter().zip(&self.grid_size).fold(0, |acc, (&i, &g)| acc * g + i)
    }

    fn too_close(&self, p: &[f64]) -> bool {
        let idx = self.cell_of(p);
        let lo: Vec<usize> = idx.iter().map(|&i| i.saturating_sub(2)).collect();
        let hi: Vec<usize> = idx.iter().zip(&self.grid_size).map(|(&i, &g)| (i + 3).min(g)).collect();
        let mut cur = lo.clone();
        loop {
            if let Some(j) = self.grid[self.flat(&cur)] {
                let d2: f64 = self.points[j].iter().zip(p).map(|(a, b)| (a - b) * (a - b)).sum();
                if d2 < self.radius * self.radius {
                    return true;
                }
            }
            let mut k = self.d;
            loop {
                if k == 0 {
                    return false;
                }
                k -= 1;
                cur[k] += 1;
                if cur[k] < hi[k] {
                    break;
                }
                cur[k] = lo[k];
            }
        }
    }

    fn add(&mut self, p: Vec<f64>, out: &mut Vec<Vec<f64>>) {
        let f = self.flat(&self.cell_of(&p));
        self.grid[f] = Some(self.points.len());
        self.pool.push(self.points.len());
        self.points.push(p.clone());
        out.push(p);
    }

    fn normal<G: UniformGenerator>(rng: &mut G) -> f64 {
        let u = rng.next_f64().max(f64::MIN_POSITIVE);
        let v = rng.next_f64();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }

    // ANCHOR: poisson_disk
    /// Up to `n` new points (fewer when the box is full).
    pub fn random<G: UniformGenerator>(&mut self, n: usize, rng: &mut G) -> Vec<Vec<f64>> {
        let mut out = Vec::new();
        if n == 0 {
            return out;
        }
        if self.points.is_empty() {
            let p: Vec<f64> = (0..self.d).map(|k| self.lower[k] + rng.next_f64() * (self.upper[k] - self.lower[k])).collect();
            self.add(p, &mut out);
        }
        while !self.pool.is_empty() && out.len() < n {
            let at = (rng.next_u64() % self.pool.len() as u64) as usize;
            let centre = self.points[self.pool.swap_remove(at)].clone();
            for _ in 0..self.ncandidates {
                let dir: Vec<f64> = (0..self.d).map(|_| Self::normal(rng)).collect();
                let norm = dir.iter().map(|v| v * v).sum::<f64>().sqrt().max(f64::MIN_POSITIVE);
                let r = match self.hypersphere {
                    Hypersphere::Volume => 2.0 * self.radius * rng.next_f64().powf(1.0 / self.d as f64),
                    Hypersphere::Surface => 1.001 * self.radius,
                };
                let cand: Vec<f64> = centre.iter().zip(&dir).map(|(c, v)| c + r * v / norm).collect();
                if cand.iter().enumerate().any(|(k, &v)| v < self.lower[k] || v > self.upper[k]) || self.too_close(&cand) {
                    continue;
                }
                self.add(cand, &mut out);
                if out.len() >= n {
                    break;
                }
            }
        }
        out
    }
    // ANCHOR_END: poisson_disk

    /// Points until no more fit (SciPy's `fill_space`).
    pub fn fill_space<G: UniformGenerator>(&mut self, rng: &mut G) -> Vec<Vec<f64>> {
        self.random(usize::MAX, rng)
    }

    /// Forget all points (SciPy's `reset`).
    pub fn reset(&mut self) {
        self.grid.iter_mut().for_each(|c| *c = None);
        self.points.clear();
        self.pool.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quasi_random::{HaltonSequence, SobolSequence};
    use crate::uniform_generator::UniformRNG;

    #[test]
    fn discrepancy_of_small_sets() {
        // One point at the centre of the unit interval: the centered discrepancy is
        // 13/12 − 2(1) + 1 = 1/12; the L²-star discrepancy is √(1/3 − 3/4 + 1/2) = √(1/12).
        let p = vec![vec![0.5]];
        assert!((discrepancy(&p, Discrepancy::Centered).unwrap() - 1.0 / 12.0).abs() < 1e-15);
        assert!((discrepancy(&p, Discrepancy::L2Star).unwrap() - (1.0f64 / 12.0).sqrt()).abs() < 1e-15);
        assert!(discrepancy(&[vec![1.5]], Discrepancy::Centered).is_err());
        assert!(discrepancy(&[], Discrepancy::Centered).is_err());
    }

    #[test]
    fn scrambled_sequences_beat_random_points() {
        let mut rng = UniformRNG::from_seed(7);
        let n = 256;
        let random: Vec<Vec<f64>> = (0..n).map(|_| (0..4).map(|_| rng.next_f64()).collect()).collect();
        let mut sobol = SobolSequence::scrambled(4, &mut rng).unwrap();
        let sobol_pts: Vec<Vec<f64>> = (0..n).map(|_| sobol.next_point()).collect();
        let mut halton = HaltonSequence::scrambled(4, &mut rng).unwrap();
        let halton_pts: Vec<Vec<f64>> = (0..n).map(|_| halton.next_point()).collect();
        let lhs = latin_hypercube(n, 4, false, &mut rng);
        let cd = |p: &[Vec<f64>]| discrepancy(p, Discrepancy::Centered).unwrap();
        assert!(cd(&sobol_pts) < cd(&random) / 10.0, "{} {}", cd(&sobol_pts), cd(&random));
        assert!(cd(&halton_pts) < cd(&random) / 10.0);
        assert!(cd(&lhs) < cd(&random));
        // Scrambled Sobol keeps the net property: each 1-D projection of 256 points is stratified.
        for k in 0..4 {
            let mut cells: Vec<usize> = sobol_pts.iter().map(|p| (p[k] * 256.0) as usize).collect();
            cells.sort_unstable();
            assert!(cells.iter().enumerate().all(|(i, &c)| c == i));
            let mut cells: Vec<usize> = lhs.iter().map(|p| (p[k] * 256.0) as usize).collect();
            cells.sort_unstable();
            assert!(cells.iter().enumerate().all(|(i, &c)| c == i));
        }
        assert!(sobol_pts.iter().chain(&halton_pts).flatten().all(|&x| (0.0..1.0).contains(&x)));
        let boxed = scale(&lhs, &[-1.0; 4], &[3.0; 4], false).unwrap();
        let back = scale(&boxed, &[-1.0; 4], &[3.0; 4], true).unwrap();
        assert!(back.iter().flatten().zip(lhs.iter().flatten()).all(|(a, b)| (a - b).abs() < 1e-15));
    }

    #[test]
    fn poisson_disk_spacing_and_coverage() {
        let mut rng = crate::uniform_generator::UniformRNG::from_seed(5);
        for hs in [Hypersphere::Volume, Hypersphere::Surface] {
            let mut pd = PoissonDisk::new(2, 0.05, hs, 30, None).unwrap();
            let pts = pd.fill_space(&mut rng);
            for i in 0..pts.len() {
                for j in 0..i {
                    let d = ((pts[i][0] - pts[j][0]).powi(2) + (pts[i][1] - pts[j][1]).powi(2)).sqrt();
                    assert!(d >= 0.05, "{d}");
                }
            }
            // A maximal packing: every point of a fine grid is within 2r of some sample.
            for a in 0..20 {
                for b in 0..20 {
                    let q = [(a as f64 + 0.5) / 20.0, (b as f64 + 0.5) / 20.0];
                    assert!(pts.iter().any(|p| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt() < 0.1));
                }
            }
            assert!(pts.len() > 150, "{}", pts.len());
        }
    }
}
