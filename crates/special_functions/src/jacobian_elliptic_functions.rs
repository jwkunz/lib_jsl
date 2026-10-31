//! Jacobian elliptic functions.

/// The Jacobian elliptic functions `(sn(u | m), cn(u | m), dn(u | m))` with parameter `m = k²`
/// for `0 <= m <= 1`.
///
/// Computed with the descending Landen / arithmetic-geometric mean scheme (Abramowitz & Stegun
/// 16.4), with series fallbacks at the endpoints `m → 0` and `m → 1`. Returns `NaN`s for `m`
/// outside `[0, 1]`.
pub fn jacobian_elliptic(u: f64, m: f64) -> (f64, f64, f64) {
    const EPS: f64 = 1e-16;
    if u.is_nan() || m.is_nan() || !(0.0..=1.0).contains(&m) {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    if m < 1e-9 {
        let (s, c) = u.sin_cos();
        let t = 0.25 * m * (u - s * c);
        return (s - t * c, c + t * s, 1.0 - 0.5 * m * s * s);
    }
    if m > 1.0 - 1e-9 {
        let m1 = 1.0 - m;
        let (t, sech) = (u.tanh(), 1.0 / u.cosh());
        let shch = u.sinh() * u.cosh();
        return (
            t + 0.25 * m1 * (shch - u) * sech * sech,
            sech - 0.25 * m1 * (shch - u) * t * sech,
            sech + 0.25 * m1 * (shch + u) * t * sech,
        );
    }
    let mut a = vec![1.0];
    let mut c = vec![m.sqrt()];
    let mut b = (1.0 - m).sqrt();
    while c.last().unwrap().abs() > EPS && a.len() < 40 {
        let an = *a.last().unwrap();
        let cn = 0.5 * (an - b);
        let anext = 0.5 * (an + b);
        b = (an * b).sqrt();
        a.push(anext);
        c.push(cn);
    }
    let n = a.len() - 1;
    let mut phi = (1u64 << n.min(62)) as f64 * a[n] * u;
    let mut phi1 = phi; // φ₁, the iterate just before the final backward step
    for i in (1..=n).rev() {
        if i == 1 {
            phi1 = phi;
        }
        // t = 2 φ_{i-1} - φ_i satisfies sin t = (c_i / a_i) sin φ_i and |t| < π/2.
        let t = (c[i] / a[i] * phi.sin()).asin();
        phi = 0.5 * (phi + t);
    }
    let phi0 = phi;
    let sn = phi0.sin();
    let cn = phi0.cos();
    let dn = cn / (phi1 - phi0).cos();
    (sn, cn, dn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        let cases = [
            (0.5, 0.5, 0.47075047365565728, 0.882_266_394_890_440_3, 0.942_972_425_777_385_7),
            (1.0, 0.9, 0.770_085_724_907_880_7, 0.637_940_417_510_212_1, 0.682_840_522_130_748_6),
            (3.0, 0.3, 0.411_422_660_634_000_6, -0.911_444_674_303_843_8, 0.974_278_922_226_613_7),
            (10.0, 0.99, -0.991_420_744_860_682_1, -0.130_709_244_737_663_4, 0.16405504440062638),
            (2.0, 1e-12, 0.909_297_426_825_929_2, -0.41614683654660172, 0.999_999_999_999_586_6),
            (1.5, 0.2, 0.989_808_228_380_719_8, 0.14240670991853183, 0.896_691_660_609_132_7),
        ];
        for &(u, m, sn, cn, dn) in &cases {
            let (s, c, d) = jacobian_elliptic(u, m);
            assert!((s - sn).abs() < 1e-14, "sn({u}|{m}) = {s} vs {sn}");
            assert!((c - cn).abs() < 1e-14, "cn({u}|{m}) = {c} vs {cn}");
            assert!((d - dn).abs() < 1e-14, "dn({u}|{m}) = {d} vs {dn}");
        }
    }

    #[test]
    fn identities() {
        for &m in &[0.0, 0.2, 0.7, 0.999, 1.0] {
            for &u in &[-3.0, 0.0, 0.4, 2.2, 7.5] {
                let (s, c, d) = jacobian_elliptic(u, m);
                assert!((s * s + c * c - 1.0).abs() < 1e-12, "m={m} u={u}");
                assert!((d * d + m * s * s - 1.0).abs() < 1e-12, "m={m} u={u}");
            }
        }
    }

    #[test]
    fn limits_and_period() {
        let (s, c, d) = jacobian_elliptic(0.7, 0.0);
        assert!((s - 0.7f64.sin()).abs() < 1e-15 && (c - 0.7f64.cos()).abs() < 1e-15 && d == 1.0);
        let (s, c, d) = jacobian_elliptic(0.7, 1.0);
        assert!((s - 0.7f64.tanh()).abs() < 1e-15);
        assert!((c - 1.0 / 0.7f64.cosh()).abs() < 1e-15 && (d - c).abs() < 1e-15);
        // sn(K) = 1, cn(K) = 0
        let m: f64 = 0.6;
        let kk = crate::elliptic_integrals::elliptic_integral(m.sqrt());
        let (s, c, _) = jacobian_elliptic(kk, m);
        assert!((s - 1.0).abs() < 1e-14 && c.abs() < 1e-14);
        // Period 4K for sn.
        let (s1, _, _) = jacobian_elliptic(0.3, m);
        let (s2, _, _) = jacobian_elliptic(0.3 + 4.0 * kk, m);
        assert!((s1 - s2).abs() < 1e-12);
        assert!(jacobian_elliptic(1.0, 1.5).0.is_nan());
    }
}
