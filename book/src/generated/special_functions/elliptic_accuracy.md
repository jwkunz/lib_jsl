| function | test points | median relative error | largest relative error | at arguments |
|---|---:|---:|---:|---|
| `elliptic_integral` $K(k)$ | 6 | 1e-16 | 7e-13 | (0.999999) |
| `elliptic_integral_second` $E(k)$ | 6 | 2e-16 | 3e-15 | (0.999999) |
| `elliptic_integral_third` $\Pi(n,k)$ | 18 | 2e-16 | 9e-13 | (0.9, 0.999999) |
| `elliptic_f` $F(\varphi,k)$ | 30 | 1e-16 | 7e-13 | (3, 0.999999) |
| `elliptic_e` $E(\varphi,k)$ | 30 | 1e-16 | 3e-15 | (3, 0.999999) |
| `elliptic_pi` $\Pi(\varphi,n,k)$ | 30 | 1e-16 | 7e-13 | (3, 0.3, 0.999999) |
| `jacobian_elliptic` sn | 33 | 0 | 4e-15 | (5, 0.9) |
| `jacobian_elliptic` cn | 30 | 0 | 8e-15 | (5, 0.999) |
| `jacobian_elliptic` dn | 33 | 0 | 8e-16 | (-3, 0.999) |
