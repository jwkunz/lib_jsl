| problem (SciPy reference) | result |
|---|---:|
| cubature of exp(-r^2) on [0, 1]^3 (`integrate.cubature`) | relative difference 9e-15 |
| cubature of cos(x + 2y) on [0, 1] x [0, 2] (`integrate.cubature`) | relative difference 4e-15 |
| van der Pol, mu = 1, state at t = 10: Dormand-Prince (`solve_ivp`, DOP853) | largest difference 1e-11 |
| van der Pol, mu = 100, state at t = 100 (`solve_ivp`, Radau) | Rosenbrock 4e-7, BDF 6e-8 |
| Robertson's reactions, state at t = 1000 (`solve_ivp`, Radau) | relative: Rosenbrock 5e-11, BDF 4e-8 |
| Bratu's problem y'' + e^y = 0, solution at 11 points: collocation (`solve_bvp`) | largest difference 4e-15 |
