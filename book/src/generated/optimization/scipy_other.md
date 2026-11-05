| problem (SciPy reference) | result |
|---|---:|
| Rastrigin's function, 3 variables (`differential_evolution`) | library f = 7.1e-15 at the global minimum; SciPy f = 0.9950, a local minimum (seed 1) |
| exponential decay fit, weighted (`curve_fit`) | parameters 2e-9, standard errors 4e-7 (relative) |
| exponential decay fit, unweighted (`curve_fit`) | parameters 1e-8, standard errors 8e-7 (relative) |
| diet problem, 3 variables (`linprog`, HiGHS) | solution 4e-16, cost 0 |
| random program, 10 variables, 8 constraints (`linprog`, HiGHS) | solution 9e-15, objective 4e-15 |
| largest box of surface area 6 (`minimize`, SLSQP) | SQP 2e-8, augmented Lagrangian 2e-8 in the solution; objective 5e-12 |
| Hock-Schittkowski 71 (`minimize`, SLSQP) | SQP 2e-8, augmented Lagrangian 7e-8 in the solution; objective 5e-9 |
| 0/1 knapsack, 10 items (`milp`, HiGHS) | value 309 (SciPy 309), same items: true |
