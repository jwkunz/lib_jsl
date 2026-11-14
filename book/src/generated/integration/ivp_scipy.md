| problem | method | rtol | steps (library / SciPy) | f evaluations (library / SciPy) | final error (library / SciPy) | method switches |
|---|---:|---:|---:|---:|---:|---:|
| Van der Pol, μ = 100 (stiff) | Radau | 1e-4 | 236 / 236 | 2152 / 2152 | 1e-6 / 1e-6 |  |
| Van der Pol, μ = 100 (stiff) | Radau | 1e-7 | 1202 / 1202 | 9586 / 9586 | 1e-10 / 1e-10 |  |
| Van der Pol, μ = 100 (stiff) | LSODA | 1e-4 | 532 / 519 | 1198 / 1035 | 3e-4 / 4e-4 | 5 |
| Van der Pol, μ = 100 (stiff) | LSODA | 1e-7 | 1085 / 1221 | 2282 / 2145 | 7e-7 / 6e-7 | 5 |
| Robertson kinetics (stiff) | Radau | 1e-4 | 62 / 62 | 535 / 535 | 3e-8 / 3e-8 |  |
| Robertson kinetics (stiff) | Radau | 1e-7 | 304 / 304 | 2319 / 2319 | 6e-11 / 6e-11 |  |
| Robertson kinetics (stiff) | LSODA | 1e-4 | 185 / 182 | 344 / 244 | 5e-6 / 5e-6 | 1 |
| Robertson kinetics (stiff) | LSODA | 1e-7 | 834 / 602 | 1678 / 924 | 2e-8 / 3e-9 | 1 |
| Arenstorf orbit (non-stiff) | Radau | 1e-4 | 181 / 181 | 1565 / 1461 | 9e-3 / 9e-3 |  |
| Arenstorf orbit (non-stiff) | Radau | 1e-7 | 995 / 995 | 8186 / 8158 | 2e-6 / 2e-6 |  |
| Arenstorf orbit (non-stiff) | LSODA | 1e-4 | 262 / 403 | 524 / 753 | 1e0 / 9e-1 | 0 |
| Arenstorf orbit (non-stiff) | LSODA | 1e-7 | 604 / 759 | 1185 / 1633 | 4e-3 / 3e-5 | 0 |
