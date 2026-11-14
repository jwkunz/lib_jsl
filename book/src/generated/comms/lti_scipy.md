| computation (SciPy reference) | largest difference |
|---|---:|
| `tf2ss`, third order with a second-order numerator | 0 |
| `ss2tf`, 3 states, input 0 | 4e-15 |
| `ss2tf`, 3 states, input 1 | 7e-15 |
| `lsim`, 3 states, 2 inputs, 201 steps, zero-order hold | 1e-15 |
| `lsim`, 3 states, 2 inputs, 201 steps, linear interpolation | 2e-15 |
| `lsim` starting at t = 0.5 | 2e-15 |
| `step` | 4e-15 |
| `impulse` | 4e-15 |
| `bode`, magnitude in dB | 2e-14 |
| `bode`, phase in degrees | 1e-13 |
| `cont2discrete`, zoh | 1e-16 |
| `cont2discrete`, foh | 1e-16 |
| `cont2discrete`, bilinear | 1e-16 |
| `cont2discrete`, euler | 0 |
| `cont2discrete`, backward_diff | 6e-17 |
| `cont2discrete`, gbt | 1e-16 |
| `cont2discrete`, impulse | 1e-16 |
| `place_poles`, one input: the gain | 2e-13 |
| `place_poles`, two inputs: the achieved poles | 1e-14 |
