| transform (SciPy reference) | largest relative difference |
|---|---:|
| complex DFT, 8 points (power of two) (`fft.fft`) | 2e-17 |
| complex DFT, 64 points (power of two) (`fft.fft`) | 2e-16 |
| complex DFT, 512 points (power of two) (`fft.fft`) | 2e-16 |
| complex DFT, 1000 points (Bluestein) (`fft.fft`) | 2e-16 |
| complex DFT, 997 points (prime, Bluestein) (`fft.fft`) | 4e-16 |
| inverse DFT, 64 points, scaled by 1/n (`fft.ifft`) | 2e-16 |
| inverse DFT, 997 points, scaled by 1/n (`fft.ifft`) | 7e-16 |
| real DFT, 16 points (`fft.rfft`) | 2e-16 |
| real DFT, 17 points (`fft.rfft`) | 2e-16 |
| 2D DFT, 8 x 12 (`fft.fft2`) | 3e-16 |
| 3D DFT, 4 x 6 x 5 (`fft.fftn`) | 4e-16 |
| fftshift, 9 points (`fft.fftshift`) | 0 |
