| scipy.ndimage function and case | largest difference ÷ largest value |
|---|---:|
| binary_erosion cross | identical |
| binary_dilation cross | identical |
| binary_opening cross | identical |
| binary_closing cross | identical |
| binary_fill_holes cross | identical |
| binary_erosion square | identical |
| binary_dilation square | identical |
| binary_opening square | identical |
| binary_closing square | identical |
| binary_fill_holes square | identical |
| binary_erosion cross 2 iterations | identical |
| binary_dilation disk5 | identical |
| grey_erosion square reflect | identical |
| grey_dilation disk5 reflect | identical |
| grey_erosion square nearest | identical |
| grey_dilation disk5 nearest | identical |
| grey_erosion square mirror | identical |
| grey_dilation disk5 mirror | identical |
| grey_erosion square wrap | identical |
| grey_dilation disk5 wrap | identical |
| grey_erosion square constant | identical |
| grey_dilation disk5 constant | identical |
| grey_opening square | identical |
| grey_closing square | identical |
| white_tophat disk5 | 1.4e-16 |
| black_tophat disk5 | 1.4e-16 |
| morphological_gradient square | 1.3e-16 |
| distance_transform_edt | 1.6e-15 |
| distance_transform_edt sampling 2 0.5 | 1.8e-16 |
| distance_transform_cdt taxicab | identical |
| distance_transform_cdt chessboard | identical |
| label cross | identical |
| label cross sizes | identical |
| label cross sum | 1.2e-16 |
| label cross mean | 2.4e-15 |
| label cross minimum | identical |
| label cross maximum | identical |
| label cross center_of_mass | 2.0e-15 |
| label cross find_objects | identical |
| label square | identical |
| label square sizes | identical |
| label square sum | 1.2e-16 |
| label square mean | 2.4e-15 |
| label square minimum | identical |
| label square maximum | identical |
| label square center_of_mass | 2.0e-15 |
| label square find_objects | identical |
| median_filter 3x3 reflect | identical |
| median_filter 3x3 nearest | identical |
| median_filter 3x3 mirror | identical |
| median_filter 3x3 wrap | identical |
| median_filter 3x3 constant | identical |
| median_filter 4x4 | identical |
| median_filter 2x5 | identical |
| percentile_filter 25 5x5 | identical |
| percentile_filter 100 3x3 | identical |
| minimum_filter 3x2 | identical |
| maximum_filter 2x3 | identical |
| rank_filter -2 disk5 | identical |
| gaussian_filter 1.5 reflect | 1.4e-15 |
| gaussian_filter 1.5 nearest | 1.4e-15 |
| gaussian_filter 1.5 mirror | 1.4e-15 |
| gaussian_filter 1.5 wrap | 1.7e-15 |
| gaussian_filter 1.5 constant | 1.8e-15 |
| gaussian_filter (1, 2.5) order (0, 1) | 4.1e-15 |
| gaussian_filter 2 order (2, 0) | 1.3e-15 |
| gaussian_filter 1.2 order (1, 1) | 8.8e-16 |
| gaussian_laplace 1.5 | 2.9e-15 |
| gaussian_gradient_magnitude 1.5 | 3.1e-15 |
| laplace | 4.3e-16 |
| map_coordinates order 0 constant | identical |
| map_coordinates order 1 constant | 4.2e-17 |
| map_coordinates order 2 constant | 3.3e-16 |
| map_coordinates order 3 constant | 5.9e-16 |
| map_coordinates order 4 constant | 1.5e-15 |
| map_coordinates order 5 constant | 9.2e-16 |
| map_coordinates order 0 nearest | identical |
| map_coordinates order 1 nearest | 1.7e-16 |
| map_coordinates order 2 nearest | 7.2e-16 |
| map_coordinates order 3 nearest | 5.3e-15 |
| map_coordinates order 4 nearest | 4.3e-12 |
| map_coordinates order 5 nearest | 2.1e-10 |
| map_coordinates order 0 mirror | identical |
| map_coordinates order 1 mirror | 5.0e-16 |
| map_coordinates order 2 mirror | 6.7e-16 |
| map_coordinates order 3 mirror | 8.4e-16 |
| map_coordinates order 4 mirror | 1.5e-15 |
| map_coordinates order 5 mirror | 9.2e-16 |
| map_coordinates order 0 reflect | identical |
| map_coordinates order 1 reflect | 1.7e-16 |
| map_coordinates order 2 reflect | 7.5e-16 |
| map_coordinates order 3 reflect | 1.2e-15 |
| map_coordinates order 4 reflect | 7.8e-16 |
| map_coordinates order 5 reflect | 1.0e-15 |
| map_coordinates order 0 wrap | identical |
| map_coordinates order 1 wrap | 7.5e-16 |
| map_coordinates order 2 wrap | 8.4e-16 |
| map_coordinates order 3 wrap | 1.2e-15 |
| map_coordinates order 4 wrap | 7.5e-15 |
| map_coordinates order 5 wrap | 3.2e-15 |
| affine_transform shear order 3 | 1.5e-15 |
| shift (2.5, -1.25) order 1 nearest | 1.7e-16 |
| shift (2.5, -1.25) order 3 | 1.8e-15 |
| zoom (1.5, 0.75) order 3 | 1.4e-15 |
| zoom (0.5, 2) order 1 | 8.5e-16 |
| rotate 30 reshape order 3 | 1.4e-15 |
| rotate -20 order 1 nearest | 1.9e-15 |
| rotate 30 reshape order 3 shape | identical |
