//! Mathematical morphology: erosion, dilation, opening, closing and hole filling for binary and
//! greyscale images, following `scipy.ndimage`.
//!
//! Morphology probes an image with a small shape, the **structuring element**. Erosion keeps a
//! pixel only if the element, centred there, fits inside the foreground; dilation turns a pixel on
//! if the element touches the foreground. Combinations of the two remove small objects (opening),
//! fill small gaps (closing), and extract outlines and peaks (gradients and top-hats). For
//! greyscale images, "fits inside" becomes a minimum and "touches" a maximum over the element.
//!
//! See the book entry [Morphology, distance transforms and labelling](https://jwkunz.github.io/lib_jsl/book/image/morphology.html).

use std::collections::VecDeque;

use lib_jsl_core::ErrorsJSL;
use ndarray::Array2;

use super::boundary::{Mode, sample};

/// The 3 × 3 cross: the four edge neighbours and the centre (4-connectivity; SciPy's default).
pub fn cross() -> Array2<bool> {
    Array2::from_shape_fn((3, 3), |(i, j)| i == 1 || j == 1)
}

/// The full 3 × 3 square (8-connectivity).
pub fn square(size: usize) -> Array2<bool> {
    Array2::from_elem((size, size), true)
}

/// A disc of the given radius, `(2r + 1) × (2r + 1)`.
pub fn disk(radius: usize) -> Array2<bool> {
    let r = radius as isize;
    Array2::from_shape_fn((2 * radius + 1, 2 * radius + 1), |(i, j)| {
        let (di, dj) = (i as isize - r, j as isize - r);
        di * di + dj * dj <= r * r
    })
}

/// The offsets `(di, dj)` of the true elements of a structuring element, relative to its centre.
fn offsets(structure: &Array2<bool>) -> Result<Vec<(isize, isize)>, ErrorsJSL> {
    let (h, w) = structure.dim();
    if h == 0 || w == 0 || h % 2 == 0 || w % 2 == 0 {
        return Err(ErrorsJSL::InvalidInputRange("structuring elements must have odd, non-zero dimensions"));
    }
    let (ch, cw) = ((h / 2) as isize, (w / 2) as isize);
    Ok(structure.indexed_iter().filter(|(_, b)| **b).map(|((i, j), _)| (i as isize - ch, j as isize - cw)).collect())
}

fn binary_pass(image: &Array2<bool>, offsets: &[(isize, isize)], erode: bool, border: bool) -> Array2<bool> {
    let (rows, cols) = image.dim();
    let at = |r: isize, c: isize| {
        if r < 0 || c < 0 || r >= rows as isize || c >= cols as isize { border } else { image[(r as usize, c as usize)] }
    };
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        let (r, c) = (r as isize, c as isize);
        if erode {
            offsets.iter().all(|&(di, dj)| at(r + di, c + dj))
        } else {
            // Dilation uses the reflected element, as in SciPy.
            offsets.iter().any(|&(di, dj)| at(r - di, c - dj))
        }
    })
}

fn repeat(image: &Array2<bool>, structure: &Array2<bool>, iterations: usize, erode: bool) -> Result<Array2<bool>, ErrorsJSL> {
    let offsets = offsets(structure)?;
    let mut out = binary_pass(image, &offsets, erode, false);
    let mut k = 1;
    // `iterations == 0` repeats until nothing changes.
    while iterations == 0 || k < iterations {
        let next = binary_pass(&out, &offsets, erode, false);
        if next == out {
            break;
        }
        out = next;
        k += 1;
    }
    Ok(out)
}

// ANCHOR: binary
/// Binary erosion: a pixel stays on only if every element of `structure`, centred on it, lies on
/// the foreground. Pixels beyond the image count as background, so objects touching the edge are
/// eroded from it (SciPy's `binary_erosion` with `border_value=0`). `iterations` repeats the
/// operation; `0` repeats until the image stops changing.
pub fn binary_erosion(image: &Array2<bool>, structure: &Array2<bool>, iterations: usize) -> Result<Array2<bool>, ErrorsJSL> {
    repeat(image, structure, iterations, true)
}

/// Binary dilation: a pixel turns on if the reflected `structure`, centred on it, touches the
/// foreground (SciPy's `binary_dilation`).
pub fn binary_dilation(image: &Array2<bool>, structure: &Array2<bool>, iterations: usize) -> Result<Array2<bool>, ErrorsJSL> {
    repeat(image, structure, iterations, false)
}
// ANCHOR_END: binary

/// Opening: erosion then dilation. Removes objects (and protrusions) smaller than the element,
/// and leaves larger ones nearly unchanged.
pub fn binary_opening(image: &Array2<bool>, structure: &Array2<bool>, iterations: usize) -> Result<Array2<bool>, ErrorsJSL> {
    binary_dilation(&binary_erosion(image, structure, iterations)?, structure, iterations)
}

/// Closing: dilation then erosion. Fills gaps and holes smaller than the element.
pub fn binary_closing(image: &Array2<bool>, structure: &Array2<bool>, iterations: usize) -> Result<Array2<bool>, ErrorsJSL> {
    binary_erosion(&binary_dilation(image, structure, iterations)?, structure, iterations)
}

// ANCHOR: fill_holes
/// Fills the holes of a binary image: background regions that cannot be reached from outside the
/// image by steps allowed by `structure` (the cross gives 4-connected steps, the square
/// 8-connected). A breadth-first search from the background pixels next to the border marks the
/// outside; everything else becomes foreground (SciPy's `binary_fill_holes`).
pub fn binary_fill_holes(image: &Array2<bool>, structure: &Array2<bool>) -> Result<Array2<bool>, ErrorsJSL> {
    let steps: Vec<(isize, isize)> = offsets(structure)?.into_iter().filter(|&d| d != (0, 0)).collect();
    let (rows, cols) = image.dim();
    let inside = |r: isize, c: isize| r >= 0 && c >= 0 && r < rows as isize && c < cols as isize;
    let mut outside = Array2::from_elem((rows, cols), false);
    let mut queue = VecDeque::new();
    for r in 0..rows {
        for c in 0..cols {
            if image[(r, c)] {
                continue;
            }
            // A background pixel is reached from beyond the edge if one of its steps leaves the image.
            if steps.iter().any(|&(di, dj)| !inside(r as isize - di, c as isize - dj)) {
                outside[(r, c)] = true;
                queue.push_back((r, c));
            }
        }
    }
    while let Some((r, c)) = queue.pop_front() {
        for &(di, dj) in &steps {
            let (nr, nc) = (r as isize + di, c as isize + dj);
            if inside(nr, nc) {
                let (nr, nc) = (nr as usize, nc as usize);
                if !image[(nr, nc)] && !outside[(nr, nc)] {
                    outside[(nr, nc)] = true;
                    queue.push_back((nr, nc));
                }
            }
        }
    }
    Ok(outside.mapv(|o| !o))
}
// ANCHOR_END: fill_holes

fn grey_pass(image: &Array2<f64>, footprint: &Array2<bool>, mode: Mode, erode: bool) -> Result<Array2<f64>, ErrorsJSL> {
    let offsets = offsets(footprint)?;
    if image.is_empty() {
        return Err(ErrorsJSL::InvalidInputRange("image must be non-empty"));
    }
    Ok(Array2::from_shape_fn(image.dim(), |(r, c)| {
        let (r, c) = (r as isize, c as isize);
        if erode {
            offsets.iter().map(|&(di, dj)| sample(image, r + di, c + dj, mode)).fold(f64::INFINITY, f64::min)
        } else {
            offsets.iter().map(|&(di, dj)| sample(image, r - di, c - dj, mode)).fold(f64::NEG_INFINITY, f64::max)
        }
    }))
}

// ANCHOR: grey
/// Greyscale erosion: the minimum over the footprint centred on each pixel (SciPy's
/// `grey_erosion` with a flat footprint; the default mode there is `Reflect`).
pub fn grey_erosion(image: &Array2<f64>, footprint: &Array2<bool>, mode: Mode) -> Result<Array2<f64>, ErrorsJSL> {
    grey_pass(image, footprint, mode, true)
}

/// Greyscale dilation: the maximum over the reflected footprint (SciPy's `grey_dilation`).
pub fn grey_dilation(image: &Array2<f64>, footprint: &Array2<bool>, mode: Mode) -> Result<Array2<f64>, ErrorsJSL> {
    grey_pass(image, footprint, mode, false)
}
// ANCHOR_END: grey

/// Greyscale opening (erosion then dilation): removes bright features smaller than the footprint.
pub fn grey_opening(image: &Array2<f64>, footprint: &Array2<bool>, mode: Mode) -> Result<Array2<f64>, ErrorsJSL> {
    grey_dilation(&grey_erosion(image, footprint, mode)?, footprint, mode)
}

/// Greyscale closing (dilation then erosion): removes dark features smaller than the footprint.
pub fn grey_closing(image: &Array2<f64>, footprint: &Array2<bool>, mode: Mode) -> Result<Array2<f64>, ErrorsJSL> {
    grey_erosion(&grey_dilation(image, footprint, mode)?, footprint, mode)
}

/// The white top-hat, `image − opening`: the bright features smaller than the footprint, with the
/// background removed.
pub fn white_tophat(image: &Array2<f64>, footprint: &Array2<bool>, mode: Mode) -> Result<Array2<f64>, ErrorsJSL> {
    Ok(image - &grey_opening(image, footprint, mode)?)
}

/// The black top-hat, `closing − image`: the dark features smaller than the footprint.
pub fn black_tophat(image: &Array2<f64>, footprint: &Array2<bool>, mode: Mode) -> Result<Array2<f64>, ErrorsJSL> {
    Ok(&grey_closing(image, footprint, mode)? - image)
}

/// The morphological gradient, `dilation − erosion`: large on edges, zero in flat regions.
pub fn morphological_gradient(image: &Array2<f64>, footprint: &Array2<bool>, mode: Mode) -> Result<Array2<f64>, ErrorsJSL> {
    Ok(&grey_dilation(image, footprint, mode)? - &grey_erosion(image, footprint, mode)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from_rows(rows: &[&str]) -> Array2<bool> {
        Array2::from_shape_fn((rows.len(), rows[0].len()), |(i, j)| rows[i].as_bytes()[j] == b'#')
    }

    #[test]
    fn opening_removes_a_speck_and_closing_fills_a_hole() {
        let img = from_rows(&["........", ".####...", ".####..#", ".####...", ".##.#...", "........"]);
        let open = binary_opening(&img, &cross(), 1).unwrap();
        assert!(!open[(2, 7)], "the isolated pixel is removed");
        let holed = from_rows(&["......", ".####.", ".#.##.", ".####.", "......"]);
        let close = binary_closing(&holed, &cross(), 1).unwrap();
        assert!(close[(2, 2)], "the one-pixel hole is filled");
    }

    #[test]
    fn holes_are_filled_but_open_bays_are_not() {
        let img = from_rows(&["#####.", "#...#.", "#.#.#.", "#...#.", "##.##.", "......"]);
        let filled = binary_fill_holes(&img, &cross()).unwrap();
        // The interior is connected to the outside through the gap at (4, 2): not a hole.
        assert!(!filled[(1, 1)]);
        let closed = from_rows(&["#####", "#...#", "#...#", "#####"]);
        assert!(binary_fill_holes(&closed, &cross()).unwrap().iter().all(|&b| b));
    }

    #[test]
    fn grey_erosion_and_dilation_are_minimum_and_maximum() {
        let img = Array2::from_shape_fn((5, 5), |(i, j)| (i * 5 + j) as f64);
        let e = grey_erosion(&img, &square(3), Mode::Reflect).unwrap();
        let d = grey_dilation(&img, &square(3), Mode::Reflect).unwrap();
        assert_eq!(e[(2, 2)], 6.0);
        assert_eq!(d[(2, 2)], 18.0);
        assert_eq!(e[(0, 0)], 0.0);
        assert_eq!(d[(4, 4)], 24.0);
    }
}
