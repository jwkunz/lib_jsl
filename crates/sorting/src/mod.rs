//! Sorting and selection algorithms.

/// Sorts a slice using quicksort.
pub fn quicksort<T: Ord>(_slice: &mut [T]) {
    todo!("quicksort is not yet implemented")
}

/// Sorts a slice using heapsort.
pub fn heapsort<T: Ord>(_slice: &mut [T]) {
    todo!("heapsort is not yet implemented")
}

/// Selects the M largest elements from a slice.
pub fn select_m_largest<T: Ord + Clone>(_slice: &[T], _m: usize) -> Vec<T> {
    todo!("select_m_largest is not yet implemented")
}

/// Computes equivalence classes given an equivalence relation.
pub fn equivalence_classes<T, F>(_items: &[T], _equiv: F) -> Vec<Vec<T>>
where
    T: Clone,
    F: Fn(&T, &T) -> bool,
{
    todo!("equivalence_classes is not yet implemented")
}

/// Sorts a slice using Shell's method.
pub fn shell_sort<T: Ord>(_slice: &mut [T]) {
    todo!("shell_sort is not yet implemented")
}
