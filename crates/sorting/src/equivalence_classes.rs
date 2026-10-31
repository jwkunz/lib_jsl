//! Equivalence classes.

/// Computes equivalence classes given an equivalence relation.
pub fn equivalence_classes<T, F>(_items: &[T], _equiv: F) -> Vec<Vec<T>>
where
    T: Clone,
    F: Fn(&T, &T) -> bool,
{
    todo!("equivalence_classes is not yet implemented")
}
