//! Sorting and selection algorithms.

pub mod quicksort;
pub mod heapsort;
pub mod select_m_largest;
pub mod equivalence_classes;
pub mod shells_method;

pub use quicksort::quicksort;
pub use heapsort::heapsort;
pub use select_m_largest::select_m_largest;
pub use equivalence_classes::equivalence_classes;
pub use shells_method::shell_sort;
