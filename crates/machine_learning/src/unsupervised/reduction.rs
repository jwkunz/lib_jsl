//! Dimensionality reduction.

pub mod principal_component_analysis;
pub mod kernel_pca;
pub mod t_sne;
pub mod umap;
pub mod lda;

pub use principal_component_analysis::Pca;
pub use kernel_pca::KernelPca;
pub use t_sne::Tsne;
pub use umap::umap;
pub use lda::Lda;
