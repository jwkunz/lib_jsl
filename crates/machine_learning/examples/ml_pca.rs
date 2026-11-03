//! Principal component analysis and kernel PCA.

use lib_jsl_core::R2D;
use lib_jsl_machine_learning::common::{MlRng, Transformer};
use lib_jsl_machine_learning::datasets::make_circles;
use lib_jsl_machine_learning::kernels::Kernel;
use lib_jsl_machine_learning::unsupervised::reduction::{KernelPca, Pca};

fn main() {
    // Five measured features driven by two hidden factors, plus noise.
    let mut rng = MlRng::new(1);
    let z = R2D::from_shape_fn((300, 2), |_| rng.gaussian(0.0, 1.0));
    let mix = ndarray::arr2(&[[3.0, 1.0, 0.0, 2.0, -1.0], [0.0, 2.0, 1.0, -1.0, 1.0]]);
    let x = z.dot(&mix) + R2D::from_shape_fn((300, 5), |_| rng.gaussian(0.0, 0.3));
    let mut pca = Pca::new(5);
    pca.fit(x.view()).unwrap();
    println!("explained variance ratio {:.4?}", pca.explained_variance_ratio());
    let mut two = Pca::new(2);
    let scores = two.fit_transform(x.view()).unwrap();
    let back = two.inverse_transform(scores.view()).unwrap();
    let rms = ((&back - &x).mapv(|v| v * v).mean().unwrap()).sqrt();
    println!("two components: reconstruction RMS error {rms:.3} (noise level 0.3)");
    println!("first axis {:.3}", two.components().row(0));

    // Rings: the third RBF kernel PCA component separates them.
    let (rings, labels) = make_circles(100, 0.03, 0.45, 1);
    let mut kp = KernelPca::new(Kernel::Rbf { gamma: 2.0 }, 3);
    let k = kp.fit_transform(rings.view()).unwrap();
    let mean = |c: usize| k.column(2).iter().zip(&labels).filter(|(_, l)| **l == c).map(|(v, _)| *v).sum::<f64>() / 100.0;
    println!("kernel PCA eigenvalues {:.2?}; component 3 means: outer {:.3}, inner {:.3}", kp.eigenvalues(), mean(0), mean(1));
}
