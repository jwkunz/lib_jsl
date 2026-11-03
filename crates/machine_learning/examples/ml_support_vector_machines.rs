//! Kernel support vector machines for classification and regression.

use lib_jsl_core::R2D;
use lib_jsl_machine_learning::common::{Classifier, Regressor, Transformer};
use lib_jsl_machine_learning::datasets::{make_circles, make_sine};
use lib_jsl_machine_learning::kernels::Kernel;
use lib_jsl_machine_learning::metrics::{accuracy, r2_score};
use lib_jsl_machine_learning::preprocessing::StandardScaler;
use lib_jsl_machine_learning::supervised::classification::SupportVectorMachine;
use lib_jsl_machine_learning::supervised::regression::SupportVectorRegression;

fn main() {
    // Concentric circles: no line separates them, but an RBF kernel does.
    let (x, labels) = make_circles(100, 0.1, 0.5, 1);
    let (x_test, test_labels) = make_circles(500, 0.1, 0.5, 2);
    for (name, kernel) in [("linear", Kernel::Linear), ("polynomial, degree 2", Kernel::Polynomial { degree: 2, gamma: 1.0, coef0: 1.0 }), ("RBF, γ = 2", Kernel::Rbf { gamma: 2.0 })] {
        let mut svm = SupportVectorMachine::new(kernel, 1.0);
        svm.fit(x.view(), &labels).unwrap();
        println!("{name:<22} accuracy {:.3} with {} support vectors", accuracy(&test_labels, &svm.predict(x_test.view()).unwrap()).unwrap(), svm.support_vector_count());
    }
    // Regression: standardise x, then fit an ε-tube of width 0.1.
    let (x, y) = make_sine(150, 0.1, 3);
    let mut scaler = StandardScaler::new();
    let xs: R2D = scaler.fit_transform(x.view()).unwrap();
    let mut svr = SupportVectorRegression::new(Kernel::Rbf { gamma: 1.0 }, 10.0, 0.1).with_limits(20_000, 1e-8);
    svr.fit(xs.view(), y.view()).unwrap();
    println!("SVR: R² {:.3} with {} of 150 points as support vectors", r2_score(y.view(), svr.predict(xs.view()).unwrap().view()).unwrap(), svr.support_vector_count());
}
