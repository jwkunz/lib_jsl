//! Machine learning algorithms.

pub mod supervised {
    pub mod regression {
        pub fn least_squares(_x: &[Vec<f64>], _y: &[f64]) -> Vec<f64> {
            todo!("least_squares is not yet implemented")
        }

        pub fn general_least_squares(_x: &[Vec<f64>], _y: &[f64], _w: &[f64]) -> Vec<f64> {
            todo!("general_least_squares is not yet implemented")
        }

        pub fn ridge_regression(_x: &[Vec<f64>], _y: &[f64], _alpha: f64) -> Vec<f64> {
            todo!("ridge_regression is not yet implemented")
        }

        pub fn elastic_net(_x: &[Vec<f64>], _y: &[f64], _alpha: f64, _l1_ratio: f64) -> Vec<f64> {
            todo!("elastic_net is not yet implemented")
        }
    }

    pub mod classification {
        pub fn k_nearest_neighbors(_x_train: &[Vec<f64>], _y_train: &[usize], _x_test: &[f64], _k: usize) -> usize {
            todo!("k_nearest_neighbors is not yet implemented")
        }

        pub fn random_forest() {
            todo!("random_forest is not yet implemented")
        }

        pub fn logistic_regression() {
            todo!("logistic_regression is not yet implemented")
        }

        pub fn naive_bayes() {
            todo!("naive_bayes is not yet implemented")
        }
    }
}

pub mod unsupervised {
    pub mod clustering {
        pub fn k_means(_data: &[Vec<f64>], _k: usize) -> Vec<usize> {
            todo!("k_means is not yet implemented")
        }

        pub fn dbscan(_data: &[Vec<f64>], _eps: f64, _min_samples: usize) -> Vec<i32> {
            todo!("dbscan is not yet implemented")
        }
    }

    pub mod reduction {
        pub fn pca(_data: &[Vec<f64>], _n_components: usize) -> Vec<Vec<f64>> {
            todo!("pca is not yet implemented")
        }

        pub fn tsne(_data: &[Vec<f64>], _n_components: usize) -> Vec<Vec<f64>> {
            todo!("tsne is not yet implemented")
        }
    }
}
