//! Figures and tables for the machine learning chapter on clustering and dimensionality reduction.

use crate::linear_algebra::time;
use crate::svg::{Legend, Plot, Series, Style};
use crate::write;
use lib_jsl_core::R2D;
use lib_jsl_machine_learning::common::{Classifier, Clusterer, MlRng, Transformer, pairwise_distances, squared_distance};
use lib_jsl_machine_learning::datasets::{make_blobs, make_circles, make_classification, make_moons, make_swiss_roll};
use lib_jsl_machine_learning::kernels::Kernel;
use lib_jsl_machine_learning::metrics::{accuracy, adjusted_rand_index, silhouette_score};
use lib_jsl_machine_learning::supervised::classification::{neighbor_joining, upgma};
use lib_jsl_machine_learning::unsupervised::clustering::{Affinity, CovarianceType, Dbscan, Dendrogram, GaussianMixture, Hdbscan, HierarchicalClustering, KMeans, KMedoids, Linkage, SpectralClustering, agglomerate};
use lib_jsl_machine_learning::unsupervised::reduction::{KernelPca, Lda, Pca, Tsne, Umap};
use ndarray::{Axis, concatenate};

const CHAPTER: &str = "unsupervised";

pub fn generate() {
    kmeans_figures();
    kmedoids_figures();
    hierarchical_figures();
    linkage_reference_table();
    phylogeny_table();
    density_figures();
    gmm_figures();
    spectral_figures();
    pca_figures();
    kernel_pca_figures();
    lda_figures();
    embedding_figures();
    comparison_table();
}

// ------------------------------------------------------------------ helpers

fn as_options(l: &[usize]) -> Vec<Option<usize>> {
    l.iter().map(|&c| Some(c)).collect()
}

/// Scatter of 2D points coloured by cluster, with noise (`None`) in grey.
fn cluster_plot(title: &str, x: &R2D, labels: &[Option<usize>]) -> Plot {
    let mut plot = Plot::new(title, "x₁", "x₂");
    let k = labels.iter().flatten().max().map_or(0, |m| m + 1);
    for c in 0..k {
        let pts = (0..x.nrows()).filter(|&i| labels[i] == Some(c)).map(|i| (x[[i, 0]], x[[i, 1]])).collect();
        plot.series.push(Series::new("", pts, Style::Markers, c % 6));
    }
    let noise: Vec<(f64, f64)> = (0..x.nrows()).filter(|&i| labels[i].is_none()).map(|i| (x[[i, 0]], x[[i, 1]])).collect();
    if !noise.is_empty() {
        plot.series.push(Series::new("noise", noise, Style::Markers, 7));
    }
    plot
}

fn md_table(header: &[&str], rows: &[Vec<String>]) -> String {
    let mut s = format!("| {} |\n|{}\n", header.join(" | "), header.iter().enumerate().map(|(i, _)| if i == 0 { "---|" } else { "---:|" }).collect::<String>());
    for r in rows {
        s += &format!("| {} |\n", r.join(" | "));
    }
    s
}

fn ari(a: &[Option<usize>], truth: &[usize]) -> f64 {
    adjusted_rand_index(a, &as_options(truth)).unwrap()
}

fn line_segment(plot: &mut Plot, a: (f64, f64), b: (f64, f64), color: usize, width_label: &str) {
    plot.series.push(Series::new(width_label, vec![a, b], Style::Line, color));
}

/// The 2σ ellipse of a 2×2 covariance centred at `m`.
fn ellipse(m: (f64, f64), c: &R2D) -> Vec<(f64, f64)> {
    let (a, b, d) = (c[[0, 0]], c[[0, 1]], c[[1, 1]]);
    let tr = 0.5 * (a + d);
    let disc = (0.25 * (a - d).powi(2) + b * b).sqrt();
    let (l1, l2) = (tr + disc, (tr - disc).max(0.0));
    let theta = 0.5 * (2.0 * b).atan2(a - d);
    (0..=80)
        .map(|k| {
            let t = 2.0 * std::f64::consts::PI * k as f64 / 80.0;
            let (u, v) = (2.0 * l1.sqrt() * t.cos(), 2.0 * l2.sqrt() * t.sin());
            (m.0 + u * theta.cos() - v * theta.sin(), m.1 + u * theta.sin() + v * theta.cos())
        })
        .collect()
}

/// Fraction of each point's `k` nearest neighbours in `high` that are also among its `k` nearest
/// in `low`.
fn neighbour_preservation(high: &R2D, low: &R2D, k: usize) -> f64 {
    let n = high.nrows();
    let knn = |x: &R2D, i: usize| {
        let ri = x.row(i).to_vec();
        let mut d: Vec<(usize, f64)> = (0..n).filter(|&j| j != i).map(|j| (j, squared_distance(&ri, &x.row(j).to_vec()))).collect();
        d.sort_by(|a, b| a.1.total_cmp(&b.1));
        d.into_iter().take(k).map(|(j, _)| j).collect::<Vec<_>>()
    };
    (0..n).map(|i| {
        let a = knn(high, i);
        let b = knn(low, i);
        a.iter().filter(|j| b.contains(j)).count() as f64 / k as f64
    }).sum::<f64>() / n as f64
}

// ------------------------------------------------------------------ k-means

fn four_blobs() -> (R2D, Vec<usize>) {
    make_blobs(60, &[vec![0.0, 0.0], vec![5.0, 0.5], vec![1.0, 5.0], vec![6.0, 5.5]], 1.0, 3)
}

fn kmeans_figures() {
    let (x, _) = four_blobs();
    // Lloyd's iterations from one k-means++ seeding: run with an increasing iteration limit.
    for (iters, file) in [(1usize, "kmeans_iter1.svg"), (3, "kmeans_iter3.svg"), (50, "kmeans_final.svg")] {
        let mut km = KMeans::new(4, 11).with_restarts(1).with_limits(iters, 1e-8);
        let labels = km.fit_predict(x.view()).unwrap();
        let mut plot = cluster_plot(&format!("k-means after {} iteration{}", km.iterations(), if km.iterations() == 1 { "" } else { "s" }), &x, &labels);
        let c = km.centroids();
        plot.series.push(Series::new("centroids", (0..4).map(|i| (c[[i, 0]], c[[i, 1]])).collect(), Style::Markers, 6));
        plot.legend = Legend::BottomRight;
        plot.x_range = Some((-3.5, 9.5));
        plot.y_range = Some((-3.5, 9.0));
        write(CHAPTER, file, &plot.render());
    }
    // Choosing k: inertia and silhouette.
    let ks: Vec<usize> = (1..=9).collect();
    let mut inertia = Vec::new();
    let mut silhouette = Vec::new();
    for &k in &ks {
        let mut km = KMeans::new(k, 1).with_restarts(10);
        let labels = km.fit_predict(x.view()).unwrap();
        inertia.push((k as f64, km.inertia()));
        if k >= 2 {
            silhouette.push((k as f64, silhouette_score(x.view(), &labels).unwrap()));
        }
    }
    let mut plot = Plot::new("Inertia against the number of clusters (four true blobs)", "k", "inertia");
    plot.series.push(Series::new("inertia", inertia, Style::LineMarkers, 0));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "kmeans_elbow.svg", &plot.render());
    let mut plot = Plot::new("Mean silhouette against the number of clusters", "k", "silhouette");
    plot.series.push(Series::new("silhouette", silhouette, Style::LineMarkers, 1));
    write(CHAPTER, "kmeans_silhouette.svg", &plot.render());
    // Local optima: single runs from different seeds.
    let (x, _) = make_blobs(40, &(0..9).map(|i| vec![(i % 3) as f64 * 4.0, (i / 3) as f64 * 4.0]).collect::<Vec<_>>(), 0.9, 5);
    let best = {
        let mut km = KMeans::new(9, 1).with_restarts(200);
        km.fit_predict(x.view()).unwrap();
        km.inertia()
    };
    let mut rows = Vec::new();
    for restarts in [1usize, 5, 20] {
        let runs: Vec<f64> = (0..200u64)
            .map(|s| {
                let mut km = KMeans::new(9, 1000 + s).with_restarts(restarts);
                km.fit_predict(x.view()).unwrap();
                km.inertia()
            })
            .collect();
        let hit = runs.iter().filter(|&&v| v <= best * (1.0 + 1e-9)).count() as f64 / runs.len() as f64;
        let mean = runs.iter().sum::<f64>() / runs.len() as f64;
        let worst = runs.iter().copied().fold(0.0, f64::max);
        rows.push(vec![format!("{restarts}"), format!("{:.0}%", 100.0 * hit), format!("{:.3}", mean / best), format!("{:.3}", worst / best)]);
    }
    write(CHAPTER, "kmeans_restarts.md", &md_table(&["restarts (k-means++ seedings)", "runs reaching the best inertia", "mean inertia / best", "worst inertia / best"], &rows));
    // Failure on non-convex shapes.
    let (m, ml) = make_moons(150, 0.06, 2);
    let labels = KMeans::new(2, 1).fit_predict(m.view()).unwrap();
    let mut plot = cluster_plot(&format!("k-means on the moons: adjusted Rand index {:.2}", ari(&labels, &ml)), &m, &labels);
    plot.x_range = Some((-1.5, 2.5));
    plot.y_range = Some((-1.0, 1.5));
    write(CHAPTER, "kmeans_moons.svg", &plot.render());
}

fn kmedoids_figures() {
    let (blob, _) = make_blobs(40, &[vec![0.0, 0.0], vec![6.0, 0.0]], 0.8, 7);
    let outliers = ndarray::arr2(&[[3.0, 24.0], [2.5, 25.0], [3.5, 24.5]]);
    let x = concatenate(Axis(0), &[blob.view(), outliers.view()]).unwrap();
    let mut km = KMeans::new(2, 1);
    let lm = km.fit_predict(x.view()).unwrap();
    let mut kd = KMedoids::new(2);
    let ld = kd.fit_predict(x.view()).unwrap();
    for (title, labels, centres, file) in [
        ("k-means with three outliers", &lm, (0..2).map(|i| (km.centroids()[[i, 0]], km.centroids()[[i, 1]])).collect::<Vec<_>>(), "kmeans_outliers.svg"),
        ("k-medoids with three outliers", &ld, kd.medoids().iter().map(|&i| (x[[i, 0]], x[[i, 1]])).collect(), "kmedoids_outliers.svg"),
    ] {
        let mut plot = cluster_plot(title, &x, labels);
        plot.series.push(Series::new(if file.starts_with("kmeans") { "centroids" } else { "medoids" }, centres, Style::Markers, 6));
        plot.x_range = Some((-3.0, 9.0));
        plot.y_range = Some((-3.0, 27.0));
        plot.legend = Legend::TopRight;
        write(CHAPTER, file, &plot.render());
    }
    // Cost of PAM against k-means.
    let mut rows = Vec::new();
    for n in [100usize, 200, 400] {
        let (x, _) = make_blobs(n / 4, &[vec![0.0, 0.0], vec![5.0, 0.0], vec![0.0, 5.0], vec![5.0, 5.0]], 1.0, 1);
        let tk = time(|| {
            KMeans::new(4, 1).fit_predict(x.view()).unwrap();
        });
        let td = time(|| {
            KMedoids::new(4).fit_predict(x.view()).unwrap();
        });
        rows.push(vec![format!("{n}"), format!("{:.2}", tk * 1e3), format!("{:.1}", td * 1e3)]);
    }
    write(CHAPTER, "kmedoids_time.md", &md_table(&["points", "k-means, 5 restarts (ms)", "k-medoids, PAM (ms)"], &rows));
}

// ------------------------------------------------------------------ hierarchical clustering

/// Draws a dendrogram: leaves along x in tree order, merges at their heights.
fn dendrogram_plot(title: &str, tree: &Dendrogram, leaf_colour: &[usize]) -> Plot {
    let n = tree.points;
    let mut kids: Vec<(usize, usize)> = vec![(0, 0); n + tree.merges.len()];
    for (i, m) in tree.merges.iter().enumerate() {
        kids[n + i] = (m.a, m.b);
    }
    let root = n + tree.merges.len() - 1;
    let mut order = Vec::new();
    let mut stack = vec![root];
    while let Some(u) = stack.pop() {
        if u < n {
            order.push(u);
        } else {
            stack.push(kids[u].1);
            stack.push(kids[u].0);
        }
    }
    let mut xpos = vec![0.0; n + tree.merges.len()];
    let mut height = vec![0.0; n + tree.merges.len()];
    for (k, &leaf) in order.iter().enumerate() {
        xpos[leaf] = k as f64;
    }
    let mut plot = Plot::new(title, "leaves", "merge height");
    for (i, m) in tree.merges.iter().enumerate() {
        let id = n + i;
        xpos[id] = 0.5 * (xpos[m.a] + xpos[m.b]);
        height[id] = m.distance;
        let pts = vec![(xpos[m.a], height[m.a]), (xpos[m.a], m.distance), (xpos[m.b], m.distance), (xpos[m.b], height[m.b])];
        plot.series.push(Series::new("", pts, Style::Line, 6));
    }
    let k = leaf_colour.iter().max().map_or(0, |m| m + 1);
    for c in 0..k {
        let pts = (0..n).filter(|&i| leaf_colour[i] == c).map(|i| (xpos[i], 0.0)).collect();
        plot.series.push(Series::new("", pts, Style::Markers, c));
    }
    plot.x_range = Some((-1.0, n as f64));
    plot
}

fn hierarchical_figures() {
    let (x, l) = make_blobs(8, &[vec![0.0, 0.0], vec![5.0, 0.0], vec![2.5, 4.5]], 0.9, 4);
    for (lk, name, file) in [(Linkage::Single, "single", "dendrogram_single.svg"), (Linkage::Average, "average", "dendrogram_average.svg"), (Linkage::Ward, "Ward", "dendrogram_ward.svg")] {
        let tree = agglomerate(pairwise_distances(x.view()).view(), lk).unwrap();
        let plot = dendrogram_plot(&format!("Dendrogram, {name} linkage"), &tree, &l);
        write(CHAPTER, file, &plot.render());
    }
    let (m, ml) = make_moons(150, 0.06, 2);
    for (lk, name, file) in [(Linkage::Single, "single", "linkage_single_moons.svg"), (Linkage::Complete, "complete", "linkage_complete_moons.svg"), (Linkage::Ward, "Ward", "linkage_ward_moons.svg")] {
        let mut h = HierarchicalClustering::new(2, lk);
        let labels = h.fit_predict(m.view()).unwrap();
        let mut plot = cluster_plot(&format!("{name} linkage: ARI {:.2}", ari(&labels, &ml)), &m, &labels);
        plot.x_range = Some((-1.5, 2.5));
        plot.y_range = Some((-1.0, 1.5));
        write(CHAPTER, file, &plot.render());
    }
    // Single linkage chaining through a bridge of noise.
    let (blobs, bl) = make_blobs(40, &[vec![0.0, 0.0], vec![6.0, 0.0]], 0.7, 3);
    let bridge = R2D::from_shape_fn((20, 2), |(i, j)| if j == 0 { 1.1 + 0.2 * i as f64 } else { 0.05 * ((i * 7) % 3) as f64 });
    let xb = concatenate(Axis(0), &[blobs.view(), bridge.view()]).unwrap();
    let mut truth = bl.clone();
    truth.extend(std::iter::repeat_n(0, 20));
    let mut rows = Vec::new();
    for (lk, name) in [(Linkage::Single, "single"), (Linkage::Complete, "complete"), (Linkage::Average, "average"), (Linkage::Ward, "Ward")] {
        let moons = ari(&HierarchicalClustering::new(2, lk).fit_predict(m.view()).unwrap(), &ml);
        let labels = HierarchicalClustering::new(2, lk).fit_predict(xb.view()).unwrap();
        let blob_ari = adjusted_rand_index(&labels[..80], &as_options(&bl)).unwrap();
        rows.push(vec![name.to_string(), format!("{moons:.3}"), format!("{blob_ari:.3}")]);
    }
    write(CHAPTER, "linkage_table.md", &md_table(&["linkage", "two moons (ARI)", "two blobs joined by a bridge of 20 points (ARI on the blobs)"], &rows));
}

fn parse_rows(s: &str) -> R2D {
    let rows: Vec<Vec<f64>> = s.split(';').map(|r| r.split(',').map(|v| v.parse().unwrap()).collect()).collect();
    R2D::from_shape_fn((rows.len(), rows[0].len()), |(i, j)| rows[i][j])
}

fn linkage_reference_table() {
    let reference = include_str!("../data/cluster_reference.txt");
    let mut rows = Vec::new();
    for line in reference.lines() {
        let f: Vec<&str> = line.split('|').collect();
        let x = parse_rows(f[2]);
        let want: Vec<f64> = f[3].split(',').map(|v| v.parse().unwrap()).collect();
        match f[1] {
            "pca" => {
                let p = x.ncols();
                let mut pca = Pca::new(p);
                pca.fit(x.view()).unwrap();
                let var_err = pca.explained_variance().iter().zip(&want[..p]).map(|(a, b)| (a - b).abs() / b.abs()).fold(0.0, f64::max);
                // Only the two leading axes are well determined; the rest span the noise.
                let axes = pca.components();
                let axis_err = (0..2).flat_map(|i| (0..p).map(move |j| (i, j))).map(|(i, j)| (axes[[i, j]].abs() - want[p + i * p + j]).abs()).fold(0.0, f64::max);
                rows.push(vec![f[0].to_string(), "PCA (NumPy SVD)".to_string(), format!("{var_err:.1e}"), format!("{axis_err:.1e}")]);
            }
            method => {
                let lk = match method {
                    "single" => Linkage::Single,
                    "complete" => Linkage::Complete,
                    "average" => Linkage::Average,
                    _ => Linkage::Ward,
                };
                let tree = agglomerate(pairwise_distances(x.view()).view(), lk).unwrap();
                let got: Vec<f64> = tree.merges.iter().map(|m| m.distance).collect();
                let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs() / b.abs().max(1e-300)).fold(0.0, f64::max);
                rows.push(vec![f[0].to_string(), format!("{method} linkage (SciPy)"), format!("{err:.1e}"), "—".to_string()]);
            }
        }
    }
    write(CHAPTER, "reference.md", &md_table(&["data", "compared with", "merge heights or variances (largest relative difference)", "leading two principal axes (largest difference)"], &rows));
}

/// A random binary tree on `n` leaves with random branch lengths; returns its leaf distance matrix.
fn random_tree_distances(n: usize, clock: bool, rng: &mut MlRng) -> R2D {
    // Build by repeatedly joining random pairs of subtrees.
    let mut nodes: Vec<Vec<(usize, f64)>> = (0..n).map(|i| vec![(i, 0.0)]).collect(); // leaf -> depth below subtree root
    let mut height = vec![0.0f64; n];
    let mut d = R2D::zeros((n, n));
    while nodes.len() > 1 {
        let a = rng.below(nodes.len());
        let sa = nodes.swap_remove(a);
        let ha = height.swap_remove(a);
        let b = rng.below(nodes.len());
        let sb = nodes.swap_remove(b);
        let hb = height.swap_remove(b);
        let (la, lb, h) = if clock {
            let h = ha.max(hb) + rng.range(0.1, 1.0);
            (h - ha, h - hb, h)
        } else {
            (rng.range(0.05, 2.0), rng.range(0.05, 2.0), 0.0)
        };
        for &(i, di) in &sa {
            for &(j, dj) in &sb {
                d[[i, j]] = di + la + dj + lb;
                d[[j, i]] = d[[i, j]];
            }
        }
        let mut joined: Vec<(usize, f64)> = sa.iter().map(|&(i, di)| (i, di + la)).collect();
        joined.extend(sb.iter().map(|&(j, dj)| (j, dj + lb)));
        nodes.push(joined);
        height.push(h);
    }
    d
}

fn phylogeny_table() {
    let mut rng = MlRng::new(17);
    let names: Vec<String> = (0..12).map(|i| format!("t{i}")).collect();
    let labels: Vec<&str> = names.iter().map(String::as_str).collect();
    let mut rows = Vec::new();
    for (case, clock, noise) in [("molecular clock (ultrametric)", true, 0.0), ("unequal rates (additive)", false, 0.0), ("unequal rates, 5% noise", false, 0.05)] {
        let (mut eu, mut en) = (0.0f64, 0.0f64);
        let trials = 50;
        for _ in 0..trials {
            let truth = random_tree_distances(12, clock, &mut rng);
            let mut d = truth.clone();
            if noise > 0.0 {
                for i in 0..12 {
                    for j in i + 1..12 {
                        let v = d[[i, j]] * (1.0 + rng.gaussian(0.0, noise));
                        d[[i, j]] = v;
                        d[[j, i]] = v;
                    }
                }
            }
            let rel = |t: &R2D| (0..12).flat_map(|i| (i + 1..12).map(move |j| (i, j))).map(|(i, j)| (t[[i, j]] - truth[[i, j]]).abs() / truth[[i, j]]).fold(0.0, f64::max);
            eu = eu.max(rel(&upgma(d.view(), &labels).unwrap().patristic_distances()));
            en = en.max(rel(&neighbor_joining(d.view(), &labels).unwrap().patristic_distances()));
        }
        rows.push(vec![case.to_string(), format!("{eu:.1e}"), format!("{en:.1e}")]);
    }
    write(CHAPTER, "phylogeny.md", &md_table(&["true tree (12 leaves, 50 random trees)", "UPGMA", "neighbour joining"], &rows));
}

// ------------------------------------------------------------------ density-based clustering

fn two_densities() -> (R2D, Vec<usize>) {
    let (dense, _) = make_blobs(80, &[vec![0.0, 0.0], vec![2.0, 0.0]], 0.25, 1);
    let (sparse, _) = make_blobs(80, &[vec![7.0, 4.0]], 1.6, 2);
    let x = concatenate(Axis(0), &[dense.view(), sparse.view()]).unwrap();
    let mut l = vec![0; 80];
    l.extend(vec![1; 80]);
    l.extend(vec![2; 80]);
    (x, l)
}

fn density_figures() {
    let (m, _) = make_moons(150, 0.06, 2);
    let mut rng = MlRng::new(9);
    let noise = R2D::from_shape_fn((20, 2), |(_, j)| if j == 0 { rng.range(-1.5, 2.5) } else { rng.range(-1.0, 1.5) });
    let x = concatenate(Axis(0), &[m.view(), noise.view()]).unwrap();
    let mut db = Dbscan::new(0.2, 5);
    let labels = db.fit_predict(x.view()).unwrap();
    let core = db.core_points().to_vec();
    let mut plot = Plot::new("DBSCAN, ε = 0.2, min_samples = 5", "x₁", "x₂");
    let k = labels.iter().flatten().max().map_or(0, |m| m + 1);
    for c in 0..k {
        let pc = (0..x.nrows()).filter(|&i| labels[i] == Some(c) && core.contains(&i)).map(|i| (x[[i, 0]], x[[i, 1]])).collect();
        plot.series.push(Series::new(if c == 0 { "core" } else { "" }, pc, Style::Markers, c));
        let pb = (0..x.nrows()).filter(|&i| labels[i] == Some(c) && !core.contains(&i)).map(|i| (x[[i, 0]], x[[i, 1]])).collect();
        plot.series.push(Series::new(if c == 0 { "border" } else { "" }, pb, Style::Markers, 4));
    }
    plot.series.push(Series::new("noise", (0..x.nrows()).filter(|&i| labels[i].is_none()).map(|i| (x[[i, 0]], x[[i, 1]])).collect(), Style::Markers, 7));
    plot.x_range = Some((-1.5, 2.5));
    plot.y_range = Some((-1.0, 1.5));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "dbscan_moons.svg", &plot.render());
    // k-distance plot.
    let d = pairwise_distances(m.view());
    let mut kd: Vec<f64> = (0..m.nrows())
        .map(|i| {
            let mut r = d.row(i).to_vec();
            r.sort_by(f64::total_cmp);
            r[4]
        })
        .collect();
    kd.sort_by(|a, b| b.total_cmp(a));
    let mut plot = Plot::new("Distance to the 4th nearest neighbour, sorted (moons)", "points, sorted", "4-NN distance");
    plot.series.push(Series::new("", kd.iter().enumerate().map(|(i, &v)| (i as f64, v)).collect(), Style::Line, 0));
    write(CHAPTER, "dbscan_kdistance.svg", &plot.render());
    // Different densities: DBSCAN with two eps against HDBSCAN.
    let (x, l) = two_densities();
    for (eps, file) in [(0.3, "dbscan_density_small.svg"), (1.0, "dbscan_density_large.svg")] {
        let labels = Dbscan::new(eps, 5).fit_predict(x.view()).unwrap();
        let clusters = labels.iter().flatten().max().map_or(0, |m| m + 1);
        let noise = labels.iter().filter(|v| v.is_none()).count();
        let mut plot = cluster_plot(&format!("DBSCAN, ε = {eps}: {clusters} clusters, {noise} noise"), &x, &labels);
        plot.x_range = Some((-2.0, 12.0));
        plot.y_range = Some((-3.0, 9.0));
        plot.legend = Legend::BottomRight;
        write(CHAPTER, file, &plot.render());
    }
    let mut h = Hdbscan::new(15);
    let labels = h.fit_predict(x.view()).unwrap();
    let clusters = labels.iter().flatten().max().map_or(0, |m| m + 1);
    let noise = labels.iter().filter(|v| v.is_none()).count();
    let mut plot = cluster_plot(&format!("HDBSCAN, min_cluster_size = 15: {clusters} clusters, {noise} noise"), &x, &labels);
    plot.x_range = Some((-2.0, 12.0));
    plot.y_range = Some((-3.0, 9.0));
    plot.legend = Legend::BottomRight;
    write(CHAPTER, "hdbscan_density.svg", &plot.render());
    // Table: ARI on clustered points for an eps sweep, and HDBSCAN.
    let mut rows = Vec::new();
    for eps in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5] {
        let labels = Dbscan::new(eps, 5).fit_predict(x.view()).unwrap();
        rows.push(density_row(&format!("DBSCAN, ε = {eps}"), &labels, &l));
    }
    for mcs in [5usize, 15, 40] {
        let labels = Hdbscan::new(mcs).fit_predict(x.view()).unwrap();
        rows.push(density_row(&format!("HDBSCAN, min_cluster_size = {mcs}"), &labels, &l));
    }
    write(CHAPTER, "density_table.md", &md_table(&["method", "clusters", "noise points", "ARI (noise as its own group)"], &rows));
}

fn density_row(name: &str, labels: &[Option<usize>], truth: &[usize]) -> Vec<String> {
    let clusters = labels.iter().flatten().max().map_or(0, |m| m + 1);
    let noise = labels.iter().filter(|v| v.is_none()).count();
    vec![name.to_string(), format!("{clusters}"), format!("{noise}"), format!("{:.3}", ari(labels, truth))]
}

// ------------------------------------------------------------------ Gaussian mixtures

fn anisotropic() -> (R2D, Vec<usize>) {
    let (mut x, l) = make_blobs(120, &[vec![0.0, 0.0], vec![3.0, 3.0], vec![-1.0, 5.5]], 1.0, 8);
    for i in 0..x.nrows() {
        let (a, b) = (x[[i, 0]], x[[i, 1]]);
        x[[i, 0]] = 0.6 * a - 0.64 * b;
        x[[i, 1]] = -0.4 * a + 0.85 * b;
    }
    (x, l)
}

fn gmm_figures() {
    let (x, l) = anisotropic();
    let km = KMeans::new(3, 1).fit_predict(x.view()).unwrap();
    let mut plot = cluster_plot(&format!("k-means on stretched clusters: ARI {:.2}", ari(&km, &l)), &x, &km);
    plot.x_range = Some((-6.0, 4.0));
    plot.y_range = Some((-3.5, 7.5));
    write(CHAPTER, "gmm_kmeans.svg", &plot.render());
    let mut g = GaussianMixture::new(3, CovarianceType::Full, 1);
    let gl = g.fit_predict(x.view()).unwrap();
    let mut plot = cluster_plot(&format!("Gaussian mixture (full covariances): ARI {:.2}", ari(&gl, &l)), &x, &gl);
    for c in 0..3 {
        let e = ellipse((g.means()[[c, 0]], g.means()[[c, 1]]), &g.covariances()[c]);
        plot.series.push(Series::new(if c == 0 { "2σ ellipses" } else { "" }, e, Style::Line, 6));
    }
    plot.x_range = Some((-6.0, 4.0));
    plot.y_range = Some((-3.5, 7.5));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "gmm_full.svg", &plot.render());
    // Log-likelihood by iteration.
    let mut ll = Vec::new();
    for it in 1..=g.iterations() {
        let mut g = GaussianMixture::new(3, CovarianceType::Full, 1).with_limits(it, 0.0);
        g.fit_predict(x.view()).unwrap();
        ll.push((it as f64, g.mean_log_likelihood()));
    }
    let mut plot = Plot::new("EM: mean log-likelihood per iteration", "iteration", "mean log-likelihood");
    plot.series.push(Series::new("", ll, Style::LineMarkers, 0));
    write(CHAPTER, "gmm_loglik.svg", &plot.render());
    // BIC against k.
    let mut full = Vec::new();
    let mut diag = Vec::new();
    for k in 1..=7 {
        for (cov, out) in [(CovarianceType::Full, &mut full), (CovarianceType::Diagonal, &mut diag)] {
            let mut g = GaussianMixture::new(k, cov, 1);
            g.fit_predict(x.view()).unwrap();
            out.push((k as f64, g.bic()));
        }
    }
    let mut plot = Plot::new("BIC against the number of components", "components k", "BIC");
    plot.series.push(Series::new("full covariances", full, Style::LineMarkers, 0));
    plot.series.push(Series::new("diagonal covariances", diag, Style::LineMarkers, 1));
    plot.legend = Legend::TopRight;
    write(CHAPTER, "gmm_bic.svg", &plot.render());
    // Recovery of the true parameters.
    let mut rows = Vec::new();
    for c in 0..3 {
        let idx: Vec<usize> = (0..x.nrows()).filter(|&i| l[i] == c).collect();
        let mean = (idx.iter().map(|&i| x[[i, 0]]).sum::<f64>() / idx.len() as f64, idx.iter().map(|&i| x[[i, 1]]).sum::<f64>() / idx.len() as f64);
        let k = (0..3).min_by(|&a, &b| ((g.means()[[a, 0]] - mean.0).hypot(g.means()[[a, 1]] - mean.1)).total_cmp(&(g.means()[[b, 0]] - mean.0).hypot(g.means()[[b, 1]] - mean.1))).unwrap();
        rows.push(vec![format!("{c}"), format!("({:.2}, {:.2})", mean.0, mean.1), format!("({:.2}, {:.2})", g.means()[[k, 0]], g.means()[[k, 1]]), format!("{:.3}", g.weights()[k])]);
    }
    write(CHAPTER, "gmm_parameters.md", &md_table(&["true cluster", "sample mean", "fitted mean", "fitted weight (true 1/3)"], &rows));
}

// ------------------------------------------------------------------ spectral clustering

fn spectral_figures() {
    let (c, cl) = make_circles(150, 0.04, 0.45, 1);
    let km = KMeans::new(2, 1).fit_predict(c.view()).unwrap();
    let mut plot = cluster_plot(&format!("k-means on rings: ARI {:.2}", ari(&km, &cl)), &c, &km);
    plot.x_range = Some((-1.3, 1.3));
    plot.y_range = Some((-1.3, 1.3));
    write(CHAPTER, "spectral_kmeans_rings.svg", &plot.render());
    let mut sc = SpectralClustering::new(2, Affinity::NearestNeighbors { k: 8 }, 1);
    let sl = sc.fit_predict(c.view()).unwrap();
    let mut plot = cluster_plot(&format!("Spectral clustering (8-NN graph): ARI {:.2}", ari(&sl, &cl)), &c, &sl);
    plot.x_range = Some((-1.3, 1.3));
    plot.y_range = Some((-1.3, 1.3));
    write(CHAPTER, "spectral_rings.svg", &plot.render());
    // Eigenvalues for 3 and 5 clusters.
    let mut plot = Plot::new("Smallest eigenvalues of the normalised Laplacian (8-NN graph)", "index", "eigenvalue");
    for (k, colour) in [(3usize, 0usize), (5, 1)] {
        let centres: Vec<Vec<f64>> = (0..k).map(|i| vec![6.0 * (i as f64 * 2.0 * std::f64::consts::PI / k as f64).cos(), 6.0 * (i as f64 * 2.0 * std::f64::consts::PI / k as f64).sin()]).collect();
        let (x, _) = make_blobs(40, &centres, 0.8, 3);
        let mut sc = SpectralClustering::new(8, Affinity::NearestNeighbors { k: 8 }, 1);
        sc.fit_predict(x.view()).unwrap();
        plot.series.push(Series::new(&format!("{k} blobs"), sc.eigenvalues().iter().enumerate().map(|(i, &v)| (i as f64 + 1.0, v)).collect(), Style::LineMarkers, colour));
    }
    plot.legend = Legend::TopLeft;
    write(CHAPTER, "spectral_eigengap.svg", &plot.render());
    // Embedding: the two leading eigenvectors of the moons.
    let mut rows = Vec::new();
    let (m, ml) = make_moons(150, 0.06, 2);
    for k in [3usize, 5, 10, 20, 40] {
        let a = ari(&SpectralClustering::new(2, Affinity::NearestNeighbors { k }, 1).fit_predict(m.view()).unwrap(), &ml);
        let b = ari(&SpectralClustering::new(2, Affinity::NearestNeighbors { k }, 1).fit_predict(c.view()).unwrap(), &cl);
        rows.push(vec![format!("{k}-NN graph"), format!("{a:.3}"), format!("{b:.3}")]);
    }
    for gamma in [1.0, 10.0, 30.0, 100.0] {
        let a = ari(&SpectralClustering::new(2, Affinity::Rbf { gamma }, 1).fit_predict(m.view()).unwrap(), &ml);
        let b = ari(&SpectralClustering::new(2, Affinity::Rbf { gamma }, 1).fit_predict(c.view()).unwrap(), &cl);
        rows.push(vec![format!("RBF, γ = {gamma}"), format!("{a:.3}"), format!("{b:.3}")]);
    }
    write(CHAPTER, "spectral_graphs.md", &md_table(&["similarity graph", "moons (ARI)", "rings (ARI)"], &rows));
}

// ------------------------------------------------------------------ PCA

fn pca_figures() {
    let mut rng = MlRng::new(4);
    let n = 300;
    let x = R2D::from_shape_fn((n, 2), |_| 0.0);
    let mut x = x;
    for i in 0..n {
        let (a, b) = (rng.gaussian(0.0, 2.0), rng.gaussian(0.0, 0.6));
        x[[i, 0]] = 3.0 + 0.8 * a - 0.6 * b;
        x[[i, 1]] = 1.0 + 0.6 * a + 0.8 * b;
    }
    let mut pca = Pca::new(2);
    pca.fit(x.view()).unwrap();
    let mut plot = Plot::new("Principal axes, drawn to ±2 standard deviations", "x₁", "x₂");
    plot.series.push(Series::new("data", (0..n).map(|i| (x[[i, 0]], x[[i, 1]])).collect(), Style::Markers, 7));
    let (mx, my) = (pca.mean()[0], pca.mean()[1]);
    for c in 0..2 {
        let s = 2.0 * pca.explained_variance()[c].sqrt();
        let (ax, ay) = (pca.components()[[c, 0]], pca.components()[[c, 1]]);
        line_segment(&mut plot, (mx - s * ax, my - s * ay), (mx + s * ax, my + s * ay), c, if c == 0 { "first component" } else { "second component" });
    }
    // Equal scales on both axes, so that the axes look perpendicular.
    plot.x_range = Some((-6.0, 12.0));
    plot.y_range = Some((-4.0, 6.0));
    plot.legend = Legend::BottomRight;
    write(CHAPTER, "pca_axes.svg", &plot.render());
    // Scree plot: three latent factors in 12 dimensions.
    let n = 400;
    let mix = R2D::from_shape_fn((3, 12), |_| rng.gaussian(0.0, 1.0));
    let z = R2D::from_shape_fn((n, 3), |(_, j)| rng.gaussian(0.0, [3.0, 2.0, 1.2][j]));
    let x = z.dot(&mix) + R2D::from_shape_fn((n, 12), |_| rng.gaussian(0.0, 0.5));
    let mut pca = Pca::new(12);
    pca.fit(x.view()).unwrap();
    let ratio = pca.explained_variance_ratio();
    let mut cum = 0.0;
    let cumulative: Vec<(f64, f64)> = ratio.iter().enumerate().map(|(i, r)| {
        cum += r;
        (i as f64 + 1.0, cum)
    }).collect();
    let mut plot = Plot::new("Explained variance: three latent factors in 12 dimensions", "component", "fraction of variance");
    plot.series.push(Series::new("per component", ratio.iter().enumerate().map(|(i, &r)| (i as f64 + 1.0, r)).collect(), Style::Stems, 0));
    plot.series.push(Series::new("cumulative", cumulative, Style::LineMarkers, 1));
    plot.legend = Legend::BottomRight;
    plot.x_range = Some((0.5, 12.5));
    plot.y_range = Some((0.0, 1.05));
    write(CHAPTER, "pca_scree.svg", &plot.render());
    // Reconstruction error against components.
    let mut rows = Vec::new();
    for k in [1usize, 2, 3, 4, 6] {
        let mut p = Pca::new(k);
        let s = p.fit_transform(x.view()).unwrap();
        let back = p.inverse_transform(s.view()).unwrap();
        let mse = back.iter().zip(x.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / (n * 12) as f64;
        let discarded: f64 = pca.explained_variance()[k..].iter().sum::<f64>() / 12.0 * (n as f64 - 1.0) / n as f64;
        rows.push(vec![format!("{k}"), format!("{:.4}", p.explained_variance_ratio().iter().sum::<f64>()), format!("{mse:.4}"), format!("{discarded:.4}")]);
    }
    write(CHAPTER, "pca_reconstruction.md", &md_table(&["components kept", "variance explained", "reconstruction mean squared error", "discarded eigenvalues ÷ features"], &rows));
}

fn kernel_pca_figures() {
    let (x, l) = make_circles(150, 0.03, 0.45, 1);
    let mut lin = Pca::new(2);
    let zl = lin.fit_transform(x.view()).unwrap();
    let mut kp = KernelPca::new(Kernel::Rbf { gamma: 2.0 }, 3);
    let zk = kp.fit_transform(x.view()).unwrap();
    let classes = |z: &R2D, cx: usize, cy: usize, title: &str, xl: &str, yl: &str| {
        let mut plot = Plot::new(title, xl, yl);
        for c in 0..2 {
            plot.series.push(Series::new(if c == 0 { "outer ring" } else { "inner ring" }, (0..z.nrows()).filter(|&i| l[i] == c).map(|i| (z[[i, cx]], z[[i, cy]])).collect(), Style::Markers, c));
        }
        plot.legend = Legend::TopRight;
        plot
    };
    write(CHAPTER, "kpca_linear.svg", &classes(&zl, 0, 1, "Linear PCA of two rings", "component 1", "component 2").render());
    write(CHAPTER, "kpca_rbf.svg", &classes(&zk, 0, 2, "RBF kernel PCA (γ = 2): components 1 and 3", "component 1", "component 3").render());
    let mut plot = Plot::new("Kernel PCA eigenvalues (rings, RBF γ = 2)", "index", "eigenvalue");
    let mut all = KernelPca::new(Kernel::Rbf { gamma: 2.0 }, 12);
    all.fit(x.view()).unwrap();
    plot.series.push(Series::new("", all.eigenvalues().iter().enumerate().map(|(i, &v)| (i as f64 + 1.0, v)).collect(), Style::Stems, 0));
    plot.x_range = Some((0.5, 12.5));
    write(CHAPTER, "kpca_eigenvalues.svg", &plot.render());
}

// ------------------------------------------------------------------ LDA

fn lda_figures() {
    let mut rng = MlRng::new(6);
    let n = 150;
    let mut x = R2D::zeros((2 * n, 2));
    let mut l = vec![0usize; 2 * n];
    for i in 0..2 * n {
        let c = i / n;
        let (a, b) = (rng.gaussian(0.0, 3.0), rng.gaussian(0.0, 0.5));
        let shift = if c == 0 { -0.9 } else { 0.9 };
        // Long axis along (1, 1); the classes are offset across it, slightly tilted.
        x[[i, 0]] = 0.707 * a - 0.707 * b + shift * 0.5;
        x[[i, 1]] = 0.707 * a + 0.707 * b - shift * 0.7;
        l[i] = c;
    }
    let mut pca = Pca::new(1);
    pca.fit(x.view()).unwrap();
    let mut lda = Lda::new(1);
    lda.fit(x.view(), &l).unwrap();
    let mut plot = Plot::new("Projection directions: PCA against LDA", "x₁", "x₂");
    for c in 0..2 {
        plot.series.push(Series::new(&format!("class {c}"), (0..2 * n).filter(|&i| l[i] == c).map(|i| (x[[i, 0]], x[[i, 1]])).collect(), Style::Markers, c));
    }
    let (px, py) = (pca.components()[[0, 0]], pca.components()[[0, 1]]);
    let (dx, dy) = (lda.directions()[[0, 0]], lda.directions()[[1, 0]]);
    line_segment(&mut plot, (-7.0 * px, -7.0 * py), (7.0 * px, 7.0 * py), 6, "PCA direction");
    line_segment(&mut plot, (-4.0 * dx, -4.0 * dy), (4.0 * dx, 4.0 * dy), 4, "LDA direction");
    plot.x_range = Some((-12.6, 12.6));
    plot.y_range = Some((-7.0, 7.0));
    plot.legend = Legend::TopLeft;
    write(CHAPTER, "lda_directions.svg", &plot.render());
    // Histograms of the projections as frequency polygons.
    let zp = pca.transform(x.view()).unwrap();
    let zl = lda.transform(x.view()).unwrap();
    for (z, name, file) in [(&zp, "PCA", "lda_hist_pca.svg"), (&zl, "LDA", "lda_hist_lda.svg")] {
        let v: Vec<f64> = z.column(0).to_vec();
        let (lo, hi) = (v.iter().copied().fold(f64::INFINITY, f64::min), v.iter().copied().fold(f64::NEG_INFINITY, f64::max));
        let bins = 30;
        let mut plot = Plot::new(&format!("Projection onto the {name} direction"), "projection", "count");
        for c in 0..2 {
            let mut counts = vec![0.0; bins];
            for (_, &t) in v.iter().enumerate().filter(|(i, _)| l[*i] == c) {
                counts[(((t - lo) / (hi - lo) * bins as f64) as usize).min(bins - 1)] += 1.0;
            }
            plot.series.push(Series::new(&format!("class {c}"), counts.iter().enumerate().map(|(b, &k)| (lo + (b as f64 + 0.5) * (hi - lo) / bins as f64, k)).collect(), Style::Line, c));
        }
        plot.legend = Legend::TopRight;
        write(CHAPTER, file, &plot.render());
    }
    // Accuracy of a threshold on each projection, and LDA as a classifier.
    let (x4, l4) = make_classification(150, 2);
    let (t4, tl4) = make_classification(300, 3);
    let mut lda4 = Lda::new(2);
    lda4.fit(x4.view(), &l4).unwrap();
    let z4 = lda4.transform(x4.view()).unwrap();
    let mut plot = Plot::new("LDA projection of three classes in four dimensions", "discriminant 1", "discriminant 2");
    for c in 0..3 {
        plot.series.push(Series::new(&format!("class {c}"), (0..x4.nrows()).filter(|&i| l4[i] == c).map(|i| (z4[[i, 0]], z4[[i, 1]])).collect(), Style::Markers, c));
    }
    plot.legend = Legend::TopRight;
    write(CHAPTER, "lda_projection.svg", &plot.render());
    let mut p4 = Pca::new(2);
    let zp4 = p4.fit_transform(x4.view()).unwrap();
    let mut plot = Plot::new("PCA projection of the same data", "component 1", "component 2");
    for c in 0..3 {
        plot.series.push(Series::new(&format!("class {c}"), (0..x4.nrows()).filter(|&i| l4[i] == c).map(|i| (zp4[[i, 0]], zp4[[i, 1]])).collect(), Style::Markers, c));
    }
    plot.legend = Legend::TopRight;
    write(CHAPTER, "lda_pca_projection.svg", &plot.render());
    let lda_acc = accuracy(&tl4, &lda4.predict(t4.view()).unwrap()).unwrap();
    let mut rows = vec![vec!["LDA classifier (2 discriminants)".to_string(), format!("{lda_acc:.3}")]];
    let mut lda1 = Lda::new(1);
    lda1.fit(x4.view(), &l4).unwrap();
    rows.push(vec!["LDA classifier (1 discriminant)".to_string(), format!("{:.3}", accuracy(&tl4, &lda1.predict(t4.view()).unwrap()).unwrap())]);
    let mut nb = lib_jsl_machine_learning::supervised::classification::GaussianNaiveBayes::new();
    nb.fit(x4.view(), &l4).unwrap();
    rows.push(vec!["Gaussian naive Bayes".to_string(), format!("{:.3}", accuracy(&tl4, &nb.predict(t4.view()).unwrap()).unwrap())]);
    let mut lr = lib_jsl_machine_learning::supervised::classification::LogisticRegression::new(1e-4);
    lr.fit(x4.view(), &l4).unwrap();
    rows.push(vec!["logistic regression".to_string(), format!("{:.3}", accuracy(&tl4, &lr.predict(t4.view()).unwrap()).unwrap())]);
    write(CHAPTER, "lda_accuracy.md", &md_table(&["classifier (450 training, 900 test points)", "test accuracy"], &rows));
}

// ------------------------------------------------------------------ t-SNE and UMAP

fn roll_plot(title: &str, y: &R2D, t: &[f64], file: &str) {
    let mut plot = Plot::new(title, "", "");
    let (lo, hi) = (1.5 * std::f64::consts::PI, 4.5 * std::f64::consts::PI);
    for b in 0..6 {
        let (a, c) = (lo + (hi - lo) * b as f64 / 6.0, lo + (hi - lo) * (b + 1) as f64 / 6.0);
        plot.series.push(Series::new("", (0..y.nrows()).filter(|&i| t[i] >= a && t[i] < c + 1e-12).map(|i| (y[[i, 0]], y[[i, 1]])).collect(), Style::Markers, b));
    }
    write(CHAPTER, file, &plot.render());
}

fn embedding_figures() {
    let (roll, t) = make_swiss_roll(600, 0.2, 2);
    let mut view = Plot::new("Swiss roll seen from above (x₁, x₃), coloured by position", "x₁", "x₃");
    let (lo, hi) = (1.5 * std::f64::consts::PI, 4.5 * std::f64::consts::PI);
    for b in 0..6 {
        let (a, c) = (lo + (hi - lo) * b as f64 / 6.0, lo + (hi - lo) * (b + 1) as f64 / 6.0);
        view.series.push(Series::new("", (0..roll.nrows()).filter(|&i| t[i] >= a && t[i] < c + 1e-12).map(|i| (roll[[i, 0]], roll[[i, 2]])).collect(), Style::Markers, b));
    }
    write(CHAPTER, "roll_data.svg", &view.render());
    let yp = Pca::new(2).fit_transform(roll.view()).unwrap();
    roll_plot("PCA of the Swiss roll", &yp, &t, "roll_pca.svg");
    let mut ts = Tsne::new(2, 30.0);
    let yt = ts.fit_transform(roll.view()).unwrap();
    roll_plot("t-SNE of the Swiss roll (perplexity 30)", &yt, &t, "roll_tsne.svg");
    let yu = Umap::new(2, 15, 0.1, 1).fit_transform(roll.view()).unwrap();
    roll_plot("UMAP of the Swiss roll (15 neighbours)", &yu, &t, "roll_umap.svg");
    // Clusters in 20 dimensions: perplexity.
    let centres: Vec<Vec<f64>> = (0..5).map(|c| (0..20).map(|d| if d % 5 == c { 5.0 } else { 0.0 }).collect()).collect();
    let (xc, lc) = make_blobs(60, &centres, 1.0, 4);
    let cl = as_options(&lc);
    for perp in [3.0, 30.0, 100.0] {
        let y = Tsne::new(2, perp).fit_transform(xc.view()).unwrap();
        let mut plot = cluster_plot(&format!("t-SNE, perplexity {perp}"), &y, &cl);
        plot.x_label = String::new();
        plot.y_label = String::new();
        write(CHAPTER, &format!("tsne_perplexity_{perp}.svg"), &plot.render());
    }
    for md in [0.0, 0.5] {
        let y = Umap::new(2, 15, md, 1).fit_transform(xc.view()).unwrap();
        let mut plot = cluster_plot(&format!("UMAP, min_dist {md}"), &y, &cl);
        plot.x_label = String::new();
        plot.y_label = String::new();
        write(CHAPTER, &format!("umap_min_dist_{md}.svg"), &plot.render());
    }
    // Neighbour preservation and time.
    let mut rows = Vec::new();
    for (name, data) in [("Swiss roll (600 points, 3D)", &roll), ("five clusters (300 points, 20D)", &xc)] {
        let mut cells = vec![name.to_string()];
        let mut times = Vec::new();
        let tp = time(|| {
            Pca::new(2).fit_transform(data.view()).unwrap();
        });
        let p = Pca::new(2).fit_transform(data.view()).unwrap();
        cells.push(format!("{:.2}", neighbour_preservation(data, &p, 10)));
        times.push(tp);
        let t0 = std::time::Instant::now();
        let y = Tsne::new(2, 30.0).fit_transform(data.view()).unwrap();
        times.push(t0.elapsed().as_secs_f64());
        cells.push(format!("{:.2}", neighbour_preservation(data, &y, 10)));
        let t0 = std::time::Instant::now();
        let y = Umap::new(2, 15, 0.1, 1).fit_transform(data.view()).unwrap();
        times.push(t0.elapsed().as_secs_f64());
        cells.push(format!("{:.2}", neighbour_preservation(data, &y, 10)));
        cells.push(times.iter().map(|t| if *t < 1e-3 { format!("{:.0} µs", t * 1e6) } else if *t < 0.1 { format!("{:.1} ms", t * 1e3) } else { format!("{t:.2} s") }).collect::<Vec<_>>().join(" / "));
        rows.push(cells);
    }
    write(CHAPTER, "embedding_table.md", &md_table(&["data", "PCA", "t-SNE", "UMAP", "time (PCA / t-SNE / UMAP)"], &rows));
}

// ------------------------------------------------------------------ overview comparison

fn comparison_table() {
    let (blobs, bl) = make_blobs(60, &[vec![0.0, 0.0], vec![5.0, 0.0], vec![2.5, 4.5]], 1.0, 2);
    let (aniso, al) = anisotropic();
    let (moons, ml) = make_moons(150, 0.06, 2);
    let (rings, rl) = make_circles(150, 0.04, 0.45, 1);
    let (dens, dl) = two_densities();
    let sets: [(&str, &R2D, &Vec<usize>); 5] = [("blobs", &blobs, &bl), ("stretched", &aniso, &al), ("moons", &moons, &ml), ("rings", &rings, &rl), ("unequal densities", &dens, &dl)];
    type Method = Box<dyn Fn(&R2D, usize) -> Vec<Option<usize>>>;
    let methods: Vec<(&str, Method)> = vec![
        ("k-means", Box::new(|x: &R2D, k| KMeans::new(k, 1).fit_predict(x.view()).unwrap())),
        ("k-medoids", Box::new(|x: &R2D, k| KMedoids::new(k).fit_predict(x.view()).unwrap())),
        ("Ward linkage", Box::new(|x: &R2D, k| HierarchicalClustering::new(k, Linkage::Ward).fit_predict(x.view()).unwrap())),
        ("single linkage", Box::new(|x: &R2D, k| HierarchicalClustering::new(k, Linkage::Single).fit_predict(x.view()).unwrap())),
        ("Gaussian mixture", Box::new(|x: &R2D, k| GaussianMixture::new(k, CovarianceType::Full, 1).fit_predict(x.view()).unwrap())),
        ("DBSCAN (ε = 0.3)", Box::new(|x: &R2D, _| Dbscan::new(0.3, 5).fit_predict(x.view()).unwrap())),
        ("HDBSCAN (15)", Box::new(|x: &R2D, _| Hdbscan::new(15).fit_predict(x.view()).unwrap())),
        ("spectral (10-NN graph)", Box::new(|x: &R2D, k| SpectralClustering::new(k, Affinity::NearestNeighbors { k: 10 }, 1).fit_predict(x.view()).unwrap())),
    ];
    let mut rows = Vec::new();
    for (name, f) in &methods {
        let mut r = vec![name.to_string()];
        for (_, x, l) in &sets {
            let k = l.iter().max().unwrap() + 1;
            r.push(format!("{:.2}", ari(&f(x, k), l)));
        }
        rows.push(r);
    }
    let mut header = vec!["method (adjusted Rand index)"];
    header.extend(sets.iter().map(|s| s.0));
    write(CHAPTER, "comparison.md", &md_table(&header, &rows));
}
