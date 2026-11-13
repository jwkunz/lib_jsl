# Machine learning: clustering and dimensionality reduction

**Unsupervised learning** looks for structure in data without labels. **Clustering** divides points into groups whose members resemble
each other more than they resemble the rest. **Dimensionality reduction** describes points by a few coordinates that keep
what matters, for visualisation, compression, or as input to another method. Neither problem has a single right answer.
"Groups" and "what matters" must be defined, and each method's definition, whether distance to a centre, density,
connectivity, variance or neighbourhood, determines what it can find.

## The central questions

- **What is a cluster?** Compact around a centre (k-means, k-medoids, Gaussian mixtures), a dense region (DBSCAN, HDBSCAN), a
  well-connected part of a graph (spectral clustering, single linkage), or a branch of a hierarchy (agglomerative clustering)?
- **How many clusters?** Centroid and mixture methods need $k$, with the silhouette, the elbow or an information criterion as guides;
  density methods and dendrograms find it, or show the choice at every scale.
- **Should everything be assigned?** Density methods label outliers as noise; the others force every point into some cluster.
- **What should a projection keep?** Variance (PCA), separation of known classes (LDA), or local neighbourhoods (t-SNE, UMAP).
  Linear projections can be inverted and applied to new data; neighbour embeddings only draw a picture of the data they were given.

## Comparing clustering methods

Adjusted Rand index against the generating labels (1 is perfect agreement, 0 is chance) for each method, given the true
number of clusters where it needs one:

{{#include ../generated/unsupervised/comparison.md}}

No method wins everywhere. Centroid and mixture methods handle blobs, and only the mixture handles stretched ones. Single
linkage, density and spectral methods follow the moons and rings. Cut into three clusters, single linkage splits off two stray points
and leaves the three blobs together. A fixed DBSCAN radius suits the moons but not blobs of spread 1. HDBSCAN copes with the
unequal densities; on the ordinary blobs its three clusters are pure, but it labels 24 of the 180 points as noise, which lowers its score.

## Testing the library

Hierarchical merge heights for all four linkages match SciPy to rounding error, and PCA matches NumPy's SVD. Neighbour joining
recovers random additive trees exactly. The other methods are checked against the properties they guarantee: EM never decreases
the likelihood, the Laplacian has one zero eigenvalue per connected component, and the PCA reconstruction error equals the
discarded variance.

k-means, compared with the best of twenty k-means++ starts of SciPy's `cluster.vq.kmeans2` on four separated clusters, finds the same partition:

{{#include ../generated/unsupervised/scipy_kmeans.md}}

## In the library

The clustering and reduction methods are in `lib_jsl_machine_learning::unsupervised` ([API reference](../../api/lib_jsl_machine_learning/unsupervised/index.html)).
Clusterers implement `Clusterer::fit_predict`, returning `Option<usize>` labels with `None` for noise; projections implement
`Transformer`. The phylogenetic tree builders are in `supervised::classification`.

| You need | Use | Entry |
|---|---|---|
| Fast clustering into $k$ round groups, or clustering from a distance matrix | `KMeans`, `KMedoids` | [k-means and k-medoids](k-means.md) |
| A hierarchy of clusters, or an evolutionary tree | `agglomerate`, `HierarchicalClustering`, `upgma`, `neighbor_joining` | [Hierarchical clustering](hierarchical.md) |
| Clusters of any shape, with noise | `Dbscan`, `Hdbscan` | [DBSCAN and HDBSCAN](density.md) |
| Soft clustering, elliptical clusters, a density model | `GaussianMixture` | [Gaussian mixtures](gaussian-mixtures.md) |
| Clusters defined by graph connectivity | `SpectralClustering` | [Spectral clustering](spectral.md) |
| A linear projection that keeps variance | `Pca`, `KernelPca` | [PCA and kernel PCA](pca.md) |
| A projection that separates labelled classes | `Lda` | [Linear discriminant analysis](lda.md) |
| A two-dimensional picture of high-dimensional data | `Tsne`, `Umap` | [t-SNE and UMAP](embeddings.md) |
