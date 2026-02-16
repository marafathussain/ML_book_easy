# Chapter 5: Clustering, Dimensionality Reduction, and Visualization for Biologists

## Introduction

In Chapters 2, 3, and 4, we focused on **supervised learning**: we had labeled data (e.g., flower species, disease status) and trained models to predict labels for new samples. In this chapter, we shift to **unsupervised learning**: we do not have labels. Instead, we ask: "What structure or groups exist in the data?" and "How can we visualize high-dimensional data in two or three dimensions?"

These tools are essential for biologists. You might have gene expression data from hundreds of cells with no prior labels and want to discover cell types (clustering). You might have thousands of genes (high dimensions) and want to see which samples are similar (dimensionality reduction and visualization). Single-cell RNA-seq, microbiome data, and proteomics all produce high-dimensional measurements where clustering and visualization help reveal biological structure.

This chapter covers four topics:

1. **Clustering:** k-means and hierarchical clustering, to group similar samples together.
2. **PCA (Principal Component Analysis):** variance, components, and projections, to reduce dimensions while preserving structure.
3. **t-SNE and UMAP:** methods for visualizing high-dimensional data in 2D, widely used for single-cell and other biological data.
4. **Visualizing biological structure:** how to interpret and use these methods for problems like cell populations and sample heterogeneity.

All examples use the **Iris dataset** where helpful, plus biological contexts (e.g., cell types, gene expression). Code is written for Google Colab; no local installation is required.

---

## 5.1 Clustering: Finding Groups in Unlabeled Data

**Clustering** groups samples that are **similar** to each other and **different** from samples in other groups. We do not use labels; we let the algorithm find structure from the data. In biology, clustering helps discover cell types, patient subtypes, or species groups from measurements alone.

**Key idea:** Samples that are close in measurement space (e.g., similar gene expression) tend to belong to the same group. Clustering algorithms formalize "close" and "group" in different ways.

### 5.1.1 k-Means Clustering

**What is k-means?**

k-means divides samples into $k$ groups (clusters). Each cluster has a **center** (centroid). The algorithm minimizes the total squared distance from each sample to its assigned cluster center.

**The objective:**

$$
\text{Minimize} \quad \sum_{i=1}^{n} \| \mathbf{x}_i - \boldsymbol{\mu}_{c_i} \|^2
$$

Here $\mathbf{x}_i$ is sample $i$, $c_i$ is the cluster assigned to sample $i$, and $\boldsymbol{\mu}_k$ is the center of cluster $k$. The algorithm iterates: (1) assign each sample to the nearest center, (2) update each center to be the mean of the samples in that cluster, until convergence.

**Steps in words:**

1. Choose $k$ (number of clusters) and initialize $k$ centers (e.g., randomly from the data).
2. Assign each sample to the cluster whose center is closest.
3. Update each center to be the mean of all samples assigned to it.
4. Repeat steps 2 and 3 until assignments do not change.

**How to choose $k$:** There is no single correct answer. Use domain knowledge (e.g., you expect 3 cell types) or try several $k$ and compare. The **elbow method** plots the total within-cluster distance vs $k$; look for a "bend" where adding more clusters does not help much.

**Iris example:** With Iris petal length and width, k-means with $k=3$ typically recovers the three species reasonably well, even without using species labels. The figure below shows k-means clusters on Iris in 2D.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter5/kmeans_iris.png" alt="k-means clustering on Iris" />
  <p class="caption"><strong>Figure 5.1.</strong> k-means clustering ($k=3$) on Iris petal length and width. Colors show cluster assignment. Centroids are shown as larger markers. The algorithm finds three groups without using species labels.</p>
</div>

**How to use in Python:**

```python
from sklearn.cluster import KMeans

# X: your data (n_samples x n_features)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_
```

**Limitations:** k-means assumes clusters are roughly spherical and similar in size. It does not work well for long, stretched clusters or clusters of very different densities.

### 5.1.2 Hierarchical Clustering

**What is hierarchical clustering?**

Hierarchical clustering builds a **tree** (dendrogram) of clusters. You can cut the tree at any height to get a desired number of clusters, or explore the full hierarchy (e.g., cells, then cell types, then subtypes).

**Two main approaches:**

- **Agglomerative (bottom-up):** Start with each sample as its own cluster (n clusters). Repeatedly merge the two closest clusters until you have one cluster.
- **Divisive (top-down):** Start with all samples in one cluster, and split recursively. Less common in practice.

**How "closeness" is defined:**

- **Single linkage:** Distance between two clusters = minimum distance between any pair of points, one from each cluster.
- **Complete linkage:** Distance = maximum distance between any pair.
- **Average linkage:** Distance = average pairwise distance between points in the two clusters.
- **Ward:** Merge the two clusters that give the smallest increase in total within-cluster variance (similar to k-means).

**The distance matrix:** We need distances between all pairs of samples. For vectors, Euclidean distance is common:

$$
d(\mathbf{x}_i, \mathbf{x}_j) = \| \mathbf{x}_i - \mathbf{x}_j \| = \sqrt{\sum_{f=1}^{p} (x_{if} - x_{jf})^2}
$$

**Dendrogram:** The tree shows merge order. The vertical axis is distance (or similarity); cutting at a given height gives a partition into clusters.

**Iris example:** Hierarchical clustering on Iris produces a dendrogram. Cutting at a low height gives many small clusters; cutting higher gives fewer, larger clusters. The figure below shows an example dendrogram.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter5/hierarchical_iris.png" alt="Hierarchical clustering on Iris" />
  <p class="caption"><strong>Figure 5.2.</strong> Agglomerative hierarchical clustering on Iris (30 samples for a readable dendrogram). Left: dendrogram; x-axis shows sample indices. A horizontal cut at a given height defines clusters. Right: cluster assignment (3 clusters) in 2D for the same 30 samples.</p>
</div>

**How to use in Python:**

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Linkage matrix for dendrogram
Z = linkage(X, method='ward')
# Plot dendrogram (use plt.show() or savefig)
dendrogram(Z)

# Or get cluster labels directly
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg.fit_predict(X)
```

**When to use hierarchical vs k-means:** Use hierarchical when you want to explore multiple levels of grouping or when cluster shapes are not spherical. Use k-means when you know $k$ and want fast, scalable clustering.

---

## 5.2 PCA: Principal Component Analysis

When you have many features (e.g., thousands of genes), visualization and analysis become difficult. **PCA** reduces the number of dimensions while keeping as much variance (spread) as possible. It finds new directions (principal components) that capture the most variation in the data. To understand PCA, you first need to understand **variance** and how we use it to rank components. Below we build this from the ground up.

### 5.2.1 What is variance? A step-by-step example

**Variance** is a number that measures how spread out a set of values is. If all values are the same, variance is zero. If they are very different from each other, variance is large.

**How we compute variance (one feature):**

1. Compute the **mean** (average) of the values.
2. For each value, compute how far it is from the mean: (value minus mean).
3. Square those differences (so positive and negative do not cancel).
4. Average those squared differences. (In practice we often divide by $n-1$ instead of $n$ for a sample, as in Chapter 4.)

So variance is the **average squared distance from the mean**. In a formula, for values $x_1, x_2, \ldots, x_n$ with mean $\bar{x}$:

$$
\text{Variance} = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2
$$

**Tiny example with numbers:**

Suppose we have 5 flowers and one feature, **petal length (cm):** 1, 2, 3, 4, 5.

- Mean = (1+2+3+4+5) / 5 = **3**.
- Differences from mean: -2, -1, 0, 1, 2.
- Squared differences: 4, 1, 0, 1, 4.
- Variance = (4+1+0+1+4) / 4 = **2.5** (using $n-1=4$ in the denominator).

So the variance of this small set is 2.5. If we had another feature (e.g., petal width) we could compute its variance the same way. Each feature has its own variance.

**Two features: variance along each axis**

Imagine we have 5 samples and 2 features (e.g., petal length and petal width):

| Sample | Petal length (A) | Petal width (B) |
|--------|------------------|------------------|
| 1      | 1                | 0.5              |
| 2      | 2                | 1.0              |
| 3      | 3                | 1.5              |
| 4      | 4                | 2.0              |
| 5      | 5                | 2.5              |

Mean of A = 3, mean of B = 1.5. After **centering** (subtract the mean from each column), we get:

| Sample | A (centered) | B (centered) |
|--------|--------------|--------------|
| 1      | -2           | -1.0         |
| 2      | -1           | -0.5         |
| 3      | 0            | 0            |
| 4      | 1            | 0.5          |
| 5      | 2            | 1.0          |

**Variance of A** = ((-2)² + (-1)² + 0² + 1² + 2²) / 4 = 10/4 = **2.5**.

**Variance of B** = ((-1)² + (-0.5)² + 0² + (0.5)² + (1)²) / 4 = 2.5/4 = **0.625**.

So along the "A axis" (petal length) the spread is 2.5; along the "B axis" (petal width) the spread is 0.625. The data are more spread out along A than along B. **Total variance** (sum of the two) = 2.5 + 0.625 = **3.125**.

### 5.2.2 How PCA uses variance: new axes and component ranking

The key idea of PCA: instead of only looking at variance along the **original** axes (A and B), we ask: "Is there another **direction** (a slanted line through the cloud) along which the variance is even larger?"

In our small example, the points lie almost on a straight line (B is half of A). So there is one direction (along that line) where the spread is **maximum**: that is the **first principal component (PC1)**. The variance of the data when we project them onto PC1 is **larger** than the variance along either A or B alone. The direction **perpendicular** to PC1 has the remaining spread; that is the **second principal component (PC2)**.

**Ranking components by variance:**

- PCA finds directions in order of **decreasing variance**.
  - **PC1:** direction of **maximum** variance.
  - **PC2:** direction of maximum variance among all directions **perpendicular** to PC1.
  - **PC3:** direction of maximum variance perpendicular to PC1 and PC2, and so on.

- For each component, we compute the **variance of the projected data** along that direction. That number is the **variance explained** by that component (often reported as an eigenvalue).

- **Variance explained ratio** for a component = (variance of that component) / (total variance of all features). So we get a percentage or fraction for each component. For example: PC1 explains 80%, PC2 explains 15%, PC3 explains 4%, PC4 explains 1%. That is what you see in a **scree plot**: each bar or point is the variance (or % variance) of one component, in order PC1, PC2, PC3, …

- **Why ranking matters:** We keep the top components (e.g., PC1 and PC2) because they capture most of the spread. The later components often capture mostly noise. So we reduce dimensions by keeping only the first few components and throwing away the rest.

**Summary in one sentence:** PCA finds new axes (principal components) so that the first axis has the largest variance, the second has the next largest (and is perpendicular to the first), and so on; we rank components by this variance and use the top ones to summarize the data.

**Mathematical formulation (for reference):**

The first principal component $\mathbf{w}_1$ is the unit vector that maximizes the variance of the projected data:

$$
\mathbf{w}_1 = \arg\max_{\|\mathbf{w}\|=1} \text{Var}(\mathbf{X} \mathbf{w})
$$

The second component $\mathbf{w}_2$ is orthogonal to $\mathbf{w}_1$ and maximizes the remaining variance. The variances along these directions (eigenvalues of the covariance matrix) are exactly the "variance explained" values we use for ranking and for the scree plot.

### 5.2.3 Projections

**Projection** means projecting each sample onto a principal component. If $\mathbf{w}_1$ is the first component (a unit vector), the projection of sample $\mathbf{x}_i$ onto PC1 is:

$$
z_{i1} = \mathbf{x}_i \cdot \mathbf{w}_1 = \mathbf{x}_i^\top \mathbf{w}_1
$$

The projected values $z_{i1}, z_{i2}, \ldots$ are the **scores** or **coordinates** in the new PCA space. We can plot samples in 2D using PC1 vs PC2.

**Centering:** PCA is usually applied to **centered** data (subtract the mean of each feature). The components then pass through the center of the cloud.

### 5.2.4 Interpreting PCA for Biologists

- **PC1** often captures the main gradient or axis of variation (e.g., cell type, treatment effect, developmental stage).
- **Loadings:** The weight of each original feature in a component. High loading means that feature contributes a lot to that component. For gene expression, genes with high loadings on PC1 are the ones that vary most along PC1.
- **Scores:** The coordinates of each sample in PCA space. Samples with similar scores are similar along that component.

**Iris example:** PCA on Iris (4 features) finds that PC1 and PC2 capture most of the variance. A scatter plot of PC1 vs PC2 often separates the three species, showing that the main variation in the data aligns with species differences.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter5/pca_iris.png" alt="PCA on Iris" />
  <p class="caption"><strong>Figure 5.3.</strong> PCA on Iris. Left: scree plot (variance explained by each component). Right: scatter of PC1 vs PC2, colored by species. The first two components capture most variance and separate the species.</p>
</div>

**How to use in Python:**

```python
from sklearn.decomposition import PCA

# X: n_samples x n_features
pca = PCA(n_components=2)  # Keep 2 components for visualization
X_pca = pca.fit_transform(X)

# Variance explained
print(pca.explained_variance_ratio_)  # e.g. [0.73, 0.23]
# Loadings: pca.components_  (shape: n_components x n_features)
```

---

## 5.3 t-SNE and UMAP for Biological Data

PCA is linear: it finds straight lines that capture variance. Many biological datasets have **nonlinear** structure (e.g., curved manifolds, branching trajectories). **t-SNE** and **UMAP** are nonlinear methods that map high-dimensional data to 2D (or 3D) for visualization while preserving local structure (nearby points stay nearby).

### 5.3.1 t-SNE (t-Distributed Stochastic Neighbor Embedding)

**What does t-SNE do?**

t-SNE constructs a 2D (or low-D) embedding where points that are **close in the original space** stay close in the embedding, and points that are far apart tend to stay far apart. It converts distances in high-D into probabilities (similarities) and tries to match those probabilities in low-D.

**Key ideas (simplified):**

1. In high-D, for each point $i$, define a probability distribution over all other points: points close to $i$ get high probability, far points get low probability.
2. In low-D, define a similar probability distribution (using a Student-t distribution, hence "t-SNE").
3. Minimize the mismatch between the two distributions (Kullback-Leibler divergence) by moving points in the low-D space.

**Important caveats:**

- t-SNE is for **visualization only**. Do not use t-SNE output for clustering or downstream analysis; distances and cluster sizes in the plot can be distorted.
- **Perplexity** controls how many neighbors each point considers (typically 5 to 50). Low perplexity = focus on very local structure; high perplexity = more global structure.
- Results can vary between runs (random initialization). Use a fixed `random_state` for reproducibility.

**Biological use:** t-SNE is widely used for single-cell RNA-seq, flow cytometry, and other high-dimensional data to visualize cell populations and identify clusters by eye.

### 5.3.2 UMAP (Uniform Manifold Approximation and Projection)

**What does UMAP do?**

UMAP also maps high-D data to 2D (or 3D) for visualization. It assumes the data lie on a **manifold** (a curved surface in high-D) and approximates it in low dimensions. UMAP tends to preserve both local and global structure better than t-SNE in many cases.

**Advantages over t-SNE (in practice):**

- Often faster for large datasets.
- Better preservation of global structure (e.g., cluster separation, relative distances).
- Can be used for dimensionality reduction (not just 2D visualization); the low-D representation can sometimes be used for clustering or as input to other methods.

**Parameters:**

- **n_neighbors:** Number of local neighbors (similar to perplexity in t-SNE). Low = local structure; high = global structure.
- **min_dist:** Minimum distance between points in the embedding. Low = tighter clusters; high = more spread out.

**Biological use:** UMAP has become very popular in single-cell analysis (e.g., Scanpy, Seurat pipelines). It is often the default choice for visualizing cell populations.

### 5.3.3 t-SNE vs UMAP: When to Use Which

| Aspect | t-SNE | UMAP |
|--------|-------|------|
| Speed | Slower for large n | Often faster |
| Global structure | Can be distorted | Usually better preserved |
| Reproducibility | Varies by run | More stable with fixed seed |
| Downstream use | Visualization only | Can use embedding for clustering etc. |
| Biological adoption | Very common in single-cell | Increasingly default |

Both are powerful for exploring high-dimensional biological data. Use UMAP when you want faster runs and better global structure; use t-SNE when you need compatibility with older pipelines or want to compare with published t-SNE plots.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter5/tsne_umap_iris.png" alt="t-SNE and UMAP on Iris" />
  <p class="caption"><strong>Figure 5.4.</strong> t-SNE and UMAP 2D embeddings of Iris (4 features). Both methods separate the three species into distinct groups. Colors indicate true species (for illustration); in practice these methods do not use labels.</p>
</div>

**How to use in Python:**

```python
# t-SNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# UMAP (requires: pip install umap-learn)
import umap
reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
X_umap = reducer.fit_transform(X)
```

---

## 5.4 Visualizing Biological Structure: Cell Populations and Beyond

In biology, clustering and visualization are often used together to **discover** and **communicate** structure.

### 5.4.1 Typical Workflow for Single-Cell or High-Dimensional Data

1. **Preprocess:** Normalize, scale, maybe select highly variable features.
2. **Reduce dimensions:** PCA first (to denoise and reduce), then t-SNE or UMAP for 2D visualization.
3. **Cluster:** k-means, hierarchical, or graph-based methods (e.g., Leiden, Louvain) on PCA space or on a neighborhood graph.
4. **Visualize:** Plot the 2D embedding, color points by cluster, by a gene of interest, or by metadata (e.g., sample, condition).

### 5.4.2 Interpreting Plots for Biologists

- **Cluster separation:** Well-separated clusters in t-SNE or UMAP often correspond to distinct cell types or states. Overlapping clusters may indicate similar cell types or a continuous transition.
- **Gene expression overlay:** Color points by expression of a marker gene. High expression in one cluster suggests that gene is a marker for that population.
- **Metadata overlay:** Color by sample, condition, or batch. This helps detect batch effects or condition-specific populations.

**Example:** In single-cell RNA-seq, you might run UMAP on log-normalized counts, cluster with Leiden, and color by cluster. Then overlay CD4, CD8, or other marker genes to annotate T cell subsets. The same workflow applies to microbiome samples, proteomics, or any high-dimensional biological data.

### 5.4.3 Pitfalls and Best Practices

- **Do not over-interpret cluster boundaries.** Clustering is exploratory; validate with known markers or independent data.
- **Check for batch effects.** If samples from different batches form separate clusters, consider batch correction before clustering.
- **Use PCA before t-SNE or UMAP** when you have many features. Running t-SNE directly on thousands of genes can be slow and noisy; PCA to 50 or 100 components is a common first step.

---

## Summary

- **Clustering** (k-means, hierarchical) groups similar samples without labels. Use k-means when you know $k$ and want fast results; use hierarchical when you want to explore multiple levels or non-spherical clusters.
- **PCA** finds directions of maximum variance and projects data onto them. Use it to reduce dimensions, visualize main gradients, and as a preprocessing step before t-SNE or UMAP.
- **t-SNE and UMAP** create 2D (or 3D) embeddings for visualization. Both preserve local structure; UMAP often preserves global structure better and is faster. Use them to explore cell populations, sample heterogeneity, and high-dimensional biological data.
- **Biological workflow:** Preprocess, reduce (PCA), visualize (t-SNE or UMAP), cluster, and annotate with markers or metadata. Interpret with caution and validate with domain knowledge.

---

## Topics for the Next Class

- **Why deep learning?** When and why to move from classical ML to neural networks.
- **Structure of a neural network** Layers, neurons, and connections.
- **Activations, loss functions, optimizers** Building blocks for training.
- **Training loops and overfitting** How networks learn, and how to prevent overfitting.

---

## Further Reading

- [Scikit-learn: Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Scikit-learn: PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [Scikit-learn: t-SNE](https://scikit-learn.org/stable/modules/manifold.html#t-sne)
- [UMAP documentation](https://umap-learn.readthedocs.io/)
- Scanpy and Seurat tutorials for single-cell RNA-seq analysis
