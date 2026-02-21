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

**How to choose $k$:** There is no single correct answer. Use domain knowledge (e.g., you expect 3 cell types) or try several $k$ and compare. The **elbow method** plots the total within-cluster sum of squares (inertia) vs $k$; look for a "bend" where adding more clusters does not help much.

**Elbow plot and how to interpret it.**

We run k-means for $k = 1, 2, 3, \ldots$ and plot the **inertia** (within-cluster sum of squares) on the y-axis vs $k$ on the x-axis. As $k$ increases, inertia goes down (more clusters mean points are closer to their centroids). The **elbow** is the value of $k$ where the curve bends: before the elbow, adding a cluster reduces inertia a lot; after the elbow, adding more clusters gives only a small improvement. That $k$ is a reasonable choice. The plot below shows this for Iris (petal length and width). The curve drops sharply from $k=1$ to $k=2$ and again to $k=3$, then flattens. So $k=3$ is a natural elbow and matches the three species in the data. If there were no clear bend, you would use domain knowledge or other criteria to pick $k$.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter5/kmeans_elbow_iris.png" alt="Elbow plot for k-means on Iris" />
  <p class="caption"><strong>Figure 5.1a.</strong> Elbow method for k-means on Iris (petal length, petal width). Inertia (within-cluster sum of squares) is plotted vs number of clusters $k$. The bend near $k=3$ suggests three clusters are a good choice.</p>
</div>

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

**How "closeness" between two clusters is defined (linkage):**

When we have two clusters (e.g., two groups of Iris flowers), we must define a single "distance" between them so we can decide which pair of clusters to merge next. Different **linkage** methods define this differently; all use the same pairwise Euclidean distances between individual flowers.

- **Single linkage:** The distance between cluster A and cluster B is the **smallest** distance between any flower in A and any flower in B. So we look at the two closest flowers (one in A, one in B) and use that distance. Example: if one setosa and one versicolor flower are very close, the two clusters containing them are considered close and may merge early.
- **Complete linkage:** The distance between A and B is the **largest** distance between any flower in A and any flower in B. We use the two farthest flowers, one from each cluster. Clusters stay "far" until even their most distant members are close.
- **Average linkage:** The distance between A and B is the **average** of all pairwise distances (every flower in A with every flower in B). So we average over all cross-cluster pairs. It is a compromise between single and complete.
- **Ward:** We do not use a direct "distance" between A and B. Instead, we merge the two clusters whose merger causes the **smallest increase** in total within-cluster variance (sum of squared distances from each point to its cluster mean). So we merge clusters that, when combined, stay as compact as possible. This is similar in spirit to k-means, which also minimizes within-cluster spread.

**The distance matrix:** We need distances between all pairs of samples. For vectors, Euclidean distance is common:

$$
d(\mathbf{x}_i, \mathbf{x}_j) = \| \mathbf{x}_i - \mathbf{x}_j \| = \sqrt{\sum_{f=1}^{p} (x_{if} - x_{jf})^2}
$$

Here $p$ is the number of features (e.g., $p=2$ for petal length and petal width), and $x_{if}$ is the value of feature $f$ for sample $i$.

**Dendrogram:** The tree shows merge order. The vertical axis is distance (or the linkage value at which clusters merge); cutting the dendrogram at a given height gives a partition into a fixed number of clusters.

**Iris example:** Hierarchical clustering on Iris produces a dendrogram. Cutting at a low height gives many small clusters; cutting higher gives fewer, larger clusters. The figure below shows an example dendrogram and the corresponding cluster assignment in 2D.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter5/hierarchical_iris.png" alt="Hierarchical clustering on Iris" />
  <p class="caption"><strong>Figure 5.2.</strong> Agglomerative hierarchical clustering on Iris (Ward linkage). Left: dendrogram (merge order of the 150 flowers). A horizontal cut at a given height defines how many clusters we get. Right: the same 150 flowers in 2D, colored by the 3-cluster assignment obtained by cutting the dendrogram at a height that gives 3 clusters. Both panels come from the same hierarchical clustering; the right plot is not k-means.</p>
</div>

**Connection between the left and right panels in the above figure (both are hierarchical clustering, not k-means):** In this Iris example, clusters are merged using **Ward linkage**. So at every step, the algorithm merges the two clusters whose merger causes the **smallest increase** in total within-cluster variance (sum of squared distances to cluster means). That merge order is what the dendrogram shows: the **left** plot is the **dendrogram**, where each leaf is one of the 150 flowers and the tree shows the order in which clusters were merged (bottom to top). The **right** plot shows the same 150 flowers in 2D (petal length vs petal width), colored by **cluster label**. Those labels come from **cutting the dendrogram** at a chosen height so that we get exactly 3 clusters. So: (1) run agglomerative clustering with Ward linkage and build the dendrogram (left), (2) choose "3 clusters" by cutting the tree at the height that yields 3 groups, (3) assign each flower to one of those 3 clusters, (4) plot the flowers in 2D and color them by that assignment (right). The three colors on the right are therefore the three clusters from **hierarchical** clustering with Ward linkage (same method as the dendrogram), not from k-means.

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

### 5.2.2 How PCA uses variance: continuing the same example

We just saw that in our 5-sample table, **variance along A** = 2.5 and **variance along B** = 0.625, so **total variance** = 3.125. So far we only measured spread along the **original** axes (the "horizontal" and "vertical" directions). PCA asks a different question: *"Is there a **slanted** direction (a line through the cloud) along which the variance is even bigger?"*

**Back to our numbers.** In the table, notice that B is always exactly half of A (0.5, 1.0, 1.5, 2.0, 2.5). So all five points lie on a single straight line: the line "B = 0.5 × A". If you draw that line and project each point onto it, the projected values are spread out a lot. If you draw the direction **perpendicular** to that line and project the points onto it, they all land almost on top of each other (no spread). So:

- **First principal component (PC1):** the direction along that line. The variance of the data when we project onto this direction is **as large as it can get**; here it actually equals the total variance 3.125 (because all the spread in the data is along that one line).
- **Second principal component (PC2):** the direction perpendicular to PC1. The variance along PC2 is (in this ideal case) zero, because the points have no spread in that direction.

So we **rank** the components by how much variance they capture: PC1 has the **largest** variance (3.125), PC2 has the **next** (0). That is exactly what "component ranking" means: PC1 is first because it has the most variance, PC2 is second, and so on.

**Variance explained ratio.** We already had total variance = 3.125. So:

- Variance explained by PC1 = 3.125 → **ratio** = 3.125 / 3.125 = **100%**.
- Variance explained by PC2 = 0 → **ratio** = 0%.

In a real dataset the points do not lie on a perfect line, so PC1 will capture most but not all of the variance, and PC2 will capture the rest. For example you might get PC1 ≈ 80%, PC2 ≈ 15%, PC3 ≈ 4%, PC4 ≈ 1%. A **scree plot** is just a bar or line plot of these ratios (or the raw variances) in order: PC1, then PC2, then PC3, … So the example above is the same idea: we **rank** components by variance, and the **variance explained ratio** is each component’s variance divided by total variance.

**Three common questions about the example**

**(1) What if one point is off the line?** In our table, we had B exactly half of A for all five samples, so all points lay on one line. Suppose we change **one** value: for sample 3, let B be 2.0 instead of 1.5 (so B is no longer 0.5×A for that row). Then the five points no longer lie on a perfect line; one point is slightly off. In that case:

- **PC1** still points roughly along the "main" direction of the cloud (the line that best fits the points). Most of the variance is still along that direction, so PC1 might explain about 95% or 98% of the total variance.
- **PC2** is still perpendicular to PC1, but now the points have a **little** spread in that perpendicular direction (because one point stuck out). So the variance along PC2 is no longer zero; it might be 2% or 5% of the total. So we get PC1 ≈ 95–98%, PC2 ≈ 2–5%.

So: the more the points deviate from a perfect line, the more variance PC2 (and later components) capture. In the extreme where points are scattered in all directions, variance is spread across many components.

**(2) What if we have many features (not just A and B)?** We do **not** try to make every feature equal to some multiple of A. PCA works in the **full** space of all features. Suppose we have features A, B, C, D (e.g. four measurements per flower). Each sample is then a point in **4-dimensional** space. PC1 is the **single direction** in that 4D space (a line through the origin) along which the data have the largest variance. That direction is a **linear combination** of all features: for example, PC1 might be "0.5×A + 0.4×B + 0.1×C − 0.2×D" (after centering and scaling, the coefficients are the loadings). So we are not collapsing B onto A; we are finding the **best single axis** in the full space, which uses all features at once. With only A and B, that best axis happened to be the line B = 0.5×A. With more features, PC1 is the best line in 3D, 4D, or 100D. PC2 is the best direction perpendicular to PC1 in that same high-dimensional space, and so on. So PCA always gives one direction (PC1), then the next best perpendicular direction (PC2), and so on, using **all** features together.

**(3) Why do we look for PCs in perpendicular directions?** Because we want each new component to capture **new** variance that we have not already counted. After we use PC1, all the variance along that direction is already "used." The **remaining** variance in the data lies in the directions perpendicular to PC1 (everything that is not along PC1). So we restrict our search to that perpendicular space and choose the direction with the **maximum variance there**; that direction is PC2, and it is perpendicular to PC1 by construction. If we did not require perpendicular, we could pick any second direction; but then we might pick something very similar to PC1 and count the same variance twice. By requiring PC2 to be perpendicular to PC1, we ensure that PC2 captures only **new** variance (spread that PC1 did not capture). So perpendicular means "no overlap": each component accounts for a separate part of the total variance, and together they form a clear set of axes (like north–south and east–west on a map).

**Why this matters.** We keep the first few components (e.g. PC1 and PC2) because they contain most of the spread. The later components often add little; we drop them to reduce dimension. So PCA uses the **same** variance we computed in 5.2.1; it just measures that variance along **new** directions (PC1, PC2, …) instead of only along the original features (A, B), and then **ranks** those directions by how much variance they have.

**In one sentence:** PCA finds new axes (PC1, PC2, …) so that the first axis has the largest variance, the second has the next largest and is perpendicular to the first, and so on; we rank components by this variance and use the top ones to summarize the data.

**Math (for reference):** The first component is the unit direction $\mathbf{w}_1$ that maximizes the variance of the projected data:

$$
\mathbf{w}_1 = \arg\max_{\|\mathbf{w}\|=1} \text{Var}(\mathbf{X} \mathbf{w})
$$

The second component is perpendicular to $\mathbf{w}_1$ and maximizes the remaining variance. The variances along these directions are the **eigenvalues** of the covariance matrix; they are the numbers we use for ranking and for the scree plot. The next section explains what eigenvalues and eigenvectors are and how PCA uses them.

### 5.2.3 Eigenvalues and eigenvectors: what they are and how PCA uses them

PCA is computed using **eigenvalues** and **eigenvectors** of the data’s covariance matrix. This section explains what those are and how they give us the principal components.

**What is an eigenvector?**

Suppose we have a square matrix $\mathbf{A}$ (for PCA, this will be the covariance matrix of the centered data). An **eigenvector** of $\mathbf{A}$ is a nonzero vector $\mathbf{v}$ such that when we multiply $\mathbf{A}$ by $\mathbf{v}$, we get a vector that points in the **same direction** as $\mathbf{v}$, just possibly scaled:

$$
\mathbf{A} \mathbf{v} = \lambda \mathbf{v}
$$

The number $\lambda$ (lambda) is called the **eigenvalue** for that eigenvector. So the matrix $\mathbf{A}$ does not rotate $\mathbf{v}$; it only **stretches or shrinks** it by the factor $\lambda$. If $\lambda > 1$, the vector gets longer; if $0 < \lambda < 1$, it gets shorter; if $\lambda < 0$, it flips to the opposite direction and then is scaled.

**Picture: how the eigenvalue changes vector length.**

Imagine a 2D vector $\mathbf{v}$. When we apply the matrix $\mathbf{A}$, we get a new vector $\mathbf{A}\mathbf{v}$. For most vectors, $\mathbf{A}\mathbf{v}$ points in a **different** direction than $\mathbf{v}$. But for **eigenvectors**, $\mathbf{A}\mathbf{v}$ stays on the same line as $\mathbf{v}$: it is just $\lambda$ times $\mathbf{v}$. So the eigenvalue $\lambda$ is the factor by which the length of $\mathbf{v}$ changes (stretch or shrink). The figure below illustrates this: an eigenvector $\mathbf{v}$ and the result $\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$ lie on the same line; the ratio of their lengths is $|\lambda|$.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter5/eigenvalue_eigenvector.png" alt="Eigenvalue and eigenvector: matrix stretches vector along same direction" />
  <p class="caption"><strong>Figure 5.3a.</strong> An eigenvector $\mathbf{v}$ of a matrix $\mathbf{A}$: when we multiply $\mathbf{A}\mathbf{v}$, the result lies on the same line as $\mathbf{v}$, scaled by the eigenvalue $\lambda$. Left: $\lambda > 1$ (stretch). Right: $0 < \lambda < 1$ (shrink). For other vectors (not eigenvectors), $\mathbf{A}$ would change the direction as well.</p>
</div>

**Example: the same 5-sample, 2-feature data from section 5.2.1.**

We had two centered features A and B with variance of A = 2.5, variance of B = 0.625, and total variance = 3.125. The **covariance matrix** of that centered data is a 2×2 square matrix:

$$
\mathbf{\Sigma} = \begin{pmatrix} \text{Var}(A) & \text{Cov}(A,B) \\ \text{Cov}(A,B) & \text{Var}(B) \end{pmatrix} = \begin{pmatrix} 2.5 & 1.25 \\ 1.25 & 0.625 \end{pmatrix}
$$

(The covariance of A and B here is 1.25; it is the same in both off-diagonal positions.) This is the matrix whose eigenvectors and eigenvalues we care about. For this small example you can check that:

- The **eigenvalues** of $\mathbf{\Sigma}$ are $\lambda_1 = 3.125$ and $\lambda_2 = 0$. So the first eigenvalue equals the total variance (all spread is along one line), and the second is zero (no spread perpendicular to that line). That is exactly the variance we attributed to PC1 and PC2 in section 5.2.2.
- The **eigenvector** for $\lambda_1 = 3.125$ points in the direction (2, 1) (or any multiple of it), i.e. the line "B = 0.5 × A" along which the five points lie. So that eigenvector is PC1. The eigenvector for $\lambda_2 = 0$ is perpendicular to that and is PC2.

So the same numbers we used earlier (2.5, 0.625, 3.125) appear in this square matrix; its eigenvalues are the variances of the components, and its eigenvectors are the component directions.

**How eigenvalues and eigenvectors give us PCA (in general).**

For centered data, the **covariance matrix** (like the 2×2 example above) summarizes the variances and covariances of the features. It turns out that:

1. The **eigenvectors** of the covariance matrix are exactly the **principal component directions** (PC1, PC2, …). So the first principal component is the eigenvector with the **largest** eigenvalue, the second is the eigenvector with the **second-largest** eigenvalue, and so on. We rank components by eigenvalue.

2. The **eigenvalues** are exactly the **variances** of the data when projected onto the corresponding eigenvectors. So the eigenvalue for PC1 is the variance along PC1 (the number we plot in the scree plot). The sum of all eigenvalues equals the total variance of the data.

So when we said earlier that "we rank components by variance," that variance is the **eigenvalue** of the covariance matrix for that direction; and the direction itself is the **eigenvector**. PCA is nothing more than: (1) center the data, (2) compute the covariance matrix, (3) find its eigenvectors and eigenvalues, (4) sort by eigenvalue (largest first), and (5) use the top eigenvectors as the new axes (principal components).

**Summary:** Eigenvectors are directions that the matrix only stretches or shrinks (no rotation). Eigenvalues are the stretch factors. For the covariance matrix, the eigenvectors are the PC directions and the eigenvalues are the variances along those directions, so they tell us how to rank and plot the principal components.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter5/pca_eigenvectors_iris.png" alt="PCA: eigenvectors on scatter plot" />
  <p class="caption"><strong>Figure 5.3b.</strong> PCA on Iris (petal length, petal width). The arrows show the two eigenvectors of the covariance matrix (PC1 and PC2). PC1 (largest eigenvalue) points along the direction of maximum variance; PC2 (smaller eigenvalue) is perpendicular. The lengths of the arrows can be drawn proportional to the square root of the eigenvalue (standard deviation along that direction).</p>
</div>

### 5.2.4 Projections

**Projection** means projecting each sample onto a principal component. If $\mathbf{w}_1$ is the first component (a unit vector), the projection of sample $\mathbf{x}_i$ onto PC1 is:

$$
z_{i1} = \mathbf{x}_i \cdot \mathbf{w}_1 = \mathbf{x}_i^\top \mathbf{w}_1
$$

The projected values $z_{i1}, z_{i2}, \ldots$ are the **scores** or **coordinates** in the new PCA space. We can plot samples in 2D using PC1 vs PC2.

**Centering:** PCA is usually applied to **centered** data (subtract the mean of each feature). The components then pass through the center of the cloud.

### 5.2.5 Interpreting PCA for Biologists

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

t-SNE was introduced by Laurens van der Maaten and Geoffrey Hinton. The goal is simple: take high-dimensional points $\mathbf{x}_i$, place them in 2D as $\mathbf{y}_i$, and make neighbors stay neighbors. The clever move is that instead of preserving distances directly, t-SNE preserves **probabilities of being neighbors**. That shift changes everything. Below we peel it apart step by step.

**High-dimensional similarities**

For each point $\mathbf{x}_i$, we define a probability distribution over all other points:

$$
p_{j \mid i} = \frac{\exp\bigl(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / (2\sigma_i^2)\bigr)}{\sum_{k \neq i} \exp\bigl(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / (2\sigma_i^2)\bigr)}.
$$

Break this down slowly. The term $\|\mathbf{x}_i - \mathbf{x}_j\|^2$ is squared distance in high-D. We divide by $2\sigma_i^2$ (a Gaussian width), exponentiate so that closer points get a bigger value, and normalize so the probabilities sum to 1. So for each point $i$, you have a probability distribution over all other points. Interpretation: “If I pick a neighbor of $i$, how likely is it to be $j$?” Close points get high probability; far points get tiny probability.

**Perplexity** controls $\sigma_i$. It roughly corresponds to the effective number of neighbors. Small perplexity means focus on very local structure; large perplexity means look more globally. Typically you set perplexity between 5 and 50.

Then we symmetrize:

$$
p_{ij} = \frac{p_{j \mid i} + p_{i \mid j}}{2n}.
$$

Now $p_{ij}$ is a symmetric joint probability that points $i$ and $j$ are neighbors in high-D. High-D space has been converted into a probability distribution over pairs. We are no longer thinking in geometry; we are thinking in information theory.

**Low-dimensional similarities**

In 2D we define:

$$
q_{ij} = \frac{\bigl(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2\bigr)^{-1}}{\sum_{k \neq \ell} \bigl(1 + \|\mathbf{y}_k - \mathbf{y}_\ell\|^2\bigr)^{-1}}.
$$

Notice what changed: we no longer use a Gaussian. We use a **Student-t distribution with one degree of freedom**. That heavy tail is critical. A Gaussian shrinks rapidly; moderate distances become almost zero probability, which causes crowding—everything piles into the center. The Student-t decays slowly, so moderately distant points still repel each other and clusters spread apart. This “heavy tail trick” is what makes t-SNE work.

**The objective**

We minimize the **Kullback–Leibler (KL) divergence** between the two distributions:

$$
\mathrm{KL}(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}.
$$

KL measures how different two probability distributions are. What does minimizing it do? If $p_{ij}$ is large (points are close in high-D) and $q_{ij}$ is small (far in 2D), the term becomes large—big penalty. If $p_{ij}$ is tiny and $q_{ij}$ is moderate, the penalty is small. So t-SNE cares a lot about preserving **true neighbors**, and cares less about falsely making distant points somewhat close. That is why t-SNE preserves local clusters extremely well, but global distances between clusters can be distorted. It is intentionally biased toward getting neighborhoods right.

**What t-SNE is actually doing**

1. Convert high-D geometry into probabilities.
2. Convert the 2D embedding into probabilities.
3. Move points so the two probability distributions match.

It is not preserving Euclidean structure; it is preserving neighbor likelihoods. Instead of “keep distances,” it says “keep who likes whom.”

**Important caveats**

- t-SNE does **not** preserve global scale. Distances between clusters in 2D are not reliable indicators of true separation in high-D. The algorithm optimizes neighborhood agreement, not metric fidelity. Do not interpret cluster spacing in t-SNE as meaningful geometry; it is a neighborhood map, not a ruler.
- t-SNE is for **visualization only**. Do not use t-SNE output for clustering or downstream analysis.
- **Perplexity** controls how many neighbors each point considers (typically 5 to 50). Low perplexity = very local structure; high perplexity = more global structure.
- Results can vary between runs (random initialization). Use a fixed `random_state` for reproducibility.

**Biological use:** t-SNE is widely used for single-cell RNA-seq, flow cytometry, and other high-dimensional data to visualize cell populations and identify clusters by eye.

### 5.3.2 UMAP (Uniform Manifold Approximation and Projection)

UMAP was introduced by Leland McInnes and John Healy. The phrase "fuzzy simplicial sets on manifolds" sounds mystical, but underneath it is disciplined geometry plus probability. The core idea is simple: high-dimensional data often lie on a lower-dimensional curved surface (a **manifold**). We want to unfold that surface into 2D without tearing it apart too badly. Let us dismantle it step by step.

**Build structure in high dimensions**

You have points $\mathbf{x}_i$ in high-D space. For each point, find its k nearest neighbors. That builds a local neighborhood graph. For two nearby points, compute their distance:
$$
d_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\|
$$

Then UMAP assigns a weight:

$$
w_{ij} = \exp\bigl(-(d_{ij} - \rho_i) / \sigma_i\bigr)
$$

Break this down. Here $d_{ij}$ is the actual distance between the points; $\rho_i$ is the distance from a point to its closest neighbor; and $\sigma_i$ is a local scaling factor.

**Why subtract $\rho_i$?** Because UMAP wants the closest neighbor to have weight about 1. It shifts distances so that the first neighbor effectively sits at zero.

**Why divide by $\sigma_i$?** Different regions of the dataset have different densities. In a dense region, distances are small; in sparse regions, they are larger. The factor $\sigma_i$ rescales things so that each point has roughly the same "effective neighborhood size." The exponential means: close points get weight near 1, far points get weight near 0. This builds a fuzzy graph — not just "connected" or "not connected," but weighted closeness.

Then the graph is symmetrized so similarity between two points is mutual. That gives high-D similarities $p_{ij}$. Step one: encode who is close to whom on the manifold.

**Build similarities in low dimensions**

We assign each point a 2D coordinate. Then we define:
  $$
  q_{ij} \propto \bigl(1 + a\,\|\mathbf{y}_i - \mathbf{y}_j\|^{2b}\bigr)^{-1}
  $$

Break this apart. We take the 2D distance, raise it to power $2b$, multiply by constant $a$, add 1, and take the reciprocal. This curve behaves like: small distance gives value near 1; large distance decays smoothly toward 0. Unlike a Gaussian (used in t-SNE), this function has heavier tails. Distant points do not get squashed as aggressively, which helps preserve more global relationships.

**The optimization**

UMAP minimizes:
  $$
  -\sum_{i,j} \Bigl( p_{ij} \log q_{ij} + (1 - p_{ij}) \log(1 - q_{ij}) \Bigr).
  $$
  This is **binary cross-entropy**. If $p_{ij}$ is high (points are neighbors in high-D), we want $q_{ij}$ high — they should be close in 2D. If $p_{ij}$ is low (not neighbors), we want $q_{ij}$ small — they should be far apart. The second term $(1-p_{ij})\log(1-q_{ij})$ explicitly penalizes false neighbors. That is something t-SNE does less directly, which is why UMAP often spreads clusters in a way that reflects larger-scale structure better.

**Geometrically, what is happening?** You first build a weighted neighborhood graph in high dimensions. Then you move 2D points around so that the 2D graph matches the high-D graph as closely as possible.
**The manifold language**

"Fuzzy simplicial set" sounds like it escaped a topology seminar. In practice it means: instead of "these points are connected," we say "these points are connected with strength 0.83." That fuzzy graph approximates the manifold structure. The embedding tries to preserve that fuzzy connectivity.

**The min_dist parameter**

This controls how close points are allowed to sit in 2D. Low min_dist means points can pack tightly (sharp clusters); high min_dist means clusters spread out more. It does not change the high-D structure; it changes how compressed the embedding is.

**Big-picture intuition**

UMAP is not trying to preserve raw distances. It is trying to preserve neighborhood relationships — the local geometry of the manifold — while still pushing non-neighbors apart enough to maintain larger structure. Think of it like flattening a crumpled brain cortex onto a sheet without gluing together gyri that should not touch. Some distortion is inevitable; the art is choosing where to allow it.

**A quiet philosophical point:** UMAP does not "discover the true 2D structure." It finds one 2D configuration that preserves neighborhood relationships according to its assumptions. Different random seeds give slightly different maps. That is not a flaw; it is the geometry admitting ambiguity. Dimensionality reduction is not truth extraction. It is controlled distortion. The craft lies in distorting the right things.

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
