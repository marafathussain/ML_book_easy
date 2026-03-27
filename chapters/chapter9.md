# Chapter 9: Single-Cell RNA-seq, Scanpy Workflow, and Machine Learning

## Introduction

In Chapters 7 and 8 we worked with **images** (2D grids) and **sequences** (DNA, protein). Many biological questions are about **cell types** and **cell states**: which cells are T cells, neurons, or tumor cells, and how do they differ in gene expression? **Single-cell RNA sequencing (scRNA-seq)** measures the abundance of messenger RNA (mRNA) in **individual cells**, so we get a **profile** of which genes are turned on or off in each cell. That gives a matrix: **cells × genes**. The number of cells can be thousands to millions; the number of genes is typically tens of thousands. This data is **high-dimensional** and **sparse** (many zeros, because not every gene is expressed in every cell). In this chapter we cover: (1) **what single-cell data looks like** and how it is stored; (2) the **Scanpy workflow**: normalization, highly variable genes (HVGs), neighborhood graph, and UMAP; (3) **clustering and marker genes** to discover and name cell populations; (4) **integrating scRNA with machine learning** so we can treat the data as a feature matrix for classifiers; and (5) **using ML for cell-type prediction** (e.g., training a classifier on labeled cells and predicting labels for new cells or new datasets).

You do not need prior experience with single-cell data: we introduce the **AnnData** object, the standard Python container for such data, and walk through a typical analysis pipeline step by step.

---

## 9.1 Introduction to Single-Cell Data

### 9.1.1 What is single-cell RNA-seq?

In **bulk RNA-seq**, we take a tissue or a population of cells, extract all RNA, and sequence it. The result is an **average** gene expression over many cells. We lose information about **which cell** expressed which gene. In **single-cell RNA-seq**, we isolate **individual cells** (or nuclei), capture their RNA, and sequence each cell separately. So we get one **expression profile** per cell: a vector of (typically) thousands of gene counts.

**End-to-end flowchart (from one cell to the expression matrix):**

```text
[1] SINGLE CELL (one genome / DNA)
        |
        |  Transcription (different genes expressed)
        v
[2] mRNA POOL inside the cell
    - mRNA from Gene A
    - mRNA from Gene B
    - mRNA from Gene C
    (all from SAME DNA, different genes)
        |
        |  Cell is encapsulated in droplet (GEM)
        v
[3] GEM (Gel Bead-in-Emulsion)
    Contains:
      - One cell
      - One bead with MANY identical oligos
        |
        |  Cell lysis -> mRNA released
        v
[4] OLIGO CAPTURE (on bead)

Each oligo structure:
[Primer handle] - [Cell Barcode] - [UMI] - [Poly(dT)]

        |
        +-- Poly(dT) binds -> mRNA poly(A) tail
        |
        +-- Different oligos capture different mRNAs:
        |     Oligo 1 -> Gene A mRNA
        |     Oligo 2 -> Gene B mRNA
        |     Oligo 3 -> Gene A mRNA (another copy)
        |
        v
[5] REVERSE TRANSCRIPTION -> cDNA (KEY STEP)

Each captured mRNA becomes:

cDNA molecule with:
    - Cell barcode (same for all in this GEM)
    - UMI (unique per molecule)
    - Gene-specific sequence

THIS is where RNA -> cDNA happens

        |
        v
[6] BREAK DROPLETS + POOL ALL cDNA

Now:
- cDNA from ALL cells are mixed together
- But labels preserve identity:
    - Cell barcode -> which cell
    - UMI -> which molecule

        |
        v
[7] PCR AMPLIFICATION

- Uses primer handle
- Amplifies cDNA
- Creates duplicates (important later for UMI correction)

        |
        v
[8] SEQUENCING

Each molecule generates reads:

Read 1:
    -> Cell Barcode + UMI

Read 2:
    -> cDNA sequence (maps to gene)

        |
        v
[9] BIOINFORMATICS PROCESSING

Step 1: Group by CELL BARCODE
    -> reconstruct each cell

Step 2: Map Read 2 to genes
    -> identify Gene A, B, C...

Step 3: Collapse UMIs
    -> remove PCR duplicates
    -> count TRUE molecules

        |
        v
[10] FINAL OUTPUT: GENE EXPRESSION MATRIX

            Cell 1   Cell 2   Cell 3
Gene A        10        2        0
Gene B         5        8        1
Gene C         0        3        7

Each value = number of UNIQUE UMIs
(true mRNA molecules per gene per cell)
```

### 9.1.2 Droplet-based scRNA-seq chemistry: GEM bead, barcode, UMI, and cDNA

Many popular single-cell RNA-seq protocols are **droplet-based** (for example, 10x Genomics). A droplet contains a single cell (ideally) and a bead inside a GEM (Gel Bead-in-Emulsion). The bead carries many copies of the same oligo, which acts like a capture probe with multiple parts:

- **Primer handle**: a known DNA sequence that enables PCR amplification later.
- **Cell barcode**: a sequence shared across all oligos on that bead, so sequencing reads can be assigned back to the correct cell.
- **UMI (Unique Molecular Identifier)**: a short random sequence attached to each captured molecule, so PCR duplicates can be collapsed to estimate true molecule counts.
- **`poly(dT)` tail**: a stretch of T’s that binds the **poly(A) tail** of mRNA, so mainly mRNA is captured, and reverse transcription starts near the 3' end in common 3' protocols.

Step-by-step intuition:

1. A cell is lysed inside the GEM and releases its mRNA molecules.
2. The mRNA poly(A) tail hybridizes to `poly(dT)` on bead oligos.
3. Reverse transcription converts each captured RNA molecule into **cDNA** while attaching the **cell barcode** and **UMI** so the molecule is labeled with “which cell” and “which original molecule”.
4. Droplets are broken and all cDNA is pooled together. PCR amplification uses the primer handle to make enough DNA for sequencing.
5. In sequencing, **Read 1** provides the cell barcode plus UMI, and **Read 2** provides the cDNA sequence that is mapped back to a gene.

In bioinformatics we then:

- Group reads by **cell barcode** to reconstruct a per-cell profile.
- Deduplicate using **UMIs** to remove PCR duplicates.
- Aggregate unique UMIs per gene per cell to form the final **cells × genes** count matrix.

Important clarification about “same bead”: in standard 3' droplet protocols, the barcode is not there because different oligos capture different parts of the same mRNA. Instead, the bead captures many different mRNA molecules (often from different genes) from the same cell. The **barcode** identifies the cell origin, and the **UMI** identifies each original captured molecule.

QC note: if a droplet captures more than one cell (a **doublet**), barcodes and UMIs mix across cells, which can create misleading profiles. This is one reason QC filtering is essential before running Scanpy.


**Example:** Suppose we have 3,000 cells and 20,000 genes. The **raw count matrix** has shape **3,000 × 20,000**: rows = cells, columns = genes. Entry $(i, j)$ is the number of RNA molecules (or reads) assigned to gene $j$ in cell $i$. These counts are **non-negative integers** and are often **sparse**: many genes have zero counts in many cells, because only a subset of genes is active in any given cell.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter9/count_matrix.png" alt="Single-cell count matrix: cells × genes" />
  <p class="caption"><strong>Figure 9.1.</strong> The single-cell count matrix. Rows are cells (observations), columns are genes (features). Each entry is a raw count; the matrix is sparse (many zeros).</p>
</div>



### 9.1.3 Why is the data challenging?

- **High dimension:** Tens of thousands of genes → each cell is a point in a very high-dimensional space. We need **dimensionality reduction** and **feature selection** to visualize and cluster.
- **Sparsity:** Many genes have zero counts in many cells (dropouts, or true absence of expression). Models and distances must handle zeros sensibly.
- **Library size:** Different cells have different total counts (different sequencing depth). We **normalize** so that cells are comparable.
- **Batch effects:** Cells from different experiments, donors, or batches can cluster by technical origin rather than biology. **Batch correction** or **integration** is often needed when combining datasets.

### 9.1.4 The AnnData object

Single-cell data in Python is usually stored in an **AnnData** object (from the `anndata` package). Think of it as a **table with extra layers**:

- **`X`:** The main matrix (cells × genes). Can be raw counts or normalized/log-transformed values.
- **`obs`:** **Observations** = rows = cells. A DataFrame with **cell-level metadata** (e.g., batch, donor, total counts, number of genes detected).
- **`var`:** **Variables** = columns = genes. A DataFrame with **gene-level metadata** (e.g., gene name, whether it is a highly variable gene).
- **`obsm`:** Multi-dimensional **annotations** for observations (e.g., PCA coordinates, UMAP coordinates), one array per key.
- **`uns`:** Unstructured **metadata** (e.g., parameters used for PCA or UMAP).

This structure is similar to **Pandas** (rows and columns with metadata) but optimized for large matrices and for the Scanpy ecosystem. Saving to disk gives an **`.h5ad`** file, which can hold millions of cells efficiently.

**Example (conceptual):** After loading a dataset, you might have `adata` with `adata.X` of shape (3000, 20000), `adata.obs['n_genes_by_counts']` (how many genes per cell), and `adata.var['gene_ids']` (gene identifiers). Scanpy functions take `adata`, modify it in place (e.g., add normalized values, PCA, UMAP), and you keep working with the same object.

---

## 9.2 Scanpy Workflow: Normalization, HVGs, Neighbors, UMAP

**Scanpy** is a Python library for single-cell analysis. A typical workflow has these steps: **quality control (QC)** → **normalization** → **log transform** → **highly variable genes (HVGs)** → **scale (optional)** → **PCA** → **neighborhood graph** → **UMAP** → **clustering**. We outline each step and what it does.

### 9.2.1 Quality control and filtering

Before any modeling, we usually **filter** cells and genes:

- **Cells:** Remove cells with too few total counts (likely broken or empty) or too many (possible doublets). Also filter by number of genes detected (e.g., keep cells with at least 200 genes).
- **Genes:** Remove genes that are detected in very few cells (e.g., in fewer than 3 cells); they add noise and slow computation.

These thresholds depend on the experiment; Scanpy and tutorials provide sensible defaults.

### 9.2.2 Normalization

Raw counts are **not comparable across cells**: one cell might have 5,000 total reads, another 20,000. We **normalize** so that each cell has the same total count (e.g., 10,000 counts per cell). So we **scale** the counts in each cell so that they sum to a target (e.g., 10,000). Formula: for cell $i$ with total count $T_i$, each count $x_{ij}$ becomes $x_{ij} \cdot (\text{target} / T_i)$. This is **library-size normalization**.

**In Scanpy:** `sc.pp.normalize_total(adata, target_sum=1e4)` does this. After this step, the matrix still has raw-like scale; the next step brings it to a log scale.

### 9.2.3 Log transformation

Gene expression is often **multiplicative** (e.g., “twice as many transcripts”). Taking **log** makes differences **additive** and stabilizes variance. The usual choice is **log1p**: $\log(1 + x)$, so zeros stay zero and positive values are compressed.

**In Scanpy:** `sc.pp.log1p(adata)` replaces the values in `adata.X` with $\log(1 + x)$. After this, we work in **log-normalized** space for PCA and neighbors.

### 9.2.4 Highly variable genes (HVGs)

Not all genes are informative for **separating cell types**. Many genes are lowly expressed or similar across cells. We **select a subset of genes** that vary a lot across cells, **highly variable genes (HVGs)**, and use only those for dimensionality reduction and clustering. That reduces noise and computation and focuses on biology.

**Idea:** For each gene, we compute (e.g.) **mean** expression across cells and **dispersion** (variance/mean or similar). Genes that are both expressed enough and variable (e.g., high dispersion for their mean) are marked as HVGs. Scanpy can use methods like **Seurat** or **Seurat v3** to select a few thousand HVGs.

**In Scanpy:** `sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3')` (or similar) adds a column in `adata.var` marking which genes are HVGs. Downstream steps (PCA, neighbors) can then use only `adata[:, adata.var['highly_variable']]` or the pipeline can subset automatically.

### 9.2.5 Scaling (optional)

Sometimes we **scale** the data so that each gene has zero mean and unit variance across cells. That makes distances in PCA space more balanced across genes. Not all workflows do this; Scanpy tutorials often do it before PCA with `sc.pp.scale(adata, max_value=10)` (capping large values to avoid outliers dominating).

### 9.2.6 PCA

We reduce the **gene space** (thousands of dimensions) to a smaller number of **principal components (PCs)**, e.g., 50. PCA finds linear combinations of genes that capture the most variance. We use the **HVG matrix** (or the full log-normalized matrix) as input.

**In Scanpy:** `sc.tl.pca(adata, svd_solver='arpack', n_comps=50)` computes PCA and stores the cell embeddings in `adata.obsm['X_pca']`. The result is a matrix of shape (n_cells, 50).

### 9.2.7 Neighborhood graph

To **cluster** cells and build a **UMAP**, we need a notion of **similarity** between cells. We build a **neighborhood graph**: each cell is a node; we connect each cell to its **nearest neighbors** in PCA space (e.g., 15 neighbors). Distance is usually **Euclidean** in the first 50 PCs. This graph encodes “which cells are close.”

**In Scanpy:** `sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)` builds the graph and stores it in `adata.obsp` (e.g., `adata.obsp['connectivities']`). The same graph is used for **Leiden** (or Louvain) clustering and for UMAP.

### 9.2.8 UMAP

**UMAP** (Uniform Manifold Approximation and Projection) is a **non-linear** dimensionality reduction method. It takes the neighborhood graph (or the high-dimensional data) and embeds the cells into **2D** (or 3D) so that cells that are neighbors in the graph stay close in the 2D plot. We use UMAP for **visualization**: each point is a cell, and we can color by cluster, cell type, or gene expression.

**In Scanpy:** `sc.tl.umap(adata)` computes the embedding and stores it in `adata.obsm['X_umap']`. Plotting is done with `sc.pl.umap(adata, color='leiden')` (or any column in `adata.obs` or a gene name).

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter9/scanpy_workflow.png" alt="Scanpy workflow: normalization, HVGs, PCA, neighbors, UMAP" />
  <p class="caption"><strong>Figure 9.2.</strong> The Scanpy workflow. Raw counts → normalize total → log1p → select HVGs → (scale) → PCA → neighborhood graph → UMAP and clustering. The same graph feeds both Leiden/Louvain and UMAP.</p>
</div>

**Minimal code example (conceptual):**

```python
import scanpy as sc

# Assume adata is loaded (e.g., from 10x or an .h5ad file)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3')
sc.tl.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
sc.tl.umap(adata)
```

After this, `adata.obsm['X_umap']` holds the 2D UMAP coordinates and we can cluster (next section).

---

## 9.3 Clustering and Marker Genes

### 9.3.1 Clustering: Leiden and Louvain

The **neighborhood graph** groups cells that are similar in gene expression. **Clustering** assigns each cell to a **group** (cluster) so that cells in the same cluster are more connected in the graph. Two standard algorithms in Scanpy are **Louvain** and **Leiden**. Both optimize a **modularity**-like objective: we want many edges inside clusters and fewer between clusters. **Leiden** often gives better connected clusters and is the default in many workflows.

**In Scanpy:** `sc.tl.leiden(adata, resolution=1.0)` adds a column `adata.obs['leiden']` with cluster labels (0, 1, 2, …). The **resolution** parameter controls how many clusters you get: higher resolution → more, smaller clusters.

We can then **visualize** the clusters on the UMAP: `sc.pl.umap(adata, color='leiden')`. Each cluster is a cloud of points; ideally they correspond to **cell types** or **cell states**.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter9/umap_clusters.png" alt="UMAP colored by Leiden clusters" />
  <p class="caption"><strong>Figure 9.3.</strong> UMAP of single-cell data colored by Leiden cluster. Each cluster can later be annotated as a cell type using marker genes.</p>
</div>

### 9.3.2 Marker genes

Once we have clusters, we ask: **which genes are differentially expressed** in one cluster compared to the rest (or between two clusters)? Those genes are **marker genes**: they help **identify** the cluster (e.g., “cluster 3 has high CD3D and CD8A → likely CD8+ T cells”).

**Methods:** Scanpy provides `sc.tl.rank_genes_groups(adata, groupby='leiden', method='wilcoxon')` (or 'logreg', 't-test') to rank genes per cluster. For each cluster we get a list of genes with scores and p-values; we take the **top genes** as markers.

**Visualization:** We can plot a **dotplot** or **heatmap**: rows = marker genes, columns = clusters (or cells); color or size indicates expression level. That makes it easy to see “cluster 2 is high in gene A, B, C.”

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter9/marker_dotplot.png" alt="Marker gene dotplot across clusters" />
  <p class="caption"><strong>Figure 9.4.</strong> Marker genes per cluster. Dotplot or heatmap: each cluster has a characteristic set of highly expressed genes used for cell-type annotation.</p>
</div>

### 9.3.3 Cell-type annotation

**Annotation** is the step where we **name** the clusters: “cluster 0 = T cells, cluster 1 = B cells, …” We use **prior knowledge** (marker genes from the literature, canonical cell-type markers) and the marker list we computed. There is no single “correct” label; it is a mix of biology and convention. Later we will see how **machine learning** can automate or assist this by training a classifier on labeled reference data and predicting labels for new cells.

---

## 9.4 Integrating scRNA with Machine Learning

Single-cell data, after preprocessing, is a **matrix**: cells (samples) × genes (features). So we can treat it like any **feature matrix** for supervised or unsupervised learning.

### 9.4.1 AnnData as a feature matrix

- **Observations (cells)** = rows = samples. Each row is one cell.
- **Variables (genes)** = columns = features. We can use **all genes** or **HVGs** (or a custom set).
- **X** (or a layer like `adata.layers['log1p']`) = the numeric matrix we feed to a model.

So we have:
- **X:** shape (n_cells, n_genes), the “design matrix.”
- **y:** labels, if we have them, e.g., `adata.obs['cell_type']` or `adata.obs['leiden']`.

We must be careful about **train/test split**: if we split **randomly**, we might leak information (same donor or batch in both sets). Often we split by **batch** or **donor** so the model is evaluated on “unseen” batches. We also need to handle **missing labels**: many cells have no annotation until we run clustering and manual annotation; we can train on a **labeled subset** and predict on the rest.

### 9.4.2 Typical ML tasks

1. **Classification:** Predict a discrete label (e.g., cell type, disease vs healthy) from gene expression. Input: log-normalized (or scaled) expression vector per cell. Model: logistic regression, random forest, or a small neural network. This is the basis of **cell-type prediction** (Section 9.5).
2. **Dimensionality reduction / representation learning:** Autoencoders or other methods can learn a **latent representation** of cells; we can use it for visualization, clustering, or as input to a downstream classifier.
3. **Batch correction:** We can use **integration** methods (e.g., Harmony, scVI, or Scanpy’s `sc.external.pp.harmony_integrate`) to align batches in a shared space; then we run the same Scanpy pipeline (neighbors, UMAP, clustering) on the corrected data.

### 9.4.3 Pipeline sketch

A simple pipeline that combines Scanpy and ML:

1. **Preprocess with Scanpy:** QC, normalize, log1p, HVGs, (optional) scale. Optionally run PCA and store `adata.obsm['X_pca']`.
2. **Define features:** Use `adata.X` (on HVGs) or `adata.obsm['X_pca']` (first 50 PCs) as the feature vector per cell. Using PCs is often more stable (fewer dimensions, denoised).
3. **Define labels:** From manual annotation, a reference, or Leiden clusters (for semi-supervised or transfer).
4. **Train/validation/test split:** Split cells (e.g., by batch or randomly, depending on the question).
5. **Train a classifier:** Fit a model (e.g., logistic regression or MLP) on the training set.
6. **Evaluate:** Accuracy, F1, or per-class metrics on the test set. If we have a separate **new dataset**, we can predict labels on it (cell-type transfer).

---

## 9.5 Using ML for Cell-Type Prediction

A central application is **cell-type prediction**: given a **reference** dataset with labeled cell types, train a classifier and **predict** the cell type for **new cells** (e.g., from a new experiment or a new donor). This is sometimes called **label transfer** or **automatic annotation**.

### 9.5.1 Setting

- **Reference:** Single-cell matrix with labels (e.g., `cell_type`: T cell, B cell, monocyte, …). We have many cells and trust the labels (from experts or a curated atlas).
- **Query:** New cells (new matrix) **without** labels. We want to assign each query cell to one of the reference cell types.

We **train** a classifier on the reference (features = gene expression or PCs; target = cell type). We **predict** on the query: for each query cell we get a predicted label (and often a **probability** or confidence score).

### 9.5.2 Features and models

- **Features:** Log-normalized expression of HVGs, or the first 50 PCs of the reference. For the **query**, we must use the **same** genes or the **same** PCA space (project query into the reference’s PCA space) so that dimensions match. Scanpy and tools like **scvi-tools** or **scanpy’s ingest** can do this.
- **Models:** Simple classifiers (logistic regression, random forest, k-NN) often work well because the representation (PCs or HVGs) is already informative. Neural networks (MLPs) can be used for larger atlases and more classes. **Weighted k-NN** on the reference graph is another option: for each query cell, find k nearest reference cells and vote on the label.

### 9.5.3 Practical considerations

- **Batch effect:** If the query comes from a different batch or technology, expression can shift. **Integration** (e.g., map query into reference PCA/UMAP with `sc.tl.ingest` or use batch-corrected embeddings) helps before prediction.
- **Unseen cell types:** The query might contain a cell type not present in the reference. The classifier will force a label; we can use **confidence scores** (e.g., max softmax probability) to flag low-confidence predictions and send them for manual review.
- **Hierarchies:** Cell types can be hierarchical (e.g., lymphocyte → T cell → CD8+ T cell). We can train a **hierarchical** classifier or multiple classifiers at different granularities.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter9/celltype_ml_pipeline.png" alt="ML pipeline for cell-type prediction" />
  <p class="caption"><strong>Figure 9.5.</strong> Cell-type prediction pipeline. Reference data (labeled) is used to train a classifier; query data (unlabeled) is projected into the same feature space and the classifier predicts cell types. Optional: integration/batch correction before prediction.</p>
</div>

### 9.5.4 Example workflow (conceptual)

1. Load reference: `adata_ref` with `adata_ref.obs['cell_type']`.
2. Preprocess reference: normalize, log1p, HVGs, PCA. Save the PCA model and HVG list.
3. Load query: `adata_query`. Subset to the **same** HVGs, normalize and log1p in the same way, then **project** into the reference PCA (so query cells get coordinates in reference PC space).
4. Train: Use `adata_ref.obsm['X_pca']` and `adata_ref.obs['cell_type']` to train a classifier (e.g., logistic regression or k-NN).
5. Predict: Use the query’s projected PCA coordinates as input; get predicted `cell_type` for each query cell.
6. Optionally: Map query onto reference UMAP for visualization and assign the same colors by predicted type.

Tools like **Scanpy’s `sc.tl.ingest`** automate the projection and allow transfer of labels from reference to query in a consistent way; then you can use the transferred labels as-is or refine them with a separate classifier.

---

## Summary

- **Single-cell data** is a **cells × genes** count matrix: high-dimensional, sparse, and affected by library size and batch. It is stored in **AnnData** objects (`.h5ad`).
- **Scanpy workflow:** Normalize total → log1p → select **HVGs** → (scale) → **PCA** → **neighborhood graph** → **UMAP** and **Leiden** (or Louvain) clustering. The same graph drives both clustering and UMAP.
- **Clustering and markers:** Leiden/Louvain assign each cell to a cluster; **rank_genes_groups** finds **marker genes** per cluster; we use them for **cell-type annotation**.
- **ML integration:** The preprocessed matrix (or PCA) is a **feature matrix**; we can train classifiers for **cell-type prediction** (label transfer from reference to query). Care is needed for **batch effects**, **train/test split**, and **unseen cell types**.

---

## Topics for the Next Class

- **Transformers and attention** for sequences and biology (e.g., protein and DNA language models).
- **Advanced single-cell** topics: multi-omics, trajectory inference, and large-scale atlases.

---

## Further Reading

- [Scanpy documentation: Preprocessing and clustering 3k PBMCs](https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html), step-by-step tutorial.
- [AnnData: Annotated data](https://anndata.readthedocs.io/), structure and usage.
- [UMAP: Uniform Manifold Approximation and Projection](https://umap-learn.readthedocs.io/), for dimensionality reduction and visualization.
- [Single-cell best practices](https://www.sc-best-practices.org/) (online book), QC, normalization, integration, and annotation.
- [Cell typist and other automatic annotation tools](https://github.com/Teichlab/celltypist), for cell-type prediction from reference atlases.
