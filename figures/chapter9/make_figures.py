"""
Generate figures for Chapter 9 (single-cell, Scanpy, ML for cell-type prediction).
Run from this directory: python make_figures.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ----- Figure 9.1: Count matrix (cells x genes) -----
def fig_count_matrix():
    np.random.seed(42)
    n_cells, n_genes = 12, 20
    # Sparse count-like matrix (many zeros)
    X = np.random.negative_binomial(2, 0.3, (n_cells, n_genes))
    X[X > 5] = 0  # sparsify
    X = np.minimum(X, 8)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(X, cmap="Blues", aspect="auto", vmin=0, vmax=8)
    ax.set_xlabel("Genes (features)")
    ax.set_ylabel("Cells (observations)")
    ax.set_title("Single-cell count matrix: cells × genes (sparse)")
    ax.set_xticks([0, n_genes - 1])
    ax.set_xticklabels(["1", str(n_genes)])
    ax.set_yticks([0, n_cells - 1])
    ax.set_yticklabels(["1", str(n_cells)])
    plt.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    plt.savefig("count_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved count_matrix.png")

# ----- Figure 9.2: Scanpy workflow schematic -----
def fig_scanpy_workflow():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    steps = [
        (1, 4, "Raw\ncounts"),
        (2.2, 4, "Normalize\n+ log1p"),
        (3.4, 4, "HVGs"),
        (4.6, 4, "PCA"),
        (5.8, 4, "Neighbors"),
        (7, 4, "UMAP &\nLeiden"),
    ]
    for i, (x, y, label) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((x, y - 0.4), 0.9, 0.8, boxstyle="round,pad=0.05",
                                    facecolor="steelblue", edgecolor="gray", alpha=0.8))
        ax.text(x + 0.45, y, label, ha="center", va="center", fontsize=8)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 1.1, y), xytext=(x + 0.95, y),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.text(5, 2.5, "Scanpy workflow", ha="center", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("scanpy_workflow.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved scanpy_workflow.png")

# ----- Figure 9.3: UMAP with clusters -----
def fig_umap_clusters():
    np.random.seed(42)
    n = 400
    # Three blobs as fake clusters
    c1 = np.random.randn(n // 3, 2) * 0.5 + np.array([1, 2])
    c2 = np.random.randn(n // 3, 2) * 0.5 + np.array([3, 1])
    c3 = np.random.randn(n - 2 * (n // 3), 2) * 0.5 + np.array([2, 3.5])
    X = np.vstack([c1, c2, c3])
    labels = np.array([0] * (n // 3) + [1] * (n // 3) + [2] * (n - 2 * (n // 3)))

    fig, ax = plt.subplots(figsize=(5, 4))
    for k in range(3):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1], s=8, alpha=0.7, label=f"Cluster {k}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP colored by Leiden cluster")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("umap_clusters.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved umap_clusters.png")

# ----- Figure 9.4: Marker dotplot (schematic) -----
def fig_marker_dotplot():
    np.random.seed(42)
    n_genes, n_clusters = 6, 4
    # Fake mean expression and pct expressed
    mean_expr = np.random.rand(n_genes, n_clusters) * 2
    pct = np.random.rand(n_genes, n_clusters) * 0.8 + 0.2

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.set_xlim(-0.5, n_clusters)
    ax.set_ylim(-0.5, n_genes)
    gene_names = [f"Gene{i+1}" for i in range(n_genes)]
    for i in range(n_genes):
        for j in range(n_clusters):
            size = pct[i, j] * 80 + 20
            color = plt.cm.Reds(mean_expr[i, j] / 2.5)
            ax.scatter(j, n_genes - 1 - i, s=size, c=[color], edgecolors="gray", linewidths=0.5)
    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels([f"C{j}" for j in range(n_clusters)])
    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(gene_names[::-1])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Marker gene")
    ax.set_title("Marker genes per cluster (dotplot)")
    plt.tight_layout()
    plt.savefig("marker_dotplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved marker_dotplot.png")

# ----- Figure 9.5: Cell-type ML pipeline -----
def fig_celltype_ml_pipeline():
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 3)
    ax.axis("off")

    # Reference
    ax.add_patch(FancyBboxPatch((0.2, 1.2), 1.4, 1.2, boxstyle="round,pad=0.05",
                                facecolor="lightblue", edgecolor="steelblue"))
    ax.text(0.9, 1.8, "Reference", ha="center", fontweight="bold")
    ax.text(0.9, 1.4, "Labeled cells\n(features + type)", ha="center", fontsize=9)

    # Train
    ax.annotate("", xy=(2.0, 1.8), xytext=(1.6, 1.8),
                arrowprops=dict(arrowstyle="->", color="gray", lw=2))
    ax.add_patch(FancyBboxPatch((2.0, 1.4), 1.2, 0.8, boxstyle="round,pad=0.05",
                                facecolor="wheat", edgecolor="orange"))
    ax.text(2.6, 1.8, "Train\nclassifier", ha="center", fontsize=9)

    # Query
    ax.add_patch(FancyBboxPatch((4.2, 1.2), 1.4, 1.2, boxstyle="round,pad=0.05",
                                facecolor="lavender", edgecolor="purple"))
    ax.text(4.9, 1.8, "Query", ha="center", fontweight="bold")
    ax.text(4.9, 1.4, "New cells\n(features only)", ha="center", fontsize=9)

    # Predict
    ax.annotate("", xy=(5.8, 1.8), xytext=(5.6, 1.8),
                arrowprops=dict(arrowstyle="->", color="gray", lw=2))
    ax.text(5.5, 1.5, "Project\nsame space", fontsize=8, ha="center")
    ax.add_patch(FancyBboxPatch((5.8, 1.4), 1.0, 0.8, boxstyle="round,pad=0.05",
                                facecolor="wheat", edgecolor="orange"))
    ax.text(6.3, 1.8, "Predict\ncell type", ha="center", fontsize=9)

    ax.text(3.5, 0.4, "Cell-type prediction (label transfer)", ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("celltype_ml_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved celltype_ml_pipeline.png")

if __name__ == "__main__":
    fig_count_matrix()
    fig_scanpy_workflow()
    fig_umap_clusters()
    fig_marker_dotplot()
    fig_celltype_ml_pipeline()
    print("All chapter 9 figures done.")
