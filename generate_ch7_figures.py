"""
Generate figures for Chapter 7 (Convolutions, CNNs).
Run from project root. Saves to ML_book_easy/figures/chapter7/
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import os

base = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(base, "figures", "chapter7")
os.makedirs(out_dir, exist_ok=True)

# ========== Figure 7.1: 1D convolution ==========
fig, ax = plt.subplots(figsize=(10, 4))

# Signal
x = np.array([1, 2, 3, 4, 5])
# Kernel
w = np.array([1, 0, -1])
n, k = len(x), len(w)

# Draw signal as bars with labels
x_pos = np.arange(n)
bars = ax.bar(x_pos - 0.35, x, width=0.35, color="steelblue", label="Signal x", align="edge")
ax.set_xticks(x_pos)
ax.set_xticklabels([f"x{i}" for i in range(n)])
ax.set_ylim(0, 6)
ax.set_ylabel("Value")
ax.set_xlabel("Index")
ax.set_title("1D Convolution: Signal x = [1,2,3,4,5], Kernel w = [1,0,-1]")
ax.legend(loc="upper right")

# Annotate one position: kernel aligned at index 1
ax.annotate("Kernel here →", xy=(1.5, 5.5), fontsize=11, color="darkgreen")
ax.annotate("y[1] = 2·1 + 3·0 + 4·(-1) = -2", xy=(1.5, 5.0), fontsize=10, color="darkgreen")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "conv1d_demo.png"), bbox_inches="tight", dpi=150)
plt.close()
print("Saved conv1d_demo.png")

# ========== Figure 7.2: 2D convolution ==========
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

# Input 4x4
X = np.array([
    [1, 2, 1, 0],
    [0, 1, 2, 1],
    [1, 0, 1, 2],
    [2, 1, 0, 1]
])
# Kernel 3x3 (vertical edge)
W = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

# Left: input
im0 = axes[0].imshow(X, cmap="Blues", vmin=0, vmax=2.5)
axes[0].set_xticks(np.arange(4))
axes[0].set_yticks(np.arange(4))
axes[0].set_xticklabels(np.arange(4))
axes[0].set_yticklabels(np.arange(4))
for i in range(4):
    for j in range(4):
        axes[0].text(j, i, str(X[i, j]), ha="center", va="center", color="black", fontsize=14)
axes[0].set_title("Input X (4×4)")
axes[0].set_xlabel("Column")

# Middle: kernel
im1 = axes[1].imshow(W, cmap="RdYlBu_r", vmin=-1, vmax=1)
axes[1].set_xticks(np.arange(3))
axes[1].set_yticks(np.arange(3))
axes[1].set_xticklabels(np.arange(3))
axes[1].set_yticklabels(np.arange(3))
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, str(W[i, j]), ha="center", va="center", color="black", fontsize=14)
axes[1].set_title("Kernel W (3×3)\n(vertical edge)")
axes[1].set_xlabel("Column")

# Right: output (2x2) from convolution
Y = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        patch = X[i:i+3, j:j+3]
        Y[i, j] = np.sum(patch * W)
im2 = axes[2].imshow(Y, cmap="Greens", vmin=Y.min()-0.5, vmax=Y.max()+0.5)
axes[2].set_xticks(np.arange(2))
axes[2].set_yticks(np.arange(2))
axes[2].set_xticklabels(np.arange(2))
axes[2].set_yticklabels(np.arange(2))
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, f"{Y[i,j]:.0f}", ha="center", va="center", color="black", fontsize=14)
axes[2].set_title("Output feature map Y (2×2)\nconv(X, W)")
axes[2].set_xlabel("Column")

plt.suptitle("2D Convolution: kernel slides over input, each position → one output value", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "conv2d_demo.png"), bbox_inches="tight", dpi=150)
plt.close()
print("Saved conv2d_demo.png")

print("Done. Figures saved to", out_dir)
