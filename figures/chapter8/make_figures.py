"""
Generate figures for Chapter 8 (sequences, 1D CNN, RNN/LSTM, interpretation).
Run from this directory: python make_figures.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ----- Figure 8.1: One-hot encoding of DNA -----
def fig_onehot_dna():
    seq = "ACGT"
    letters = list(seq)
    nucs = ["A", "C", "G", "T"]
    L, K = len(letters), len(nucs)
    onehot = np.zeros((L, K))
    for i, c in enumerate(letters):
        onehot[i, nucs.index(c)] = 1

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.imshow(onehot, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(K))
    ax.set_xticklabels(nucs)
    ax.set_yticks(range(L))
    ax.set_yticklabels([f"pos {i+1}" for i in range(L)])
    ax.set_xlabel("Nucleotide (channel)")
    ax.set_ylabel("Position")
    ax.set_title("One-hot encoding: sequence \"ACGT\"")
    for i in range(L):
        for j in range(K):
            ax.text(j, i, "1" if onehot[i, j] == 1 else "0", ha="center", va="center", fontsize=12)
    plt.tight_layout()
    plt.savefig("onehot_dna.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved onehot_dna.png")

# ----- Figure 8.2: 1D CNN motif (kernel sliding over sequence) -----
def fig_1dcnn_motif():
    fig, ax = plt.subplots(figsize=(7, 2.5))
    L = 12
    k = 4
    # Sequence as colored blocks
    np.random.seed(42)
    seq_heights = np.random.rand(L) * 0.4 + 0.6
    for i in range(L):
        ax.add_patch(Rectangle((i, 0), 1, seq_heights[i], facecolor="steelblue", edgecolor="gray"))
    ax.set_xlim(-0.5, L + 1)
    ax.set_ylim(-0.3, 2.2)
    ax.set_xticks(range(L))
    ax.set_xticklabels([f"{i+1}" for i in range(L)], fontsize=9)
    ax.set_ylabel("Encoded\nsequence")
    ax.set_xlabel("Position along sequence")
    # Kernel box
    kw = k * 0.95
    ax.add_patch(FancyBboxPatch((2, 1.2), kw, 0.7, boxstyle="round,pad=0.02",
                                facecolor="orange", edgecolor="darkred", linewidth=2))
    ax.text(2 + kw/2, 1.55, "Kernel (motif)", ha="center", va="center", fontsize=10)
    ax.annotate("", xy=(2 + kw, 0.5), xytext=(2 + kw, 1.2),
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.text(2 + kw + 0.2, 0.85, "activation", fontsize=9, color="gray")
    ax.set_title("1D CNN: kernel slides along sequence → one activation per position")
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("1dcnn_motif.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 1dcnn_motif.png")

# ----- Figure 8.3: RNN unfolded in time -----
def fig_rnn_unfold():
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    n_steps = 4
    box_w, box_h = 1.0, 0.7
    for t in range(n_steps):
        x = 1.5 + t * 2.2
        # x_t
        rect = FancyBboxPatch((x, 2), box_w, box_h, boxstyle="round,pad=0.03",
                              facecolor="lightblue", edgecolor="black")
        ax.add_patch(rect)
        ax.text(x + box_w/2, 2 + box_h/2, r"$\mathbf{x}_t$", ha="center", va="center", fontsize=11)
        # h_t
        rect2 = FancyBboxPatch((x, 0.5), box_w, box_h, boxstyle="round,pad=0.03",
                               facecolor="lightgreen", edgecolor="black")
        ax.add_patch(rect2)
        ax.text(x + box_w/2, 0.5 + box_h/2, r"$\mathbf{h}_t$", ha="center", va="center", fontsize=11)
        # arrow x_t -> h_t
        ax.annotate("", xy=(x + box_w/2, 0.5 + box_h), xytext=(x + box_w/2, 2),
                    arrowprops=dict(arrowstyle="->", lw=1.5))
        if t > 0:
            ax.annotate("", xy=(x, 0.5 + box_h/2), xytext=(x - 0.25, 0.5 + box_h/2),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color="green"))
            ax.text(x - 0.5, 0.5 + box_h/2 + 0.15, r"$\mathbf{h}_{t-1}$", fontsize=9, color="green")
    ax.text(0.3, 2.35, "t=1")
    ax.text(2.5, 2.35, "t=2")
    ax.text(4.7, 2.35, "t=3")
    ax.text(6.9, 2.35, "t=4")
    ax.set_title("RNN unfolded in time (same weights at each step)")
    plt.tight_layout()
    plt.savefig("rnn_unfold.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved rnn_unfold.png")

# ----- Figure 8.4: LSTM cell (conceptual) -----
def fig_lstm_cell():
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 4)
    ax.axis("off")
    # Big box: LSTM cell
    cell = FancyBboxPatch((0.5, 1), 4, 2.5, boxstyle="round,pad=0.05",
                           facecolor="lavender", edgecolor="purple", linewidth=2)
    ax.add_patch(cell)
    ax.text(2.5, 3.2, "LSTM cell", ha="center", fontsize=12, fontweight="bold")
    ax.text(2.5, 2.6, "forget gate · input gate · output gate", ha="center", fontsize=9)
    ax.text(2.5, 1.8, r"cell state $\mathbf{c}_t$  →  hidden state $\mathbf{h}_t$", ha="center", fontsize=10)
    # Arrows in
    ax.annotate("", xy=(1, 2.25), xytext=(0.2, 2.25), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.1, 2.4, r"$\mathbf{x}_t$", fontsize=11)
    ax.annotate("", xy=(1, 1.5), xytext=(0.2, 1.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="green"))
    ax.text(0.1, 1.2, r"$\mathbf{h}_{t-1}$", fontsize=10, color="green")
    # Arrow out
    ax.annotate("", xy=(4.8, 2.25), xytext=(4, 2.25), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(4.95, 2.25, r"$\mathbf{h}_t$", fontsize=11)
    ax.set_title("LSTM: gates control flow of information (conceptual)")
    plt.tight_layout()
    plt.savefig("lstm_cell.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved lstm_cell.png")

# ----- Figure 8.5: Interpretation – sequence logo (simplified PWM bar chart) -----
def fig_interpretation_logo():
    # Fake PWM: 6 positions, 4 nucleotides (from a learned filter)
    pwm = np.array([
        [0.1, 0.8, 0.05, 0.05],
        [0.05, 0.1, 0.8, 0.05],
        [0.05, 0.05, 0.1, 0.8],
        [0.8, 0.05, 0.1, 0.05],
        [0.1, 0.1, 0.1, 0.7],
        [0.05, 0.7, 0.15, 0.1],
    ])
    nucs = ["A", "C", "G", "T"]
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
    fig, ax = plt.subplots(figsize=(5, 2.5))
    pos = np.arange(pwm.shape[0])
    left = np.zeros(pwm.shape[0])
    for j in range(4):
        ax.barh(pos, pwm[:, j], left=left, color=colors[j], height=0.6, label=nucs[j])
        left += pwm[:, j]
    ax.set_yticks(pos)
    ax.set_yticklabels([f"pos {i+1}" for i in pos])
    ax.set_xlabel("Probability (learned filter → PWM)")
    ax.set_ylabel("Position in motif")
    ax.set_title("Learned 1D conv filter as motif (PWM / sequence logo style)")
    ax.legend(loc="upper left", ncol=4, fontsize=9)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig("interpretation_logo.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved interpretation_logo.png")

if __name__ == "__main__":
    fig_onehot_dna()
    fig_1dcnn_motif()
    fig_rnn_unfold()
    fig_lstm_cell()
    fig_interpretation_logo()
    print("All chapter 8 figures generated.")
