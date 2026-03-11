# Chapter 8: Sequences in Biology, Encoding, 1D CNNs, RNNs, and Interpretation

## Introduction

In Chapter 7 we used **convolutions** on **images** (2D grids). Many biological problems involve **sequences**: DNA, RNA, and proteins are long strings of letters (nucleotides or amino acids). Sequences have a **natural order** (position 1, 2, 3, …) but no “second dimension” like an image, so we treat them as **1D signals** and use **1D convolutions** and **recurrent** models. This chapter covers: (1) **encoding** DNA and protein sequences into numbers; (2) **1D convolutional neural networks (CNNs)** for detecting local patterns (e.g., motifs); (3) **recurrent neural networks (RNNs)** and **long short-term memory (LSTM)** networks at a conceptual level; (4) **sequence classification** tasks (splice sites, binding, etc.); and (5) **basic model interpretation** so we can see what the model “looked at.”

You do not need a deep background in biology: we only assume that DNA has four letters (A, C, G, T), proteins have about 20 amino-acid letters, and that **motifs** are short, conserved patterns that often have a function (e.g., binding sites).

---

## 8.1 Encoding DNA and Protein Sequences

Raw sequences are **strings of symbols**. Machine learning models need **numeric inputs**. We encode each symbol (or short substring) as a vector or index.

### 8.1.1 One-hot encoding (DNA and protein)

**DNA** has four nucleotides: **A, C, G, T**. We assign each letter a **one-hot vector** of length 4:

| Letter | One-hot |
|--------|--------|
| A | $(1, 0, 0, 0)$ |
| C | $(0, 1, 0, 0)$ |
| G | $(0, 0, 1, 0)$ |
| T | $(0, 0, 0, 1)$ |

**Example:** The short DNA sequence `ACG` becomes three rows (one per position):

- Position 1 (A): $(1, 0, 0, 0)$  
- Position 2 (C): $(0, 1, 0, 0)$  
- Position 3 (G): $(0, 0, 1, 0)$  

So the **encoded matrix** has shape **length $\times$ 4**. For a sequence of length $L$, we get an $L \times 4$ matrix. Each row has exactly one 1; the rest are 0.

**Proteins** use **20 amino acids** (letters like A, R, N, D, C, Q, E, …). The same idea: one letter → one vector of length 20 with a single 1. So one protein sequence of length $L$ becomes an $L \times 20$ matrix.

**Why one-hot?** It is simple, interpretable, and does not impose an order on the letters (unlike assigning A=1, C=2, …). The model can learn which positions and which letters matter.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter8/onehot_dna.png" alt="One-hot encoding of a short DNA sequence" />
  <p class="caption"><strong>Figure 8.1.</strong> One-hot encoding of a short DNA sequence. Each nucleotide (A, C, G, T) is replaced by a row of four numbers with a single 1. The full sequence becomes a matrix of shape (sequence length) × 4.</p>
</div>

### 8.1.2 Index encoding (integer labels)

Sometimes we only assign an **integer index** to each letter (e.g., A→0, C→1, G→2, T→3). The sequence then becomes a **vector of integers** of length $L$. To feed this into a neural network, we usually pass it through an **embedding layer** that maps each index to a learned vector. So we still get a matrix of size $L \times d$ (where $d$ is the embedding dimension). One-hot is a special case of “no learning” in the first layer; embedding lets the model learn a representation.

### 8.1.3 K-mer encoding (optional)

A **k-mer** is a contiguous substring of length $k$. For DNA, there are $4^k$ possible k-mers (e.g., $k=3$ gives 64 triplets: AAA, AAC, …, TTT). We can:

- **Count** how many times each k-mer appears in the sequence → a vector of length $4^k$ (bag-of-k-mers), or  
- **Slide** along the sequence and encode each position by the k-mer starting there (so each position is one of $4^k$ classes, then one-hot or embed).

K-mers capture **local context** (e.g., “what triplets are common near splice sites?”). For this chapter we will mostly use **one-hot (or index + embedding)** so that 1D convolutions can learn motifs directly from the raw encoding.

---

## 8.2 1D convolutional neural networks (CNNs) for motif detection

In Chapter 7 we defined **1D convolution**: a **kernel** (a short vector) slides over a **1D signal**, and at each position we compute the inner product of the kernel with the overlapping segment. For sequences, the “signal” is the **encoded sequence** (e.g., the $L \times 4$ matrix for DNA). We apply convolution **along the sequence dimension** (position 1, 2, …, L).

### 8.2.1 One channel: sequence as a 1D signal per channel

Think of the **one-hot matrix** as having **4 channels** (one per nucleotide). At each position $i$ we have a 4-dimensional vector. A **1D convolutional kernel** that slides along the sequence can have width $k$ (e.g., $k=5$) and **depth 4** (one weight per channel at each of the $k$ positions). So the kernel is like a $k \times 4$ patch. At each position $i$, we take the $k \times 4$ block of the sequence matrix, multiply it element-wise with the kernel, sum everything, and get **one number**. That number is the **activation** of this filter at position $i$. So **one kernel** gives **one 1D feature map** of length (about) $L - k + 1$.

**Interpretation:** The kernel is learning a **local pattern** of length $k$, a **motif**. High activation at position $i$ means “this motif is present around position $i$.” So 1D CNNs naturally do **motif detection**. (We use “CNN” for convolutional neural network throughout.)

### 8.2.2 Example: a tiny sequence and one kernel

**Sequence (length 8):** encoded as 8×4 one-hot. Suppose we have one kernel of **width 3** and depth 4 (total 12 weights). The kernel slides to positions 0, 1, …, 5 (output length $8-3+1=6$). At each position we compute the dot product of the 3×4 window with the kernel. If the kernel has learned “G in the middle, A and T on the sides,” then we get a **peak** in the feature map where that pattern appears in the sequence.

So:

- **Input:** sequence of length $L$, encoded as $L \times 4$ (DNA) or $L \times 20$ (protein).  
- **1D conv:** several kernels (e.g., 32 or 64), each of width $k$ (e.g., 5–15).  
- **Output:** one feature map per kernel; each map has length $\approx L - k + 1$ (depending on padding).  
- Then we often add **rectified linear unit (ReLU)** and **pooling** (e.g., max-pool) to reduce length and combine information.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter8/1dcnn_motif.png" alt="1D CNN sliding over a sequence for motif detection" />
  <p class="caption"><strong>Figure 8.2.</strong> A 1D convolutional kernel (motif detector) slides over the encoded sequence. Each position yields one activation; high activation indicates presence of the learned pattern (motif) at that position.</p>
</div>

### 8.2.3 From motif detection to classification

For **sequence classification** (e.g., “is this region a splice site?” or “does this protein bind?”), we typically:

1. Encode the sequence (e.g., one-hot).  
2. Stack one or more **1D convolution (conv) + ReLU (+ optional pooling)** layers to get a set of feature maps.  
3. **Pool** over the whole sequence (e.g., global max-pool or average-pool) to get one vector per channel.  
4. Pass that vector through **fully connected** layers to get the class (or score).

So the CNN **extracts local motifs**, and the global pool **summarizes** “which motifs appeared anywhere in the sequence” into a fixed-size vector for the classifier.

---

## 8.3 RNNs and LSTMs (Conceptual)

So far we used **1D CNNs** to detect **local** patterns (motifs) in DNA or protein sequences. In many biological problems, though, what matters is not only a short window but **long-range** context: for example, the beginning of a gene can influence splicing or folding much later, and the structure of an RNA molecule depends on base pairing between distant positions. **Recurrent neural networks (RNNs)** and **LSTMs** are sequence models built exactly for this, and they are widely used in machine learning (ML) for DNA, RNA, and protein tasks (e.g., splice site prediction, RNA secondary structure, and protein function). So they belong naturally in this chapter alongside 1D CNNs.

**Recurrent neural networks (RNNs)** are designed for **sequences** where the **order** matters and where **long-range** dependencies can exist (e.g., the start of a gene affecting the end). Unlike a convolutional neural network (CNN), which has a **fixed-size** receptive field (the kernel width), an RNN can in principle use **all previous** positions.

### 8.3.1 The recurrence idea

At each **time step** $t$ (each position in the sequence), the RNN receives:

- The **input** at this position, $\mathbf{x}_t$ (e.g., the one-hot or embedding of the $t$-th nucleotide).  
- A **hidden state** $\mathbf{h}_{t-1}$ from the previous time step.

It computes a **new hidden state**:

$$
\mathbf{h}_t = f(\mathbf{x}_t, \mathbf{h}_{t-1}; \mathbf{W}).
$$

So the same function $f$ and the same weights $\mathbf{W}$ are applied at every step. The hidden state $\mathbf{h}_t$ is a **summary of the sequence so far** (from position 1 to $t$). We can use $\mathbf{h}_t$ to predict a label at step $t$, or use the **last** hidden state $\mathbf{h}_L$ for a single label for the whole sequence.

**Intuition:** The RNN “walks” along the sequence and updates its internal memory ($\mathbf{h}$) at each step. That memory carries information from the past into the future.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter8/rnn_unfold.png" alt="RNN unfolded in time" />
  <p class="caption"><strong>Figure 8.3.</strong> An RNN unfolded in time. At each step $t$, the model takes input $\mathbf{x}_t$ and the previous hidden state $\mathbf{h}_{t-1}$, and outputs a new hidden state $\mathbf{h}_t$. The same weights are reused at every step.</p>
</div>

### 8.3.2 Vanishing gradient and long-range dependence

Training RNNs with **simple** recurrence (e.g., $\mathbf{h}_t = \tanh(\mathbf{W}_{xh} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b})$) runs into the **vanishing gradient** problem: when the sequence is long, gradients from the loss at the end of the sequence shrink to almost zero when they are backpropagated to the beginning. So the model has trouble learning **long-range** dependencies (e.g., “position 5 matters for the label at position 100”).

### 8.3.3 LSTM: a conceptual overview

**Long Short-Term Memory (LSTM)** networks fix this by adding a **cell state** $\mathbf{c}_t$ (in addition to the hidden state $\mathbf{h}_t$) and **gates** that control how much information is **forgotten**, **updated**, and **output** at each step. You do not need the full equations here; the main ideas are:

- **Forget gate:** “How much of the old cell state should we keep?”  
- **Input gate:** “How much of the new candidate update should we add to the cell?”  
- **Output gate:** “How much of the cell state should we expose as the hidden state?”

The **cell state** $\mathbf{c}_t$ is like a “conveyor belt” that can carry information across many time steps with less decay than in a simple RNN. So LSTMs (and later, **gated** variants like the **gated recurrent unit (GRU)**) are the standard choice when we need **long-range** context in sequences.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter8/lstm_cell.png" alt="LSTM cell: gates and cell state" />
  <p class="caption"><strong>Figure 8.4.</strong> LSTM cell (conceptual). The cell state $\mathbf{c}$ is updated by forget and input gates; the hidden state $\mathbf{h}$ is filtered by the output gate. This allows the network to retain information over many steps.</p>
</div>

### 8.3.4 When to use CNN vs RNN for sequences

- **1D CNN:** Good for **local patterns** (motifs, binding sites, short motifs). Fast, parallel over the sequence; receptive field is limited by kernel size (unless you stack many layers).  
- **RNN/LSTM:** Good when **order** and **long-range** context matter (e.g., language, long-range dependencies in a gene). Sequential computation; can in principle use the full history.

In practice, **hybrid** models (e.g., 1D CNN to extract motifs, then LSTM on top) are also common in genomics and proteomics.

---

## 8.4 Sequence Classification Tasks

Here we give a few **concrete tasks** so you can see how encoding + 1D CNN (or RNN) fit together.

### 8.4.1 Splice site prediction

**Goal:** Given a short window of DNA (e.g., 100–400 nucleotides (nt)) around a candidate **donor** or **acceptor** site, predict whether it is a true splice site.

- **Input:** Sequence window (e.g., A, C, G, T string).  
- **Encoding:** One-hot (length × 4).  
- **Model:** 1D CNN (several conv layers + global pool) or CNN + LSTM; output: binary (splice site vs not) or two classes (donor vs acceptor).  
- **Label:** From annotations (known splice sites) or from RNA sequencing (RNA-seq).

### 8.4.2 Transcription factor binding (or motif presence)

**Goal:** Predict whether a short DNA sequence (e.g., 100–500 nt) contains a binding site for a given **transcription factor (TF)**.

- **Input:** Sequence.  
- **Encoding:** One-hot or k-mers.  
- **Model:** 1D CNN or a dedicated motif model; output: binding vs non-binding.  
- **Label:** From chromatin immunoprecipitation followed by sequencing (ChIP-seq) or similar experiments.

The **learned conv filters** often look like **position weight matrices (PWMs)**, i.e., interpretable motifs.

### 8.4.3 Protein secondary structure or function

**Goal:** From the **amino acid sequence** of a protein (or a region), predict secondary structure (helix, sheet, coil) at each position, or a single label (e.g., enzyme vs non-enzyme).

- **Input:** Protein sequence (letters A, R, N, …).  
- **Encoding:** One-hot (length × 20) or embedding.  
- **Model:** 1D CNN (per-position or with global pool) or LSTM/Transformer.  
- **Label:** From structure databases (e.g., the Dictionary of Secondary Structure in Proteins (DSSP)) or functional annotations.

### 8.4.4 Summary table

| Task              | Input        | Typical encoding | Typical model        |
|-------------------|-------------|------------------|-----------------------|
| Splice site       | DNA window  | One-hot (L×4)    | 1D CNN or CNN+LSTM    |
| TF (transcription factor) binding | DNA window  | One-hot / k-mer | 1D CNN                |
| Protein function  | Protein seq| One-hot (L×20)   | 1D CNN or LSTM        |
| Secondary structure | Protein seq| One-hot / embed  | 1D CNN or RNN (per-position) |

---

## 8.5 Basic Model Interpretation

After training, we often want to know **what the model used** to make its decision, which positions or which motifs. This is **interpretability** for sequences.

### 8.5.1 Visualizing learned 1D conv filters as motifs

The **weights** of a 1D convolutional kernel (over one-hot encoded DNA) can be reshaped into a $k \times 4$ matrix. Each row has four values (one per nucleotide). If we **normalize** each row to sum to 1, we get something like a **position weight matrix (PWM)**. Plotting it (e.g., as a **sequence logo**) shows which nucleotides the filter “prefers” at each position, i.e., the **learned motif**.

So the first step of interpretation is: **inspect the first-layer convolution (conv) filters** and turn them into logos or position weight matrices (PWMs).

### 8.5.2 Saliency and gradient-based attribution

**Saliency:** For a given input sequence and the model’s output (e.g., score for “splice site”), compute the **gradient** of that output with respect to the **input** (the one-hot or embedding matrix). The magnitude of the gradient at each position tells us how much a small change there would change the output, so “important” positions get large gradient magnitude. We can plot **saliency per position** along the sequence.

**Limitation:** Gradients can be noisy. Smoothed or integrated variants (e.g., **integrated gradients**) are often used in practice.

### 8.5.3 Attention (if the model uses it)

If the model has an **attention** mechanism (e.g., in a Transformer or an attention layer on top of an RNN), the **attention weights** tell us which positions the model “attended to” when making the prediction. We can plot attention over positions for a given sequence. This is a direct way to see **where** the model looked.

### 8.5.4 In practice

- **1D CNN:** Visualize first-layer (or early-layer) kernels as motifs; use saliency or gradient-based maps to highlight important positions.  
- **RNN/LSTM:** Use hidden-state analysis or attention (if available); otherwise gradient-based attribution.  
- **Hybrid:** Combine filter visualization (CNN part) and attention or gradients (recurrent part).

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter8/interpretation_logo.png" alt="Learned conv filter as sequence logo" />
  <p class="caption"><strong>Figure 8.5.</strong> A learned 1D convolutional filter (first layer) converted into a sequence logo. Height at each position indicates how much the filter “prefers” each nucleotide; this is the model’s learned motif.</p>
</div>

---

## Summary

- **Encoding:** DNA/protein sequences are turned into numbers by **one-hot encoding** (one row per position, 4 or 20 channels) or **index + embedding**. **K-mers** are an alternative for local context.
- **1D CNNs:** A kernel slides along the encoded sequence; each kernel learns a **local pattern (motif)**. Stack conv + ReLU + pooling, then global pool + dense layers for **sequence classification** (e.g., splice site, TF binding).
- **RNNs:** Recurrence: at each step, hidden state $\mathbf{h}_t$ depends on input $\mathbf{x}_t$ and previous $\mathbf{h}_{t-1}$. Good for **order** and **long-range** context; **LSTMs** (and gated recurrent units (GRUs)) use gates to avoid vanishing gradients.
- **Tasks:** Splice site prediction, TF binding, protein structure/function, all use encoded sequences and 1D CNN and/or RNN.
- **Interpretation:** Visualize **convolution (conv) filters** as motifs (e.g., position weight matrix (PWM) or sequence logo); use **saliency** (gradient w.r.t. input) or **attention** to see which positions mattered.

---

## Topics for the Next Class

- **Transformers and attention** for sequences (e.g., BERT-style models for protein or DNA).
- **Generative models** for sequences (e.g., language models for design).

---

## Further Reading

- [Deep learning for biology (Nature, 2021)](https://www.nature.com/articles/d41586-018-02174-z).
- [BPNet (Avsec et al.)](https://www.nature.com/articles/s41592-021-01282-5), interpretable CNNs for TF binding with attribution.
- [Understanding LSTM networks (colah’s blog)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/), conceptual explanation of LSTMs with diagrams.
- [PyTorch: Sequence models and LSTM](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html), RNN/LSTM in code.
- [Kipoi: model zoo for genomics](https://kipoi.org/), pretrained models for splice sites, TF binding, etc., with interpretation tools.
