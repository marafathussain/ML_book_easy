# Chapter 8: Sequences in Biology, Encoding, 1D CNNs, RNNs, and Interpretation

## Introduction

In Chapter 7 we used **convolutions** on **images** (2D grids). Many biological problems involve **sequences**: DNA, RNA, and proteins are long strings of letters (nucleotides or amino acids). Sequences have a **natural order** (position 1, 2, 3, …) but no "second dimension" like an image, so we treat them as **1D signals** and use **1D convolutions** and **recurrent** models. This chapter covers: (1) **encoding** DNA and protein sequences into numbers; (2) **1D convolutional neural networks (CNNs)** for detecting local patterns (e.g., motifs); (3) **recurrent neural networks (RNNs)** and **long short-term memory (LSTM)** networks at a conceptual level; (4) **sequence classification** tasks (splice sites, binding, etc.); and (5) **basic model interpretation** so we can see what the model "looked at."

You do not need a deep background in biology: we only assume that DNA has four letters (A, C, G, T), proteins have about 20 amino-acid letters, and that **motifs** are short, conserved patterns that often have a function (e.g., binding sites).

---

## 8.1 Encoding DNA and Protein Sequences

Raw sequences are **strings of symbols**. Machine learning models need **numeric inputs**. We encode each symbol (or short substring) as a vector or index.

### 8.1.1 One-hot encoding (DNA and protein)

**DNA** has four nucleotides: **A**denine (**A**), **C**ytosine (**C**), **G**uanine (**G**), and **T**hymine (**T**). We assign each letter a **one-hot vector** of length 4:

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

**Proteins** use a larger alphabet: **20 standard amino acids**. Each one has a **name**, a **three-letter code**, and a **single-letter code** that scientists use when writing protein sequences.

Here is the full set:

| Amino Acid    | 3-Letter | 1-Letter |
| ------------- | -------- | -------- |
| Alanine       | Ala      | **A**    |
| Arginine      | Arg      | **R**    |
| Asparagine    | Asn      | **N**    |
| Aspartic acid | Asp      | **D**    |
| Cysteine      | Cys      | **C**    |
| Glutamine     | Gln      | **Q**    |
| Glutamic acid | Glu      | **E**    |
| Glycine       | Gly      | **G**    |
| Histidine     | His      | **H**    |
| Isoleucine    | Ile      | **I**    |
| Leucine       | Leu      | **L**    |
| Lysine        | Lys      | **K**    |
| Methionine    | Met      | **M**    |
| Phenylalanine | Phe      | **F**    |
| Proline       | Pro      | **P**    |
| Serine        | Ser      | **S**    |
| Threonine     | Thr      | **T**    |
| Tryptophan    | Trp      | **W**    |
| Tyrosine      | Tyr      | **Y**    |
| Valine        | Val      | **V**    |

A useful pattern: the one-letter codes are usually the **first letter of the name**, but when multiple amino acids start with the same letter, conventions differ, e.g. **K** for Lysine (because **L** is Leucine), **F** for Phenylalanine, and **W** for Tryptophan (whose structure suggests a double ring). So a protein sequence like `MKTFFVLLL` is shorthand for Methionine → Lysine → Threonine → Phenylalanine → Phenylalanine → Valine → Leucine → Leucine → Leucine. That string is a **biological sentence**: the cell's ribosome reads it like a molecular printer, assembling amino acids into a chain that then folds into a working nanomachine (enzyme, receptor, or structural fiber). In modern computational biology, this 20-letter alphabet is often treated like **natural language**, which is why protein models borrow ideas from the same transformer architectures used in language models.

For **one-hot encoding**, the same idea as DNA applies: one letter → one vector of length 20 with a single 1. So one protein sequence of length $L$ becomes an $L \times 20$ matrix.

**Why one-hot?** It is simple, interpretable, and does not impose an order on the letters (unlike assigning A=1, C=2, …). The model can learn which positions and which letters matter.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter8/onehot_dna.png" alt="One-hot encoding of a short DNA sequence" />
  <p class="caption"><strong>Figure 8.1.</strong> One-hot encoding of a short DNA sequence. Each nucleotide (A, C, G, T) is replaced by a row of four numbers with a single 1. The full sequence becomes a matrix of shape (sequence length) × 4.</p>
</div>

### 8.1.2 Index encoding and learned embeddings

Neural networks only process numbers, not letters. So we first turn each symbol into an **integer index**. For DNA we might assign A→0, C→1, G→2, T→3. The sequence `ACGTAG` then becomes the numeric vector $(0, 1, 2, 3, 0, 2)$ of length $L$. This is just **indexing**; no learning yet.

If we fed these integers straight into the network, the model could wrongly treat them as ordered (e.g. 3 > 2 > 1 > 0), but biologically that order is meaningless: the numbers are only labels. So we use an **embedding layer**: a lookup table that maps each index to a **learned vector** of dimension $d$. For example, with $d=4$ and DNA, the layer might learn:

- A (index 0) → $(0.21, -0.5, 0.8, 0.1)$
- C (index 1) → $(-0.7, 0.2, 0.3, 0.6)$
- G (index 2) → $(0.4, 0.9, -0.2, -0.1)$
- T (index 3) → $(-0.3, 0.1, 0.7, 0.5)$

These vectors are **learned during training**: the network adjusts them to improve prediction. The full sequence then becomes a matrix of size $L \times d$, one $d$-dimensional vector per position. So we still get a matrix we can feed to 1D conv or RNN layers. One-hot is a special case where the first layer is fixed (no learning); embedding lets the model learn a representation that suits the task.

### 8.1.3 K-mer encoding (optional)

A **k-mer** is a **contiguous substring of length $k$**. You get k-mers by sliding a window of size $k$ along the sequence. For example, if the DNA sequence is `ACGTGCA` and $k = 3$, the 3-mers (also called **triplets**) are:

- ACG, CGT, GTG, TGC, GCA

Each is just a short chunk of the sequence.

**Why $4^k$ possible k-mers?** DNA has four letters (A, C, G, T). In a substring of length $k$, each of the $k$ positions can be any of the four. So the number of possible k-mers is $4 \times 4 \times \cdots \times 4 = 4^k$. Examples: $k=1$ gives 4 possibilities (A, C, G, T); $k=2$ gives $4^2 = 16$ (AA, AC, AG, AT, …); $k=3$ gives $4^3 = 64$ (AAA, AAC, …, TTT).

**How we use k-mers as features.** Two common strategies:

**1. Bag-of-k-mers (counting).** As in bag-of-words in NLP, we build a vector of **counts**: how many times each possible k-mer appears in the sequence. For $k=2$ there are 16 possible dimers, so the sequence becomes a **16-dimensional vector** (e.g. AA: 3, AC: 1, AG: 0, AT: 5, …). This **ignores position** but captures **sequence composition** and is widely used for tasks like genome classification.

**2. Sliding k-mer encoding (position-aware).** Here we keep the **order** of k-mers. At each position we have one k-mer, i.e. one of $4^k$ possible "tokens." We then represent each token by **one-hot** (a vector of length $4^k$) or by a **learned embedding**. So the sequence becomes a list of token vectors, one per position. This is the idea behind many modern genomic models that treat DNA like text: DNA letters play the role of characters, k-mers the role of words, and the genome the role of a sentence. Researchers build **DNA language models** using transformer-style architectures similar to those in natural language. For the rest of this chapter we will mostly use **one-hot or index+embedding at the single-letter level** so that 1D convolutions can learn motifs directly from the raw encoding; k-mers remain a useful alternative when you want to exploit this "sequence as language" view.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter8/2-mer.jpg" alt="One-hot encoded table for 2-mers" />
  <p class="caption"><strong>Figure 8.2.</strong> One-hot encoding for 2-mers (dimers). Each 2-mer (e.g., AA, AC, …, TT) is represented by a row of length $4^2 = 16$ with a single 1 in the column for that dimer.</p>
</div>

---

## 8.2 1D convolutional neural networks (CNNs) for motif detection

In Chapter 7 we defined **1D convolution**: a **kernel** (a short vector) slides over a **1D signal**, and at each position we compute the inner product of the kernel with the overlapping segment. For sequences, the "signal" is the **encoded sequence** (e.g., the $L \times 4$ matrix for DNA). We apply convolution **along the sequence dimension** (position 1, 2, …, L).

### 8.2.1 One channel: sequence as a 1D signal per channel

Think of the **one-hot matrix** as having **4 channels** (one per nucleotide). At each position $i$ we have a 4-dimensional vector. A **1D convolutional kernel** that slides along the sequence can have width $k$ (e.g., $k=5$) and **depth 4** (one weight per channel at each of the $k$ positions). So the kernel is like a $k \times 4$ patch. At each position $i$, we take the $k \times 4$ block of the sequence matrix, multiply it element-wise with the kernel, sum everything, and get **one number**. That number is the **activation** of this filter at position $i$. So **one kernel** gives **one 1D feature map** of length (about) $L - k + 1$.

**Interpretation:** The kernel is learning a **local pattern** of length $k$, a **motif**. High activation at position $i$ means "this motif is present around position $i$." So 1D CNNs naturally do **motif detection**. (We use "CNN" for convolutional neural network throughout.)

### 8.2.2 Example: a tiny sequence and one kernel

The kernel does not "learn letters." It learns **numbers that react strongly when certain letters appear in certain positions**. Walking through one convolution step makes this clear.

**Step 1: Encode the sequence.** Recall one-hot: A → $(1,0,0,0)$, C → $(0,1,0,0)$, G → $(0,0,1,0)$, T → $(0,0,0,1)$. So the sequence

A  C  G  T  A  G  C  T

becomes an **8 × 4 matrix** (one row per position). Row 1 is $(1,0,0,0)$, row 2 is $(0,1,0,0)$, and so on.

**Step 2: Define the kernel.** Take a kernel with **width 3** and **depth 4**: a **3 × 4 matrix** of learnable weights (12 numbers in total). At first these are random; training will adjust them. For illustration, suppose the kernel rows are:

- Position 1: $(0.8, -0.2, -0.1, 1.2)$  
- Position 2: $(-0.3, 0.5, 1.7, -0.4)$  
- Position 3: $(1.1, -0.6, -0.3, 0.9)$  

**Step 3: First window (positions 1 to 3).** The window is A, C, G. The encoded 3×4 block is:

- Row 1 (A): $(1, 0, 0, 0)$  
- Row 2 (C): $(0, 1, 0, 0)$  
- Row 3 (G): $(0, 0, 1, 0)$  

We take the **element-wise product** of this block with the kernel and **sum all 12 entries**. Because the input is one-hot, only **one value per row contributes**:

- Row 1: $(1,0,0,0) \cdot (0.8,-0.2,-0.1,1.2) = 0.8$  
- Row 2: $(0,1,0,0) \cdot (-0.3,0.5,1.7,-0.4) = 0.5$  
- Row 3: $(0,0,1,0) \cdot (1.1,-0.6,-0.3,0.9) = -0.3$  

Sum: $0.8 + 0.5 - 0.3 = 1.0$. That number is the **feature map value** at this position. The kernel then slides to the next window (positions 2 to 4), and we repeat; we get 6 values in total (output length $8 - 3 + 1 = 6$).

**Step 4: What training does.** Gradient descent updates those 12 kernel weights so that the output is **large when useful patterns appear** and small otherwise. Suppose the task is to detect the motif **T G A**. Training might push the kernel toward something like:

- Position 1: large weight on the **T** channel (fourth column)  
- Position 2: large weight on the **G** channel (third column)  
- Position 3: large weight on the **A** channel (first column)  

When the sequence has T at position 1, G at 2, and A at 3, the dot product is large. When other bases appear, the other (e.g. negative) weights reduce the score. So the kernel has learned a **motif detector**. The kernel never stores letters; it stores **weights aligned with the one-hot channels** (channel 1 = A, 2 = C, 3 = G, 4 = T). A large weight on the G channel in the middle row means "activate strongly when the middle base is G." Biologists will recognize this: such kernels often resemble **position weight matrices (PWMs)**, the same objects used to describe transcription-factor binding motifs. So "G in the middle, A and T on the sides" is exactly the kind of pattern one kernel might learn.

**Zooming out:** A typical genomic CNN might use **64 kernels** of **width 10**. Each kernel can learn a different motif (e.g. promoter-like, splice-site-like, GC-rich, repeat-like). The feature maps then tell the network **where** those motifs appear. Convolution on DNA is effectively a **motif discovery machine**: evolution wrote the patterns; gradient descent learns to notice them.

Summary of dimensions:

- **Input:** sequence of length $L$, encoded as $L \times 4$ (DNA) or $L \times 20$ (protein).  
- **1D conv:** several kernels (e.g., 32 or 64), each of width $k$ (e.g., 5 to 15).  
- **Output:** one feature map per kernel; each map has length $\approx L - k + 1$ (depending on padding).  
- Then we often add **rectified linear unit (ReLU)** and **pooling** (e.g., max-pool) to reduce length and combine information.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter8/1dcnn_motif.png" alt="1D CNN sliding over a sequence for motif detection" />
  <p class="caption"><strong>Figure 8.3.</strong> A 1D convolutional kernel (motif detector) slides over the encoded sequence. Each position yields one activation; high activation indicates presence of the learned pattern (motif) at that position.</p>
</div>

### 8.2.3 From motif detection to classification

For **sequence classification** (e.g., "is this region a splice site?" or "does this protein bind?"), we typically:

1. Encode the sequence (e.g., one-hot).  
2. Stack one or more **1D convolution (conv) + ReLU (+ optional pooling)** layers to get a set of feature maps.  
3. **Pool** over the whole sequence (e.g., global max-pool or average-pool) to get one vector per channel.  
4. Pass that vector through **fully connected** layers to get the class (or score).

So the CNN **extracts local motifs**, and the global pool **summarizes** "which motifs appeared anywhere in the sequence" into a fixed-size vector for the classifier.

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

**Intuition:** The RNN "walks" along the sequence and updates its internal memory ($\mathbf{h}$) at each step. That memory carries information from the past into the future.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter8/rnn_unfold.png" alt="RNN unfolded in time" />
  <p class="caption"><strong>Figure 8.4.</strong> An RNN unfolded in time. At each step $t$, the model takes input $\mathbf{x}_t$ and the previous hidden state $\mathbf{h}_{t-1}$, and outputs a new hidden state $\mathbf{h}_t$. The same weights are reused at every step.</p>
</div>

### 8.3.2 Vanishing gradient and long-range dependence

Training RNNs with **simple** recurrence (e.g., $\mathbf{h}\_t = \tanh(\mathbf{W}\_{xh} \mathbf{x}\_t + \mathbf{W}\_{hh} \mathbf{h}\_{t-1} + \mathbf{b})$) runs into the **vanishing gradient** problem: when the sequence is long, gradients from the loss at the end of the sequence shrink to almost zero when they are backpropagated to the beginning. So the model has trouble learning **long-range** dependencies (e.g., "position 5 matters for the label at position 100").

### 8.3.3 LSTM: a conceptual overview

**Long Short-Term Memory (LSTM)** networks fix this by adding a **cell state** $\mathbf{c}_t$ (in addition to the hidden state $\mathbf{h}_t$) and **gates** that control how much information is **forgotten**, **updated**, and **output** at each step. You do not need the full equations here; the main ideas are:

- **Forget gate:** "How much of the old cell state should we keep?"  
- **Input gate:** "How much of the new candidate update should we add to the cell?"  
- **Output gate:** "How much of the cell state should we expose as the hidden state?"

The **cell state** $\mathbf{c}_t$ is like a "conveyor belt" that can carry information across many time steps with less decay than in a simple RNN. So LSTMs (and later, **gated** variants like the **gated recurrent unit (GRU)**) are the standard choice when we need **long-range** context in sequences.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter8/lstm_cell.png" alt="LSTM cell: gates and cell state" />
  <p class="caption"><strong>Figure 8.5.</strong> LSTM cell (conceptual). The cell state $\mathbf{c}$ is updated by forget and input gates; the hidden state $\mathbf{h}$ is filtered by the output gate. This allows the network to retain information over many steps.</p>
</div>

### 8.3.4 When to use CNN vs RNN for sequences

- **1D CNN:** Good for **local patterns** (motifs, binding sites, short motifs). Fast, parallel over the sequence; receptive field is limited by kernel size (unless you stack many layers).  
- **RNN/LSTM:** Good when **order** and **long-range** context matter (e.g., language, long-range dependencies in a gene). Sequential computation; can in principle use the full history.

In practice, **hybrid** models (e.g., 1D CNN to extract motifs, then LSTM on top) are also common in genomics and proteomics.

---

## 8.4 Sequence Classification Tasks

Here we give a few **concrete tasks** so you can see how encoding + 1D CNN (or RNN) fit together.

### 8.4.0 Exons and introns: why genes are not continuous

Eukaryotic genes often look like manuscripts with long digressions between the useful paragraphs: they are not **clean uninterrupted instructions**. The structure is **exon, intron, exon, intron, exon**. **Exons** are the parts that end up in the final mRNA and encode protein; **introns** are the stretches in between. (So introns are *between* exons, not “inside” them.) During transcription the whole region is copied into RNA, and then the **spliceosome** removes the introns and joins the exons. Why keep introns at all?

- **Alternative splicing (the big one).** Introns give the spliceosome places where it can **choose different boundaries**. One gene can produce **multiple mRNAs**, e.g., Exon1–Exon2–Exon4 vs Exon1–Exon3–Exon4, and thus **multiple proteins**. Humans have about **20,000 genes** but produce many more proteins largely because of alternative splicing. Introns act like **modular connectors** in a biological Lego system.
- **Evolutionary playground.** Introns tolerate mutations more easily than exons (which encode protein). So introns become a **sandbox**: mutations accumulate, regulatory signals appear, and new splice patterns can evolve without immediately breaking the protein.
- **Regulatory information.** Introns are not empty; many contain signals that control **when** and **how strongly** a gene is expressed, and **where** splicing occurs.
- **Exon shuffling.** Because introns separate coding pieces, recombination can move **exons** (functional domains) between genes during evolution, making it easier for evolution to **rearrange modules** (e.g., signal + binding + catalytic domain).
- **Historical accident.** Bacteria mostly lack introns; complex organisms have many. Introns likely accumulated in early eukaryotes, and once the splicing machinery existed, they became part of the system.

From a **computational** perspective, introns create **long-range dependencies**: a splice signal can depend on sequence motifs tens or hundreds of bases away, or inside the intron itself. That is why genomics models often need architectures (e.g., RNNs or Transformers) that capture **context across long distances**. The genome is not a short sentence; it is more like a novel with chapters and annotations scattered between the lines.

### 8.4.1 Splice site prediction

Genes are not continuous stretches of coding sequence; they are interrupted by **introns**, which the cell removes from the RNA transcript using the **spliceosome**. The spliceosome must recognize **exact cutting points** in the RNA: these are the **splice sites**. For every intron there are two main sites: the **donor site** (the **beginning** of the intron, where cutting starts), usually marked by **GT** in DNA (GU in RNA), and the **acceptor site** (the **end** of the intron, where the next exon is joined), usually marked by **AG**. So the canonical pattern is **GT … intron … AG**. Not every GT or AG in the genome is a real splice site; the genome is huge and those dinucleotides appear everywhere. The spliceosome uses **context** (nearby motifs, regulatory signals, RNA structure, etc.) to decide which sites are real. That is exactly where machine learning helps: given a **window of DNA** around a candidate GT or AG, a model learns to discriminate “real” splice sites from false ones using sequence context, with labels from experiments such as **RNA sequencing (RNA-seq)** that show where splicing actually occurs.

**Goal:** Given a short window of DNA (e.g., 100 to 400 nucleotides (nt)) around a candidate **donor** or **acceptor** site, predict whether it is a true splice site.

- **Input:** Sequence window (e.g., A, C, G, T string).  
- **Encoding:** One-hot (length × 4).  
- **Model:** 1D CNN (several conv layers + global pool) or CNN + LSTM; output: binary (splice site vs not) or two classes (donor vs acceptor).  
- **Label:** From annotations (known splice sites) or from RNA sequencing (RNA-seq).

### 8.4.2 Transcription factor binding (or motif presence)

A **transcription factor (TF)** is a protein that binds to specific DNA sequences and controls whether a gene is turned **on or off**, like a molecular switch deciding whether transcription should start. TFs do **not** bind randomly: they prefer certain short DNA patterns called **motifs**, often around **6–12 nucleotides** long, with some positions flexible rather than perfectly fixed. The machine-learning problem is: *Given a piece of DNA, does this transcription factor bind here?* We use a short **DNA window** (e.g., 100–500 nt) because binding depends on a **local neighborhood** around the motif. After encoding (e.g., one-hot or k-mers), a **1D CNN** scans the sequence; each convolution kernel acts as a **pattern detector** that responds strongly when it sees a motif it has learned. The network outputs **binding** or **non-binding**. Labels come from experiments such as **ChIP-seq** (chromatin immunoprecipitation followed by sequencing): crosslink proteins to DNA, use an antibody to pull down one specific TF, and sequence the bound DNA to get a genome-wide map of where that TF binds. Sequences from those locations are **positive** examples; random or unbound sequences are **negative** examples. Remarkably, after training, the **learned conv filters** often resemble **position weight matrices (PWMs)**, the classic motif description in bioinformatics, and when visualized as sequence logos they look like the motifs biologists have known for decades. So the network effectively **rediscovers** known DNA-binding motifs from sequence data; the convolution filters become **motif detectors**, and the task is a bridge between pattern recognition and biology.

**Goal:** Predict whether a short DNA sequence (e.g., 100 to 500 nt) contains a binding site for a given **transcription factor (TF)**.

- **Input:** Sequence.  
- **Encoding:** One-hot or k-mers.  
- **Model:** 1D CNN or a dedicated motif model; output: binding vs non-binding.  
- **Label:** From chromatin immunoprecipitation followed by sequencing (ChIP-seq) or similar experiments.

### 8.4.3 Protein secondary structure or function

Proteins are chains of amino acids that **fold** into 3D shapes; that shape largely determines what the protein does. **Secondary structure** is the local folding pattern along the chain, typically **α-helix**, **β-sheet**, or **coil** (loop), and is determined by the sequence and local interactions. Knowing secondary structure helps interpret how a protein works and is a step toward full 3D structure. **Protein function** (e.g., enzyme vs non-enzyme, or finer labels such as kinase or protease) is also encoded in the sequence: key motifs and the overall fold dictate binding and catalysis. Experimentally, structure is determined by X-ray crystallography or cryo-EM, and function by biochemical assays or homology; both are slow and costly. Genomes produce huge numbers of protein sequences, but only a fraction have known structure or well-annotated function. The machine-learning task is to predict **secondary structure at each position** (helix/sheet/coil from sequence) or a **single functional label** (e.g., enzyme vs non-enzyme) from the amino acid sequence alone. Labels come from structure databases (e.g., **DSSP**, which assigns secondary structure from solved 3D structures) or from functional databases (e.g., Gene Ontology, enzyme commission numbers). Models that use the **order** of the sequence (e.g., RNNs or Transformers) can capture context along the chain, which matters because the fold at one position often depends on residues elsewhere.

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

The **weights** of a 1D convolutional kernel (over one-hot encoded DNA) can be reshaped into a $k \times 4$ matrix. Each row has four values (one per nucleotide). If we **normalize** each row to sum to 1, we get something like a **position weight matrix (PWM)**. Plotting it (e.g., as a **sequence logo**) shows which nucleotides the filter "prefers" at each position, i.e., the **learned motif**.

So the first step of interpretation is: **inspect the first-layer convolution (conv) filters** and turn them into logos or position weight matrices (PWMs).

### 8.5.2 Saliency and gradient-based attribution

**Saliency:** For a given input sequence and the model's output (e.g., score for "splice site"), compute the **gradient** of that output with respect to the **input** (the one-hot or embedding matrix). The magnitude of the gradient at each position tells us how much a small change there would change the output, so "important" positions get large gradient magnitude. We can plot **saliency per position** along the sequence.

**Limitation:** Gradients can be noisy. Smoothed or integrated variants (e.g., **integrated gradients**) are often used in practice.

### 8.5.3 Attention (if the model uses it)

If the model has an **attention** mechanism (e.g., in a Transformer or an attention layer on top of an RNN), the **attention weights** tell us which positions the model "attended to" when making the prediction. We can plot attention over positions for a given sequence. This is a direct way to see **where** the model looked.

### 8.5.4 In practice

- **1D CNN:** Visualize first-layer (or early-layer) kernels as motifs; use saliency or gradient-based maps to highlight important positions.  
- **RNN/LSTM:** Use hidden-state analysis or attention (if available); otherwise gradient-based attribution.  
- **Hybrid:** Combine filter visualization (CNN part) and attention or gradients (recurrent part).

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter8/interpretation_logo.png" alt="Learned conv filter as sequence logo" />
  <p class="caption"><strong>Figure 8.6.</strong> A learned 1D convolutional filter (first layer) converted into a sequence logo. Height at each position indicates how much the filter "prefers" each nucleotide; this is the model's learned motif.</p>
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

- [Deep learning for biology (Nature, 2018)](https://www.nature.com/articles/d41586-018-02174-z).
- [BPNet (Avsec et al.)](https://www.nature.com/articles/s41592-021-01282-5), interpretable CNNs for TF binding with attribution.
- [Understanding LSTM networks (colah's blog)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/), conceptual explanation of LSTMs with diagrams.
- [PyTorch: Sequence models and LSTM](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html), RNN/LSTM in code.
- [Kipoi: model zoo for genomics](https://kipoi.org/), pretrained models for splice sites, TF binding, etc., with interpretation tools.
