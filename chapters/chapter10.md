# Chapter 10: Transformers and Attention in Biology

## Introduction

In Chapters 7, 8, and 9 we used models that process inputs either as:

- grids (CNNs for images),
- strings with local receptive fields (1D CNNs for motifs),
- or sequence order with recurrence (RNNs and LSTMs, conceptually),
- and in Chapter 9 we introduced machine learning for single-cell gene expression data.

For many biology problems, a key need is to represent **relationships between distant positions**. Examples include long-range dependencies in:

- DNA regulatory signals (motifs that influence other motifs far away),
- RNA structure and base pairing (interactions across the transcript),
- protein residue interactions (non-local contacts),
- and biomedical text (entities that refer to earlier mentions).

Transformers solve this by combining **attention** with efficient parallel computation.

This chapter covers:

- Motivation and architecture of Transformers,
- self-attention and attention heads,
- sequence modeling with Transformers,
- applications in GPT, LLMs, genomics, proteomics, and biomedical NLP,
- fine-tuning pretrained models,
- and visualization of attention maps.

---

## 10.1 Motivation: Why Self-Attention?

### 10.1.1 The limitation of local context

Convolutions look at a fixed-size window. If a motif influences a position more than that window, you need to stack many layers to expand the receptive field.

RNNs, in principle, can carry information across the whole sequence. In practice, training long dependencies can be difficult, and computation is inherently sequential.

### 10.1.2 The core idea of attention

Self-attention lets each position in a sequence directly interact with every other position.

For a sequence of length $N$, the model computes:

- which tokens are relevant to a given token,
- and a weighted mixture of information from those relevant tokens.

This gives **dynamic, content-based** long-range interactions, without requiring recurrence.

---

## 10.2 Transformer Architecture (Encoder and Decoder)

A Transformer is built from repeated blocks. Most modern language models are either:

- **Encoder-only** (example: BERT-style models),
- **Decoder-only** (example: GPT-style models),
- or **Encoder-decoder** (original Transformer, used for translation).

### 10.2.1 High-level architecture

The standard components in each layer are:

- **Multi-head self-attention**,
- **Position-wise feed-forward network** (FFN),
- **Residual connections** (skip connections),
- **Layer normalization**.

<div class="figure">
  <img src="https://upload.wikimedia.org/wikipedia/commons/3/34/Transformer%2C_full_architecture.png" alt="Transformer architecture showing encoder and attention" />
  <p class="caption"><strong>Figure 10.1.</strong> Full Transformer architecture, encoder and attention mechanism. Source: Wikimedia Commons file “Transformer, full architecture.png”.</p>
</div>

### 10.2.2 Residual connections and layer normalization

Residual connections help gradients flow through deep networks. The combination:

- attention output + residual,
- FFN output + residual,
- with layer normalization around these steps,

stabilizes optimization.

---

## 10.3 Self-Attention and Attention Heads

Self-attention is the key computation inside Transformers.

### 10.3.1 Query, key, value (Q, K, V)

Let the input embeddings be:

$$
X \in \mathbb{R}^{N \times d_{model}}
$$

For a single attention head:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

where $W_Q, W_K, W_V$ are learned weight matrices.

### 10.3.2 Scaled dot-product attention

The attention weights measure how well token $i$ (query) matches token $j$ (key).

The attention output is:

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where:

- $QK^T$ gives an $N \times N$ similarity matrix,
- $\text{softmax}$ turns similarities into weights that sum to 1 per query token,
- the multiplication by $V$ produces a weighted sum of value vectors.

The scaling by $\sqrt{d_k}$ prevents the dot products from becoming too large when the dimensionality increases.

<div class="figure">
  <img src="https://upload.wikimedia.org/wikipedia/commons/8/81/Attention-qkv.png" alt="Attention calculation flow through Q, K, V" />
  <p class="caption"><strong>Figure 10.2.</strong> Computation flow through a single attention head, based on the standard Q, K, V attention computation. Source: Wikimedia Commons file “Attention-qkv.png”.</p>
</div>

### 10.3.3 Intuition for “what attention is doing”

For each position $i$:

- treat token $i$ as a **query** asking, “which tokens matter for me?”,
- treat all tokens as **keys** to compare against,
- use the resulting weights to mix **values** into a new representation at position $i$.

So attention is a learned, data-dependent routing mechanism.

---

## 10.4 Multi-Head Attention

One head can learn only a limited type of interaction. Multi-head attention uses multiple heads in parallel.

<div class="figure">
  <img src="https://upload.wikimedia.org/wikipedia/commons/6/68/Attention_Is_All_You_Need_-_Multiheaded_Attention.png" alt="Multi-headed attention diagram" />
  <p class="caption"><strong>Figure 10.3.</strong> Multi-headed attention, multiple attention heads computed in parallel and then concatenated. Source: Wikimedia Commons file “Attention Is All You Need - Multiheaded Attention.png”.</p>
</div>

### 10.4.1 Mathematical definition

For head $h$:

$$
\text{head}_h = \text{Attention}(XW_Q^{(h)}, XW_K^{(h)}, XW_V^{(h)})
$$

Then:

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1,\ldots,\text{head}_H)W_O
$$

Each head can specialize, for example:

- one head learns long-range motif interactions in DNA,
- another head tracks local patterns,
- another head focuses on syntax-like structure in text.

---

## 10.5 Positional Information

Transformers process tokens in parallel, so they need an explicit way to represent order.

Two common approaches:

1. **Sinusoidal positional encodings**, deterministic and generalizable to longer lengths,
2. **Learned positional embeddings**, trained parameters for fixed maximum length.

### 10.5.1 Sinusoidal positional encoding (classic formula)

For position `pos` and embedding dimension index `i`:

$$
\text{PE}(\text{pos},2i) = \sin\left(\frac{\text{pos}}{10000^{2i/d_{model}}}\right)
$$

$$
\text{PE}(\text{pos},2i+1) = \cos\left(\frac{\text{pos}}{10000^{2i/d_{model}}}\right)
$$

These vectors are added to token embeddings before attention layers.

---

## 10.6 Sequence Modeling with Transformers

### 10.6.1 Autoregressive modeling (GPT-style)

In GPT-style language modeling, the model predicts the next token:

$$
p(x_{t+1}\mid x_{1:t})
$$

Training typically uses cross-entropy loss over the next-token targets.

### 10.6.2 Causal masking

To prevent information leakage, decoder self-attention uses a **causal mask**. Each token at position $t$ may attend only to positions $\le t$.

<div class="figure">
  <img src="https://upload.wikimedia.org/wikipedia/commons/7/72/Decoder_self-attention_with_causal_masking%2C_detailed_diagram.png" alt="Decoder self-attention with causal masking diagram" />
  <p class="caption"><strong>Figure 10.4.</strong> Decoder self-attention with causal masking, future tokens are masked out. Source: Wikimedia Commons file “Decoder self-attention with causal masking, detailed diagram.png”.</p>
</div>

### 10.6.3 What changes across tasks

The same transformer block supports different tasks by changing the head and the masking:

- **Text generation**: decoder-only, causal mask,
- **Masked token prediction**: encoder-only, bidirectional attention with masking strategy,
- **Sequence classification**: pooling over final hidden states (or using a special token) then a classifier head,
- **Token labeling**: output per token from the final hidden states.

---

## 10.7 Fine-Tuning Pretrained Models

Pretrained transformers learn general representations. Fine-tuning adapts them to a specific dataset and task.

### 10.7.1 Three common strategies

1. **Add a task head and fine-tune all layers**
   - simple and often strong for moderate datasets.
2. **Freeze the base model and train only the head**
   - good when labeled data is small.
3. **Parameter-efficient fine-tuning (PEFT)**
   - update only a small set of parameters, for example:
     - adapters,
     - LoRA (Low-Rank Adaptation).

### 10.7.2 Simple fine-tuning skeleton (conceptual)

Many pipelines use the `transformers` library. A typical workflow is:

1. tokenize sequences,
2. create a dataset with input tokens and labels,
3. run training with an optimizer,
4. evaluate on validation and test sets.

Pseudo-code:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_CLASSES)

args = TrainingArguments(
    output_dir="out",
    num_train_epochs=3,
    per_device_train_batch_size=16,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
```

The important conceptual point is not the exact library, but the idea:

- pretrained weights provide a strong starting representation,
- fine-tuning aligns the representation with your task-specific objective.

---

## 10.8 Applications in GPT, LLMs, Genomics, Proteomics, and Biomedical NLP

### 10.8.1 GPT and LLMs

Large language models (LLMs) are transformers trained on very large text corpora. GPT-style models are typically:

- decoder-only,
- trained with next-token prediction,
- used for generation, completion, and instruction-following after additional training.

### 10.8.2 Genomics

Genomics uses a similar idea, but the “tokens” are biological symbols:

- DNA can be tokenized as nucleotides (A, C, G, T) or as k-mers,
- RNA and DNA-binding motifs can be modeled as sequence-like data,
- attention can represent relationships between distant motifs and regulatory elements.

Many genomics transformers also support tasks like:

- predicting regulatory activity,
- predicting protein binding from sequence,
- variant effect prediction.

### 10.8.3 Proteomics

Proteins are sequences of amino acids. Protein language models use transformers to learn:

- residue-level contexts,
- sequence-to-function mappings,
- embeddings used for structure and function prediction.

Attention heads can capture patterns that relate residues far apart in sequence but close in the 3D structure.

### 10.8.4 Biomedical NLP

Biomedical NLP applies transformers to tasks over scientific text and clinical notes:

- named entity recognition (genes, diseases, drugs),
- relation extraction,
- question answering,
- clinical summarization.

Transfer learning is powerful here because biomedical text is often expensive to label.

---

## 10.9 Visualization of Attention Maps

Attention maps show the attention weights between query tokens and key tokens.

### 10.9.1 How to visualize

For a single head at a chosen layer, you can extract an attention matrix:

- rows correspond to query positions,
- columns correspond to key positions,
- values correspond to attention weights.

Plotting this as a heatmap gives a human-interpretable view of which tokens influence others.

### 10.9.2 Example workflow (conceptual)

1. pick an input sequence,
2. run the model forward pass,
3. select a layer and head,
4. extract attention weights,
5. plot attention as a heatmap.

### 10.9.3 Important caveat

Attention weights are not always a guaranteed explanation. Models can use attention for computation in ways that do not directly correspond to causal feature attribution.

In practice, you can combine attention visualization with:

- perturbation tests (mask or remove tokens and measure prediction change),
- gradient-based attribution methods,
- sanity checks across layers and heads.

---

## Summary

- Transformers replace recurrence with self-attention, enabling long-range interactions and parallel computation.
- Self-attention computes weighted mixtures of value vectors, where weights come from similarities between queries and keys.
- Multi-head attention runs several attention patterns in parallel and concatenates the results.
- Sequence modeling uses causal masking for autoregressive generation.
- Pretrained transformers can be fine-tuned for biology and NLP tasks using task-specific heads and objectives.
- Attention visualization can provide intuition, but interpretability requires careful validation.

---

## Topics for the Next Class

If your course continues beyond Transformers, typical next steps are:

- applying these ideas to a concrete biological prediction task end-to-end,
- comparing fine-tuning vs prompt tuning and evaluation protocols,
- model robustness checks and reproducibility practices.

---

## Further Reading

- [Transformer (deep learning architecture), Wikipedia](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture))
- [Attention (machine learning), Wikipedia](https://en.wikipedia.org/wiki/Attention_(machine_learning))
- [GPT, Wikipedia](https://en.wikipedia.org/wiki/GPT)
- [Large language model, Wikipedia](https://en.wikipedia.org/wiki/Large_language_model)

