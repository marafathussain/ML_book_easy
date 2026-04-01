# Chapter 10: Generative Deep Learning and Attention in Biology

## Introduction

In Chapters 7, 8, and 9 we focused mostly on **discriminative** models: predicting labels, classes, or targets from inputs. Many biological and biomedical problems also need **generation**: new samples that resemble real data, or coherent continuations of sequences, under constraints that you specify.

This chapter gives a **road map through modern generative deep learning**, then connects it smoothly to **attention-based models** (Transformers), which today power both large-scale **text generation** and rich **representation learning** for sequences in genomics and proteomics.

We will cover:

- what **generative models** try to learn, and how families of models differ,
- a short **historical evolution** from classical ideas to deep architectures,
- **variational autoencoders (VAEs)** and **generative adversarial networks (GANs)** with standard diagrams,
- **diffusion** and denoising as a representative modern approach for images (and related domains),
- **autoregressive** sequence modeling as a bridge to **Transformers**,
- the **Transformer** stack: self-attention, multi-head attention, masking,
- **fine-tuning**, applications in **LLMs, genomics, proteomics, and biomedical NLP**, and **attention visualization**.

Public-domain-style figures below come from **Wikimedia Commons**; each caption names the file and author and links to the license.

---

## 10.1 What Is a Generative Model?

A **generative model** describes how data are produced, at least up to sampling or scoring.

- You may want to **draw new examples** $x$ from a distribution learned from data (images, sequences, vectors of gene expression).
- You may want **conditional** generation: samples $x$ given a context $c$ (class label, cell type, protein family, text prompt).
- In some setups you care about an explicit **density** $p(x)$ or $p(x \mid c)$; in others you only need a **sampler** and never evaluate likelihood.

**Discriminative** models focus on $p(y \mid x)$. **Generative** models target $p(x)$, or $p(x, y)$, or a sampling procedure implied by training. The same neural building blocks (convolutions, recurrence, attention) appear in both worlds; what changes is the **training objective** and whether the model admits a tractable likelihood.

---

## 10.2 Families and Evolution of Deep Generative Models

Early deep generative work explored **Boltzmann machines** and related energy-based ideas. The 2010s brought several parallel families that are still in use:

1. **Autoregressive models** factorize $p(x)$ as a product of conditionals, $p(x_1)\,p(x_2\mid x_1)\cdots$. Natural for sequences and grids (PixelCNN, language models).
2. **Variational autoencoders (VAEs)** optimize a **lower bound** on $\log p(x)$ using an approximate posterior over latents.
3. **Generative adversarial networks (GANs)** pit a **generator** against a **discriminator** in a minimax game; sampling is explicit, likelihood is usually not.
4. **Normalizing flows** use invertible transforms to obtain **exact** densities when the base density is simple.
5. **Diffusion and score-based models** learn to reverse a noise corruption process, or learn **scores** $\nabla_x \log p(x)$; they now dominate high-quality image generation.

The diagram below (after a standard taxonomy) situates these approaches. It is schematic: in practice hybrids exist (diffusion in latent space, autoregressive decoders, and so on).

<div class="figure">
  <img src="https://upload.wikimedia.org/wikipedia/commons/b/bb/Types_of_deep_generative_models.png" alt="Taxonomy of deep generative models: directed, undirected, implicit, etc." />
  <p class="caption"><strong>Figure 10.1.</strong> A common taxonomy of deep generative models (directed graphical models with tractable density, undirected models, models defined by a simulator without explicit density, and related categories). The figure is based on material from I. Goodfellow’s NIPS 2016 tutorial on GANs (arXiv:1701.00160). Image: <a href="https://commons.wikimedia.org/wiki/File:Types_of_deep_generative_models.png">“Types of deep generative models.png”</a> by Cosmia Nebula, <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>.</p>
</div>

---

## 10.3 Variational Autoencoders (VAEs)

A **VAE** pairs an **encoder** $q_\phi(z \mid x)$ that maps data to a distribution over latents $z$, with a **decoder** $p_\theta(x \mid z)$ that maps latents back to data. Training maximizes the **evidence lower bound (ELBO)** on $\log p_\theta(x)$, which balances reconstruction quality and a penalty that keeps $q_\phi(z \mid x)$ close to a prior $p(z)$ (often Gaussian).

VAEs support **interpolation** in latent space and **conditional** variants $p(x \mid c)$ by feeding $c$ into the encoder or decoder. In genomics, variational ideas appear in tools that model **count data** and **batch effects** (for example variational autoencoder–style models for single-cell data), where the latent space can capture biological variation while separating technical noise.

<div class="figure">
  <img src="https://upload.wikimedia.org/wikipedia/commons/4/4a/VAE_Basic.png" alt="Variational autoencoder: encoder to latent distribution, decoder to reconstruction" />
  <p class="caption"><strong>Figure 10.2.</strong> Structure of a variational autoencoder: data are mapped to a distribution over latents, a latent vector is sampled, and the decoder reconstructs the input. <a href="https://commons.wikimedia.org/wiki/File:VAE_Basic.png">“VAE Basic.png”</a> by EugenioTL, <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>.</p>
</div>

---

## 10.4 Generative Adversarial Networks (GANs)

**GANs**, introduced by Goodfellow et al., frame generation as a two-player game. A **generator** $G$ maps noise $z$ (often low-dimensional Gaussian) to fake samples $\tilde{x} = G(z)$. A **discriminator** $D$ receives either real $x$ from the dataset or fake $\tilde{x}$ and tries to classify **real vs. fake**. $G$ is trained to fool $D$; $D$ is trained to be accurate. Under ideal conditions, the distribution of $G(z)$ matches the data distribution.

<div class="figure">
  <img src="https://upload.wikimedia.org/wikipedia/commons/8/8b/Generative_Adversarial_Network_illustration.svg" alt="GAN: noise to generator, fake image to discriminator; real image also to discriminator" />
  <p class="caption"><strong>Figure 10.3.</strong> GAN schematic: noise feeds the generator; fake and real images feed the same discriminator, which is trained to output low score for fake and high for real, while the generator is trained to produce fakes classified as real. <a href="https://commons.wikimedia.org/wiki/File:Generative_Adversarial_Network_illustration.svg">“Generative Adversarial Network illustration.svg”</a> by Mtanti, <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>.</p>
</div>

**Practical notes** you will meet in papers and courses:

- Training can be **unstable**; remedies include architecture choices, alternative objectives (for example Wasserstein GAN variants), and careful learning-rate schedules.
- **Mode collapse** occurs when $G$ outputs a narrow set of samples that still fool $D$.
- **Conditional GANs** condition both $G$ and $D$ on labels or other inputs, useful when you want generation under class or phenotype constraints.
- **CycleGAN** and similar approaches learn mappings between domains without paired examples, relevant when paired “before/after” data are scarce.

In biology and medicine, GANs and adversarial ideas have been used for **image synthesis** (for example histology-style images), **domain adaptation**, and **data augmentation** under careful validation, because evaluation of clinical or biological fidelity is non-trivial.

---

## 10.5 Diffusion Models and Denoising

**Diffusion models** define a **forward** process that gradually adds noise until data resemble a simple distribution, and learn a **reverse** process (or **score** model) that denoises step by step. They are a strong fit for **image** generation and are increasingly used in other modalities when paired with appropriate architectures.

<div class="figure">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Example_of_Denoising_Diffusion_models.png/960px-Example_of_Denoising_Diffusion_models.png" alt="Denoising diffusion: noisy image gradually denoised to recognizable image" />
  <p class="caption"><strong>Figure 10.4.</strong> Denoising view of a diffusion-style process: a heavily noisy image is progressively denoised until a plausible image appears (with variation). <a href="https://commons.wikimedia.org/wiki/File:Example_of_Denoising_Diffusion_models.png">“Example of Denoising Diffusion models.png”</a> by MrAlanKoh, <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a> (derived from a Wikimedia photo of a rhinoceros; see file page for details).</p>
</div>

**Normalizing flows** (not illustrated here) offer another route to **exact likelihood** for continuous data when the Jacobian of each invertible layer is tractable. They are often used when calibrated density matters.

Together, VAEs, GANs, flows, and diffusion show a pattern: **different trade-offs** among sample quality, likelihood, training stability, and conditioning. Modern **large language models** are mostly **autoregressive** or **masked** objectives on sequences, implemented with **Transformers** and **attention**, which we turn to next.

---

## 10.6 From Autoregressive Generation to Attention

An **autoregressive** model writes

$$
p(x_1,\ldots,x_N) = \prod_{t=1}^{N} p(x_t \mid x_1,\ldots,x_{t-1}).
$$

For text, $x_t$ might be subword tokens; for DNA or protein, characters or k-mers. **Recurrent** networks were once the default for such chains; **convolutions** with wide receptive fields or stacked dilated convolutions also work. **Transformers** replace recurrence with **self-attention**, so every position can **attend** to every other position in one or a few layers, with heavy **parallelism** on modern hardware.

So the bridge from “generative models” to “Transformers” is direct: **decoder-only Transformers** (GPT-style) are **next-token generative models** trained with cross-entropy. **BERT-style** models are often trained with **masked token prediction** (a denoising-style objective on text). Both rely on the same attention machinery.

For many biology problems, you still need **relationships between distant positions**: regulatory interactions along DNA, RNA base pairing, protein contacts, or long-range references in biomedical text. The following sections summarize the **Transformer** architecture and **self-attention** as used in generative and representation-learning settings.

---

## 10.7 Motivation: Why Self-Attention?

### 10.7.1 The limitation of local context

Convolutions use a fixed-size window. If influence spans farther than that window, you must stack many layers to grow the receptive field.

RNNs can, in principle, carry information across time, but long-range credit assignment is hard, and computation is sequential.

### 10.7.2 The core idea of attention

**Self-attention** lets each position interact **directly** with every other position. For length $N$, the model learns **which tokens matter** for each position and forms a **weighted mixture** of information. That yields **content-based long-range** interactions without recurrence.

---

## 10.8 Transformer Architecture (Encoder and Decoder)

A Transformer stacks identical-style blocks. Common patterns:

- **Encoder-only** (BERT-style): bidirectional context for embeddings and classification.
- **Decoder-only** (GPT-style): **causal** attention, standard for **generation**.
- **Encoder-decoder** (original sequence-to-sequence): both stacks, still used in translation and some scientific pipelines.

### 10.8.1 High-level architecture

Typical ingredients per layer:

- **Multi-head self-attention**,
- **Position-wise feed-forward network** (FFN),
- **Residual connections**,
- **Layer normalization**.

<div class="figure">
  <img src="https://upload.wikimedia.org/wikipedia/commons/3/34/Transformer%2C_full_architecture.png" alt="Transformer architecture showing encoder and attention" />
  <p class="caption"><strong>Figure 10.5.</strong> Full Transformer architecture (encoder and decoder stacks). Artwork by dvgodoy (<a href="https://github.com/dvgodoy/dl-visuals">dl-visuals</a>), on Wikimedia Commons as <a href="https://commons.wikimedia.org/wiki/File:Transformer,_full_architecture.png">“Transformer, full architecture.png”</a>, <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.</p>
</div>

### 10.8.2 Residual connections and layer normalization

Residual paths help optimization in deep stacks. Layer normalization stabilizes activations across features.

---

## 10.9 Self-Attention and Attention Heads

### 10.9.1 Query, key, value (Q, K, V)

Let input embeddings be $X \in \mathbb{R}^{N \times d_{model}}$. For one attention head,

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

with learned matrices $W_Q, W_K, W_V$.

### 10.9.2 Scaled dot-product attention

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Here $QK^T$ is an $N \times N$ similarity matrix; softmax produces weights that sum to 1 per query row; multiplying by $V$ mixes value vectors. Scaling by $\sqrt{d_k}$ avoids large dot products in high dimensions.

<div class="figure">
  <img src="https://upload.wikimedia.org/wikipedia/commons/8/81/Attention-qkv.png" alt="Attention calculation flow through Q, K, V" />
  <p class="caption"><strong>Figure 10.6.</strong> Computation flow for scaled dot-product attention with Q, K, and V. <a href="https://commons.wikimedia.org/wiki/File:Attention-qkv.png">“Attention-qkv.png”</a> by Numiri, <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>.</p>
</div>

### 10.9.3 Intuition

Each position issues a **query**; keys score compatibility; **values** are blended into a new representation. Attention is **learned routing** driven by content.

---

## 10.10 Multi-Head Attention

Multiple heads run in parallel; each can specialize (syntax, long-range, local motifs).

<div class="figure">
  <img src="https://upload.wikimedia.org/wikipedia/commons/6/68/Attention_Is_All_You_Need_-_Multiheaded_Attention.png" alt="Multi-headed attention diagram" />
  <p class="caption"><strong>Figure 10.7.</strong> Multi-head attention from Vaswani et al., “Attention Is All You Need” (<a href="https://arxiv.org/abs/1706.03762">arXiv:1706.03762</a>); figure reproduced under Google’s terms noted on the paper. File on Commons: <a href="https://commons.wikimedia.org/wiki/File:Attention_Is_All_You_Need_-_Multiheaded_Attention.png">“Attention Is All You Need - Multiheaded Attention.png”</a>, <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>.</p>
</div>

For head $h$,

$$
\text{head}_h = \text{Attention}(XW_Q^{(h)}, XW_K^{(h)}, XW_V^{(h)})
$$

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1,\ldots,\text{head}_H)W_O
$$

---

## 10.11 Positional Information

Self-attention is permutation-invariant without extra structure. **Positional encodings** inject order:

1. **Sinusoidal** encodings (fixed functions of position),
2. **Learned** positional embeddings (up to a maximum length).

Classic sinusoidal form for dimension index $i$:

$$
\text{PE}(\text{pos},2i) = \sin\left(\frac{\text{pos}}{10000^{2i/d_{model}}}\right)
$$

$$
\text{PE}(\text{pos},2i+1) = \cos\left(\frac{\text{pos}}{10000^{2i/d_{model}}}\right)
$$

These are added to token embeddings before attention layers.

---

## 10.12 Sequence Modeling with Transformers

### 10.12.1 Autoregressive modeling (GPT-style)

$$
p(x_{t+1}\mid x_{1:t})
$$

Training minimizes cross-entropy for next-token prediction. This is **generative modeling** in the autoregressive sense.

### 10.12.2 Causal masking

Decoder self-attention uses a **causal mask** so position $t$ attends only to positions $\le t$.

<div class="figure">
  <img src="https://upload.wikimedia.org/wikipedia/commons/7/72/Decoder_self-attention_with_causal_masking%2C_detailed_diagram.png" alt="Decoder self-attention with causal masking diagram" />
  <p class="caption"><strong>Figure 10.8.</strong> Decoder self-attention with causal masking so future tokens do not leak into the present. Artwork by dvgodoy (<a href="https://github.com/dvgodoy/dl-visuals">dl-visuals</a>), <a href="https://commons.wikimedia.org/wiki/File:Decoder_self-attention_with_causal_masking,_detailed_diagram.png">“Decoder self-attention with causal masking, detailed diagram.png”</a>, <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.</p>
</div>

### 10.12.3 Task variants

- **Text or sequence generation**: decoder-only, causal mask.
- **Masked language modeling**: encoder-only, bidirectional attention with random masks.
- **Sequence classification**: pooling or a special token, then a classifier head.
- **Per-token labeling**: output head on each position.

---

## 10.13 Fine-Tuning Pretrained Models

Pretrained Transformers supply strong initial representations. **Fine-tuning** adapts them to your task.

### 10.13.1 Common strategies

1. **Train all layers** with a task head (strong when you have enough labels).
2. **Freeze the backbone**, train only the head (small label sets).
3. **Parameter-efficient fine-tuning (PEFT)**: adapters, **LoRA**, and similar methods update a small parameter subset.

### 10.13.2 Conceptual workflow

Typical steps: tokenize, build datasets, optimize with a framework such as Hugging Face `transformers`, evaluate on validation and test sets. The main idea is that **pretraining** plus **task loss** yields better data efficiency than training from random initialization.

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

---

## 10.14 Applications: LLMs, Genomics, Proteomics, Biomedical NLP

### 10.14.1 GPT and LLMs

Large **decoder-only** Transformers trained on text with next-token loss underpin modern **LLMs**. Instruction tuning and alignment build on that generative core.

### 10.14.2 Genomics

Tokens can be nucleotides or k-mers; attention can relate **distant regulatory context**. Applications include regulatory activity, binding prediction, and variant effect scoring, often with **fine-tuned** or **specialized** sequence models.

### 10.14.3 Proteomics

**Protein language models** learn residue-level representations; attention can reflect **long-range** patterns related to **structure** and function.

### 10.14.4 Biomedical NLP

Transformers support NER, relation extraction, QA, and summarization over literature and clinical text, where **transfer learning** reduces annotation needs.

### 10.14.5 How this ties back to generative modeling

- **GANs and diffusion** dominate many **image** pipelines and some **signal** domains.
- **Autoregressive Transformers** dominate **text** and many **sequence** benchmarks.
- **VAEs and probabilistic** models remain important where **latent structure** or **uncertainty** is central (for example some single-cell and factor models).

Choosing a family depends on **modality**, need for **likelihoods**, **data scale**, and **evaluation** constraints in your application.

---

## 10.15 Visualization of Attention Maps

Attention matrices (query vs. key positions) can be plotted as heatmaps. They offer **intuition** about what the model emphasizes.

**Caveat:** attention is not a guaranteed causal explanation. Combine it with **perturbations**, **gradients**, and **multiple heads and layers** for more reliable insight.

---

## Summary

- **Generative models** target data distributions or samplers; objectives include likelihood bounds, adversarial games, score matching, and autoregressive factors.
- **VAEs** use encoder-decoder structure and the ELBO; **GANs** use generator vs. discriminator training; **diffusion** learns to reverse a noise process.
- **Autoregressive** sequence models connect naturally to **decoder-only Transformers** for generation.
- **Self-attention** replaces recurrence with global, content-based mixing; **multi-head** attention increases expressivity.
- **Causal masking** enforces order for generation; **pretraining and fine-tuning** transfer models to biology and NLP tasks.
- **Attention plots** are useful but should be validated with other tools.

---

## Topics for the Next Class

Possible follow-ups:

- implement or fine-tune a small **sequence model** on a biological dataset,
- compare **discriminative** vs. **generative** objectives on the same features,
- discuss **evaluation** of synthetic biological or medical data (privacy, bias, utility metrics).

---

## Further Reading

- [Generative model, Wikipedia](https://en.wikipedia.org/wiki/Generative_model)
- [Variational autoencoder, Wikipedia](https://en.wikipedia.org/wiki/Variational_autoencoder)
- [Generative adversarial network, Wikipedia](https://en.wikipedia.org/wiki/Generative_adversarial_network)
- [Diffusion model, Wikipedia](https://en.wikipedia.org/wiki/Diffusion_model)
- [Transformer (deep learning architecture), Wikipedia](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture))
- [Attention (machine learning), Wikipedia](https://en.wikipedia.org/wiki/Attention_(machine_learning))
- [Large language model, Wikipedia](https://en.wikipedia.org/wiki/Large_language_model)
- Goodfellow, I., *NIPS 2016 Tutorial: Generative Adversarial Networks*, [arXiv:1701.00160](https://arxiv.org/abs/1701.00160)
