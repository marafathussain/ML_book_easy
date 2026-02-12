# Miscellaneous Notes

Personal notes on selected topics for later review. 

---

## S4 / State Space Models — Sections 2.2–2.4 and Section 3 (Gu et al., 2022)

Reference: *Efficiently Modeling Long Sequences with Structured State Spaces*, Gu et al., arXiv:2111.00396v3. The following summarizes Sections 2.2–2.4 (background on SSMs, HiPPO, discrete recurrence, convolutional view) and Section 3 (S4 parameterization: how to compute the convolution kernel efficiently).

The diagram below shows the three equivalent views: continuous SSM → discretize to recurrence → unroll to convolution.

<div class="figure">
  <img src="figures/misc/ssm_three_views.svg" alt="Three views of an SSM: continuous, discrete recurrent, and convolution" />
  <p class="caption">Three views of an SSM (Sections 2.2–2.4): continuous time, discrete recurrence, and convolution.</p>
</div>

---

## What Is a State Space Model?

A State Space Model (SSM) takes a 1-D input sequence $u$ (e.g. a signal over time) and produces an output sequence $y$, while maintaining an internal hidden state $x$:

$$
\begin{aligned}
x'(t) &= A x(t) + B u(t) \\
y(t)  &= C x(t) + D u(t)
\end{aligned}
$$

Think of $x(t)$ as a memory that evolves over time from its past and the current input. The matrices $A$, $B$, $C$, and $D$ are learned and define how the memory updates and how it is read out. For simplicity the paper sets $D=0$ (no direct input-to-output path).

---

## Section 2.2 — Long-Range Dependencies and HiPPO

Why do we care about the choice of $A$?

If $A$ is arbitrary, the hidden state often decays or explodes exponentially, so the model quickly forgets old inputs or becomes unstable. That is bad for long sequences.

HiPPO (High-Order Polynomial Projection Operators) gives a principled way to choose $A$ so that the hidden state can represent the whole input history well. The idea: instead of naive decay, a special $A$ keeps a compact summary of the past that evolves in a stable way.

The HiPPO matrix $A$ has this form (structure matters more than the exact constants):

$$
A_{n,k} =
\begin{cases}
-\sqrt{(2n+1)(2k+1)} & \text{if } n > k \\
n+1 & \text{if } n = k \\
0 & \text{if } n < k
\end{cases}
$$

Above the diagonal, the scaling encodes how to project past inputs into a basis that represents the history well. This helps gradient flow and reduces vanishing/exploding gradients.

Rough picture of the HiPPO $A$ structure (lower-triangular, with a specific pattern):

```
HiPPO A (N×N) — lower triangular
[  1    0    0   ... ]
[ -√3   2    0   ... ]
[ -√15 -√7   3  ... ]
[  ...       ...   ]  → stores history efficiently
```

A random $A$ would mix dimensions in an unstructured way; HiPPO spreads information in a controlled way across the state so that long-range dependencies are easier to learn.

---

## Section 2.3 — Discrete-Time SSM: Recurrent View

The continuous-time SSM is discretized for sequences $u_0, u_1, u_2, \ldots$.

Using a bilinear (Tustin) transform, the discrete recurrence is:

$$
x_k = \bar{A} x_{k-1} + \bar{B} u_k
$$
$$
y_k = C x_k
$$

with $\bar{A} = (I - \frac{\Delta}{2}A)^{-1}(I + \frac{\Delta}{2}A)$, where $\Delta$ is the step size. This is a recurrent update like an RNN: state at step $k$ depends on state at $k-1$ and input $u_k$.

Unrolling a few steps (with $\bar{A}$, $\bar{B}$ written as $A$, $B$ in discrete form):

$$
\begin{aligned}
x_0 &= B u_0 \\
x_1 &= A x_0 + B u_1 \\
x_2 &= A^2 B u_0 + A B u_1 + B u_2
\end{aligned}
$$

So the current state is a weighted sum of all past inputs — exactly the structure of an RNN, but with weights coming from the SSM matrices.

---

## Section 2.4 — Training SSMs: Convolutional View

The same recurrence can be written as a convolution, which is the key to efficient training.

If you unroll the recurrence, the output sequence $y$ equals a discrete convolution of the input with a kernel $K$:

$$
y[k] = (K * u)[k] = \sum_{i=0}^{k} K[i]\, u[k-i]
$$

The kernel is given by powers of the discrete $A$ matrix:

$$
K[i] = C A^{i-1} B
$$

So the SSM defines a convolution filter whose coefficients are determined by $A$, $B$, and $C$.

Rough picture of how the convolution looks:

```
u:   u0  u1  u2  u3  ...
K:   K0  K1  K2  K3  ...

y0 = K0·u0
y1 = K1·u0 + K0·u1
y2 = K2·u0 + K1·u1 + K0·u2
...
```

This is standard causal convolution: each output is a weighted sum of current and past inputs. The difference is that here the kernel $K$ is not free parameters but is derived from the state-space matrices.

Why this matters:

| View | Computation | Use case |
|------|-------------|----------|
| Recurrence | Sequential (step-by-step) | Inference, small sequences |
| Convolution | Parallel (e.g. FFT) | Fast training on GPUs/TPUs |

The rest of the paper (e.g. S4) is about choosing a structured $A$ so that this convolution kernel can be computed efficiently.

---

## Section 3 — S4 Parameterization: Computing the Kernel Efficiently

Section 2.4 showed that the SSM output is a convolution with kernel $K$, where $K[\ell] = C \bar{A}^{\ell-1} B$. So to run the model we need this kernel. The problem: computing it the obvious way is too expensive for long sequences.

### The bottleneck

To get $K$ we need the sequence $C\bar{B},\, C\bar{A}\bar{B},\, C\bar{A}^2\bar{B},\, \ldots,\, C\bar{A}^{L-1}\bar{B}$. Doing that with a general $N \times N$ matrix $\bar{A}$ means $L$ matrix–vector products, which costs $O(N^2 L)$ time and $O(NL)$ space. For long sequences (e.g. $L = 10{,}000$) and moderate $N$, that is prohibitive. We want something closer to $O(N + L)$.

### Change of basis (conjugation)

A key fact: the input–output map $u \mapsto y$ does not change if we replace $(A, B, C)$ by a “conjugated” version. For any invertible matrix $V$:

$$
(A, B, C) \quad \sim \quad (V^{-1} A V,\; V^{-1} B,\; C V)
$$

So we can put $A$ (and hence $\bar{A}$ after discretization) into a nicer form by a change of basis, without changing what the SSM computes. In particular, if $A$ is diagonalizable, we can assume it is already diagonal (in the right basis). That is the starting point for efficient kernel computation.

### Diagonal case: Vandermonde = fast kernel

Suppose $\bar{A}$ is diagonal with entries $\bar{A}_0$, $\bar{A}_1$, $\ldots$, $\bar{A}_{N-1}$. Then the $\ell$-th entry of the kernel is a sum over the $N$ dimensions:

$$
\bar{K}[\ell] = C \bar{A}^{\ell} B = \sum_{n=0}^{N-1} C_n \bar{A}_n^{\ell} B_n
$$

This is exactly one row–vector product where the “matrix” has columns of the form $(1, \bar{A}_n, \bar{A}_n^2, \ldots, \bar{A}_n^{L-1})$. That matrix is a Vandermonde matrix: its columns are powers of the $\bar{A}_n$. Vandermonde structure is well studied: we can compute this in $\widetilde{O}(N+L)$ time and $O(N+L)$ space (or a simple $O(NL)$ implementation that still avoids materializing the full $N \times L$ matrix). So diagonal SSMs give a fast way to compute $K$.

Rough picture:

```
Diagonal Ā  →  K = [row] × [Vandermonde matrix]
                        (1, Ā₀, Ā₀², ...)
                        (1, Ā₁, Ā₁², ...)
                        ...
                   →  one structured matrix–vector product, Õ(N+L)
```

### Why not use a diagonal A from the start?

HiPPO (Section 2.2) gives an $A$ that is great for long-range memory, but the HiPPO matrix is not diagonal — it is lower triangular with a specific pattern. So we cannot directly use the “diagonal = Vandermonde” trick on the raw HiPPO matrix.

### S4’s idea: diagonal + low-rank (DPLR)

S4 writes the (continuous) state matrix as:

$$
A = \text{diagonal part} + \text{low-rank correction}
$$

This is called a diagonal-plus-low-rank (DPLR) matrix. The HiPPO matrix can be approximated or transformed into this form. Why is that useful?

- The diagonal part contributes a “Vandermonde-like” structure.
- The low-rank part adds a correction that can be handled with extra terms.
- Together, the full kernel $K$ can still be computed without forming all powers $\bar{A}^0, \bar{A}^1, \ldots, \bar{A}^{L-1}$ explicitly. The algebra reduces the SSM computation to operations involving a Cauchy kernel (a structured matrix related to Vandermonde). Cauchy kernels have fast algorithms (about $\widetilde{O}(N+L)$ time and $O(N+L)$ space).

So S4 keeps the benefits of HiPPO (long-range dependencies) but makes the convolution kernel computable in roughly linear cost.

### Summary of Section 3 in plain terms

| Step | Idea |
|------|------|
| 1 | Computing $K$ naively is $O(N^2 L)$ — too slow for long $L$. |
| 2 | Conjugation: we can change basis so $A$ has a “nice” form without changing $u \mapsto y$. |
| 3 | If $A$ (hence $\bar{A}$) were diagonal, $K$ would be a Vandermonde product → $\widetilde{O}(N+L)$. |
| 4 | HiPPO’s $A$ is not diagonal; S4 writes it as diagonal + low-rank (DPLR). |
| 5 | DPLR structure reduces the SSM to a Cauchy kernel → still $\widetilde{O}(N+L)$ time and $O(N+L)$ memory. |

So Section 3 answers: “How do we compute the SSM convolution kernel $K$ efficiently?” By structuring $A$ as DPLR (inspired by HiPPO), S4 reduces the work to a Cauchy kernel and achieves nearly linear cost in $N$ and $L$, which is what makes long-sequence SSMs practical. Later work (e.g. S4D) showed that in many cases a purely diagonal $A$ with a good initialization can also capture long-range dependencies and simplifies the implementation (e.g. kernel = one Vandermonde product) while keeping similar efficiency.

