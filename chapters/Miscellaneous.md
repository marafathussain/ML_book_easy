# Miscellaneous Notes

Personal notes on selected topics. 

---

## S4 (Structured State Space for Sequence modeling) / State Space Models — Sections 2.2–2.4 and Section 3 (Gu et al., 2022)

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

Suppose $\bar{A}$ is diagonal with entries $\bar{A}_0, \bar{A}_1, \ldots, \bar{A}_{N-1}$. Then the $\ell$-th entry of the kernel is a sum over the N dimensions:

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

---

## PCA: One Equation and What It Means

The following notes unpack the core of Principal Component Analysis (PCA): a single equation, then variance and covariance, then the quadratic form that ties them together.

### The beating heart of PCA

This equation is the beating heart of Principal Component Analysis (PCA). It looks compact and innocent, but it is really asking a single question:

*"In which direction should I look at my data so that I see the most variation?"*

We have:

$$
\mathbf{w}_1 = \arg\max_{\|\mathbf{w}\|=1} \text{Var}(\mathbf{X} \mathbf{w})
$$

Unpacking it slowly:

**What is $\mathbf{X}$?** It is your data matrix. Each row is a sample; each column is a feature. Usually we assume it is mean-centered (we have subtracted the column means so the data cloud is centered at the origin). That centering matters because variance is about spread around the mean (or zero after centering).

**What is $\mathbf{X}\mathbf{w}$?** This is the projection of the data onto a direction $\mathbf{w}$. Geometrically: your dataset is a cloud of points in high-dimensional space. Multiplying by $\mathbf{w}$ collapses that cloud onto a single line. So $\mathbf{X}\mathbf{w}$ gives the coordinates of every point along that line.

**What is $\text{Var}(\mathbf{X}\mathbf{w})$?** The variance of those projected values. In plain terms: how spread out is the data along that direction?

**What does $\arg\max_{\|\mathbf{w}\|=1}$ mean?** Among all vectors $\mathbf{w}$ that have length 1, find the one that maximizes the variance. Why force $\|\mathbf{w}\| = 1$? Because otherwise we could cheat: we could scale $\mathbf{w}$ to be huge and artificially inflate the variance. The unit-length constraint makes it a pure direction problem.

So the equation says: *The first principal component $\mathbf{w}_1$ is the unit vector that makes the projected data have the largest possible variance.* That is the whole idea of PCA in one line.

**The structure underneath.** If the data is centered, we can rewrite:

$$
\text{Var}(\mathbf{X}\mathbf{w}) = \mathbf{w}^\top \mathbf{S} \mathbf{w}
$$

where $\mathbf{S}$ is the covariance matrix, $\mathbf{S} = \frac{1}{n} \mathbf{X}^\top \mathbf{X}$. So the optimization becomes:

$$
\max_{\|\mathbf{w}\|=1} \mathbf{w}^\top \mathbf{S} \mathbf{w}
$$

This is an eigenvalue problem. The solution: $\mathbf{w}_1$ is the eigenvector of $\mathbf{S}$ with the largest eigenvalue, and that eigenvalue equals the maximum variance. So PCA is saying: *Find the direction in which the covariance matrix stretches space the most.* Geometrically, if your data cloud is an ellipse (or ellipsoid in higher dimensions), PCA finds the long axis first, then the second longest axis orthogonal to the first, and so on.

PCA does not care about labels or prediction. It just asks: *Where is the structure?* In high-dimensional settings (e.g. many features, relatively few samples), this equation finds the dominant modes of variability before overfitting dominates.

### Variance and covariance: same family, different jobs

Variance and covariance are cousins. Same family. Different jobs.

**Variance** answers: "How much does one variable wiggle?"

**Covariance** answers: "How do two variables wiggle together?"

When we stack all pairwise covariances into a grid, we get the **covariance matrix**. That matrix is the full story of how everything co-moves.

**Concrete example.** Imagine three students and two exam scores:

- Math: 70, 80, 90  
- Physics: 65, 85, 95  

**Variance** of Math measures how spread out those math scores are around their mean. If the math scores were 80, 80, 80 then variance = 0 (no spread). With 70, 80, 90 they spread out, so variance is positive:

$$
\text{Var}(X) = \frac{1}{n} \sum (x_i - \bar{x})^2
$$

We square deviations because we care about magnitude of spread, not direction.

**Covariance** between Math and Physics looks at whether they increase together. If higher math tends to come with higher physics, covariance is positive. If higher math comes with lower physics, covariance is negative. If no pattern, near zero:

$$
\text{Cov}(X,Y) = \frac{1}{n} \sum (x_i - \bar{x})(y_i - \bar{y})
$$

Notice: variance is just covariance of a variable with itself, $\text{Var}(X) = \text{Cov}(X,X)$.

**The covariance matrix.** For two variables (Math and Physics):

$$
\mathbf{S} =
\begin{bmatrix}
\text{Var}(\text{Math}) & \text{Cov}(\text{Math, Physics}) \\
\text{Cov}(\text{Physics, Math}) & \text{Var}(\text{Physics})
\end{bmatrix}
$$

Diagonal entries are variances; off-diagonal entries are covariances. So variance is a single number; the covariance matrix is the full interaction blueprint.

**Geometric intuition.** Variance tells you how wide the data cloud is along one axis. Covariance tells you whether the cloud is tilted. If covariance is zero, the cloud is axis-aligned. If positive, the cloud tilts upward; if negative, downward. That is why in PCA the optimization $\max_{\|\mathbf{w}\|=1} \mathbf{w}^\top \mathbf{S} \mathbf{w}$ uses $\mathbf{S}$: variance alone gives spread in each coordinate, but covariance tells you how directions combine. The eigenvectors of $\mathbf{S}$ find the tilted axes of the ellipsoid. In high dimensions (e.g. many MRI features), variance tells you which single feature fluctuates most; the covariance matrix tells you which *patterns* of features fluctuate together. Variance is local; the covariance matrix is structural.

### Unpacking the quadratic form $\mathbf{w}^\top \mathbf{S} \mathbf{w}$

We can write $\mathbf{w}^\top \mathbf{S} \mathbf{w}$ in element form. Assume $\mathbf{w}$ is $p \times 1$ and $\mathbf{S}$ is $p \times p$. Then:

$$
\mathbf{w}^\top \mathbf{S} \mathbf{w} = \sum_{i=1}^{p} \sum_{j=1}^{p} w_i S_{ij} w_j
$$

**Interpretation.** Split into two parts:

1. **Diagonal terms** ($i = j$): $\sum_{i=1}^{p} w_i^2 S_{ii}$. Since $S_{ii} = \text{Var}(X_i)$, this is a weighted sum of individual variances.

2. **Off-diagonal terms** ($i \neq j$): $\sum_{i \neq j} w_i w_j S_{ij}$. These involve covariances between different features.

So:

$$
\mathbf{w}^\top \mathbf{S} \mathbf{w} = \sum_{i=1}^{p} w_i^2 \text{Var}(X_i) + \sum_{i \neq j} w_i w_j \text{Cov}(X_i, X_j)
$$

The variance of the projection is not just "variance of features times weights squared." It also includes every pairwise interaction, weighted by $w_i w_j$. That is why PCA does not simply pick the feature with the largest variance; it finds a direction that uses the covariance structure. In high dimensions (e.g. thousands of features), this expression contains a huge number of interaction terms; the geometry of that quadratic form determines how the data cloud is shaped.

**Punchline:** If $\mathbf{w}$ is an eigenvector of $\mathbf{S}$ with eigenvalue $\lambda$, then $\mathbf{w}^\top \mathbf{S} \mathbf{w} = \lambda$ (for a unit $\mathbf{w}$). So the messy double sum collapses to a single number. That is why PCA reduces to "largest eigenvalue."

