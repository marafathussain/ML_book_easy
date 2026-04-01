# Chapter 11: Explainable AI and Gradient-Based Optimization

## Introduction

Machine learning models in biology and medicine often affect **diagnosis, treatment planning, and resource allocation**. **Explainable AI (XAI)** asks how a model uses inputs to reach a prediction, so experts can **audit**, **trust**, and **debug** systems. No single method gives a complete causal story, but **attribution methods** (feature importance maps, saliency) and **local surrogates** (LIME) help build intuition when used critically.

Separately, nearly all deep models are trained with **gradient-based optimization**. Understanding **gradient descent**, **learning rates**, and **initialization** is essential when you tune models or interpret training curves.

This chapter combines:

- **Explainability**: SHAP, LIME, Grad-CAM and related ideas, plus other common tools,
- **Optimization**: gradient descent in two parameters, ascent vs descent relative to the loss, learning rate, and weight initialization,

with **equations** and **figures** from **Wikimedia Commons** (each figure caption cites the file, author, and license).

---

## 11.1 Why Explainable AI Matters in the Life Sciences

- **Validation**: Does the model use plausible biological signals (for example known histology patterns) or spurious shortcuts (stains, scanner artifacts)?
- **Regulation and ethics**: Some settings require transparency about decision support.
- **Science**: Attribution can suggest **hypotheses** (which genes or image regions drive a prediction), but follow-up experiments are still needed.

<div class="figure">
  <img src="https://upload.wikimedia.org/wikipedia/commons/3/37/Overview_of_TalkToModel_for_explainable_AI_conclusion-making.webp" alt="TalkToModel: user dialogue parsed to executable operations for explainable ML" />
  <p class="caption"><strong>Figure 11.1.</strong> Example of an interactive explainable-AI workflow: natural language is parsed into operations that query or explain a model (TalkToModel). From Slack et al., <em>Nature Machine Intelligence</em> (2023). File: <a href="https://commons.wikimedia.org/wiki/File:Overview_of_TalkToModel_for_explainable_AI_conclusion-making.webp">“Overview of TalkToModel for explainable AI conclusion-making.webp”</a>, uploaded by Prototyperspective, <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.</p>
</div>

---

## 11.2 A Small Taxonomy of Explanations

| **Axis** | **Examples** |
| --- | --- |
| **Local vs. global** | Local: why this patient? Global: which features matter on average? |
| **Model-agnostic vs. model-specific** | Agnostic: treat model as black box (LIME, KernelSHAP). Specific: use gradients or structure (Grad-CAM, TreeSHAP). |
| **Feature attribution vs. example-based** | Attribution: scores per input dimension. Example-based: “similar training cases” (prototypes, k-NN in embedding space). |

---

## 11.3 SHAP (SHapley Additive exPlanations)

### 11.3.1 Shapley values (game theory)

SHAP builds on **Shapley values** from cooperative game theory. Suppose $N$ players jointly produce a **payout** $v(S)$ for any subset $S \subseteq N$. Player $i$’s Shapley value averages their **marginal contribution** when joining coalitions:

$$
\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}}
\frac{|S|!\,(|N|-|S|-1)!}{|N|!}
\bigl[ v(S \cup \{i\}) - v(S) \bigr].
$$

Intuitively, $\phi_i$ is a **fair** share of the total value under certain axioms (efficiency, symmetry, dummy player, additivity).

### 11.3.2 From games to machine learning

For a model $f$ and input $x$, interpret **features** (or groups of features) as players. Define a **value function** $v_f(S)$ as the model’s expected output when only features in $S$ are present and others are **marginalized** out according to a reference or background distribution. **SHAP values** $\phi_i$ attribute how much each feature $i$ moves the prediction relative to a **baseline** $f(x')$ (for example average prediction over a background dataset).

The **additive** explanation (for a single output) takes the form:

$$
f(x) \approx \phi_0 + \sum_{i=1}^{M} \phi_i,
$$

where $\phi_0$ is the baseline and $\sum_i \phi_i$ is the sum of SHAP values over $M$ features or groups, satisfying **local accuracy** when the full Shapley setup is used exactly.

### 11.3.3 Practical algorithms

Exact Shapley values require exponential cost in the number of features. **Approximations** are standard:

- **KernelSHAP**: weighted linear regression on sampled coalitions (model-agnostic, slower for large input dimension).
- **TreeSHAP**: exact values for **tree ensembles** (XGBoost, random forests) in polynomial time in tree depth and number of trees.
- **DeepSHAP / DeepExplainer**: combines ideas from **DeepLIFT** with Shapley-style propagation for neural networks (approximate but fast for many architectures).

### 11.3.4 Extensions and improvements

- **Interaction values**: quantify **pairwise** feature interactions (beyond additive main effects).
- **Causal / interventional** variants: change how “missing” features are imputed to align better with **interventional** rather than purely observational semantics (active research area).
- **MultiSHAP** and similar work extend Shapley-style reasoning to **multimodal** models.

**Caveats**: SHAP values depend on the **choice of background distribution** and **feature grouping**; different choices yield different attributions.

---

## 11.4 LIME (Local Interpretable Model-agnostic Explanations)

**LIME** explains a **single prediction** by fitting a **simple interpretable model** $g$ (e.g., sparse linear model) **locally** around the input $x$:

1. **Perturb** the input (sample $z$ near $x$).
2. **Weight** samples by proximity $\pi_x(z)$ (for example exponential kernel in distance).
3. **Fit** $g$ to approximate $f(z)$ with low complexity (penalty on number of nonzero coefficients).

Formally, one minimizes **fidelity** to the black-box $f$ on perturbed inputs, plus a **complexity** penalty:

$$
\xi(x) = \arg\min_{g \in \mathcal{G}} \;
\sum_{z \in \mathcal{Z}} \pi_x(z)\,\bigl[ f(z) - g(z) \bigr]^2 + \Omega(g),
$$

where $\mathcal{Z}$ is a set of samples around $x$, $\mathcal{G}$ is a class of simple models (for example sparse linear models in an interpretable representation of $z$), and $\Omega$ encourages sparsity.

**Strengths**: works for **any** model $f$, including black-box APIs.

**Limitations**: explanations can be **unstable** under small changes to $x$ or sampling; locality may not match global behavior; **correlated features** are hard to separate.

---

## 11.5 Grad-CAM and Related Gradient-Based Visualizations

**Class Activation Mapping (CAM)** and **Gradient-weighted CAM (Grad-CAM)** visualize **where** a convolutional network looks for a class decision. They apply to **CNNs** with convolutional feature maps before global pooling.

<div class="figure">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Convolutional_Neural_Network.png/960px-Convolutional_Neural_Network.png" alt="Convolutional neural network: conv layers, pooling, flatten, dense classifier" />
  <p class="caption"><strong>Figure 11.2.</strong> A CNN stacks convolution and pooling, then typically flattens and classifies. Grad-CAM uses gradients with respect to the **last convolutional feature maps** to build a heatmap over the input image. <a href="https://commons.wikimedia.org/wiki/File:Convolutional_Neural_Network.png">“Convolutional Neural Network.png”</a> by Irisbox, <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.</p>
</div>

Let $A^k$ denote the $k$-th **feature map** at a chosen convolutional layer, and $y^c$ the **score** (logit or pre-softmax) for class $c$. **Grad-CAM** uses gradients of $y^c$ with respect to spatial entries of $A^k$. With global average pooling weights:

$$
\alpha_k^c = \frac{1}{Z} \sum_{i,j} \frac{\partial y^c}{\partial A^k_{i,j}},
$$

the class-discriminative map (before upsampling to input size) is:

$$
L_{\text{Grad-CAM}}^c = \operatorname{ReLU}\!\left( \sum_k \alpha_k^c \, A^k \right).
$$

**ReLU** suppresses negative contributions that would obscure localization.

### 11.5.1 Improved variants (brief)

- **Grad-CAM++**: revised weighting of gradients to improve **localization** when multiple instances of a class appear.
- **Score-CAM**: avoids gradients; uses **forward activation** scores to weight channels (can reduce noise in some settings).
- **EigenCAM / Layer-CAM**: alternative weighting schemes or use of multiple layers for finer maps.

These methods are **class-discriminative** but not guaranteed causal; combine with **randomization tests** or **occlusion** checks.

---

## 11.6 Other Explainability Tools (Overview)

| **Method** | **Idea** |
| --- | --- |
| **Integrated Gradients** | Integrate gradients along a path from a baseline to $x$; satisfies some desirable axioms for attribution. |
| **SmoothGrad** | Average gradients of many noisy versions of $x$ to reduce visual noise. |
| **Occlusion / ablation** | Mask regions or features and measure $\Delta$ prediction (simple but expensive). |
| **Attention weights** (Transformers) | Inspect attention maps; useful intuition, not always faithful explanations (see Chapter 10). |

---

## 11.7 Gradient Descent: A Two-Parameter Picture

Training minimizes a **loss** $L(w)$ over parameters $w$. For **two** scalar weights $(w_1, w_2)$, think of $L(w_1, w_2)$ as height of a **surface** over the plane: **low** loss is a **valley**, **high** loss is a **hill**.

The **gradient** at a point is:

$$
\nabla L(w_1, w_2) = \begin{pmatrix} \dfrac{\partial L}{\partial w_1} \\ \dfrac{\partial L}{\partial w_2} \end{pmatrix}.
$$

The gradient points in the direction of **steepest increase** of $L$ (uphill on the error surface). To **minimize** loss, **gradient descent** moves **opposite** to the gradient:

$$
\begin{pmatrix} w_1 \\ w_2 \end{pmatrix}_{t+1}
=
\begin{pmatrix} w_1 \\ w_2 \end{pmatrix}_t
- \eta \, \nabla L(w_1, w_2)\Big|_{(w_1,w_2)_t},
$$

where $\eta > 0$ is the **learning rate**.

If you **add** the gradient instead (or use a negative learning rate by mistake), you perform **gradient ascent** on $L$: you move **toward higher** loss, **increasing error** when your goal is minimization.

### 11.7.1 Concrete quadratic example

Consider a simple **bowl-shaped** loss (convex):

$$
L(w_1, w_2) = w_1^2 + w_2^2.
$$

Then:

$$
\nabla L(w_1, w_2) = \begin{pmatrix} 2w_1 \\ 2w_2 \end{pmatrix}.
$$

At $(w_1, w_2) = (1, 1)$, the loss is $L = 2$, and $\nabla L = (2, 2)^T$. One step of gradient **descent** with $\eta = 0.25$ gives:

$$
(w_1, w_2)_{\text{new}} = (1, 1)^T - 0.25 \cdot (2, 2)^T = (0.5, 0.5)^T,
$$

and $L_{\text{new}} = 0.5$, **smaller** than before. Repeated steps approach $(0, 0)$, the **global minimum**.

If you **maximize** $L$ by mistake (ascent), you would use $w \leftarrow w + \eta \nabla L$, moving toward larger $L$ and **away** from the minimum.

<div class="figure">
  <img src="https://upload.wikimedia.org/wikipedia/commons/6/68/Gradient_ascent_%28surface%29.png" alt="3D surface with gradient ascent path" />
  <p class="caption"><strong>Figure 11.3.</strong> A smooth surface in $(w_1, w_2)$ with a path following **gradient ascent** (moving uphill). For **training**, you typically **minimize** loss, so you follow **negative** the gradient (gradient descent). File: <a href="https://commons.wikimedia.org/wiki/File:Gradient_ascent_(surface).png">“Gradient ascent (surface).png”</a> by Joris Gillis, released into the <strong>public domain</strong>.</p>
</div>

<div class="figure">
  <img src="https://upload.wikimedia.org/wikipedia/commons/5/5b/Gradient_descent_method.png" alt="Contour plot with gradient descent path toward minimum" />
  <p class="caption"><strong>Figure 11.4.</strong> Gradient descent on a contour plot: steps move toward lower values of the objective. <a href="https://commons.wikimedia.org/wiki/File:Gradient_descent_method.png">“Gradient descent method.png”</a> by Роман Сузи, <a href="https://creativecommons.org/licenses/by-sa/3.0/">CC BY-SA 3.0</a> / GFDL (see file page).</p>
</div>

<div class="figure">
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/01/Optimizer_Animations.gif" alt="Animation of several optimizers on a two-parameter loss surface" />
  <p class="caption"><strong>Figure 11.5.</strong> Several optimizers on the same **two-parameter** loss $f(x,y)=x^2 - y^2 + xy$ (animation). Different update rules follow different paths; all are variations on using **gradients** (and sometimes momentum) to reduce loss. <a href="https://commons.wikimedia.org/wiki/File:Optimizer_Animations.gif">“Optimizer Animations.gif”</a> by VirtualVistas, <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>.</p>
</div>

**Note:** The surface in Figure 11.5 is **not** a simple convex bowl; it illustrates that real losses can be **non-convex**, and optimizer choice affects whether you find a good basin.

---

## 11.8 Learning Rate

The learning rate $\eta$ controls step size.

- **Too small**: many iterations needed; training may stall on a plateau.
- **Too large**: updates **overshoot**, loss can **oscillate** or **diverge** (values of $L$ explode or NaNs appear).
- **Schedules**: $\eta$ is often **decayed** over epochs, or set by adaptive methods (Adam, RMSprop) that maintain per-parameter scaling.

A common **first-order** mental model: at each step, you approximate $L$ by a **linear** function (first-order Taylor expansion); $\eta$ must be small enough that this approximation stays valid along the step.

---

## 11.9 Weight Initialization

Before training, weights cannot all be **zero** (symmetry: every neuron in a layer would stay identical). Typical practice:

- **Small random** values (for example Gaussian or uniform with variance tied to layer width).
- **Xavier / Glorot** initialization: variance scaled to keep activation variance stable across layers in **tanh** or **sigmoid** networks.
- **He** initialization: variant suited to **ReLU** activations (variance scaled with fan-in).

Poor initialization can lead to **vanishing** or **exploding** activations and gradients, especially in deep networks. **Batch normalization** and **residual connections** also stabilize training but do not replace sensible initialization.

---

## Summary

- **XAI** spans local surrogates (**LIME**), Shapley-style attributions (**SHAP** with Kernel/Tree/Deep variants), and **gradient-based** maps (**Grad-CAM** family, integrated gradients).
- Attributions depend on **backgrounds**, **locality**, and **model class**; validate with ablations and domain knowledge.
- **Gradient descent** minimizes loss by stepping **against** the gradient; **ascent** increases loss.
- A **two-parameter** quadratic illustrates how **error** decreases along descent and increases if you ascend.
- **Learning rate** trades speed vs. stability; **initialization** breaks symmetry and keeps signals stable early in training.

---

## Topics for the Next Class

- implement **Grad-CAM** or **integrated gradients** on a small CNN for microscopy or natural images,
- compare **LIME** vs **SHAP** stability on correlated gene features,
- experiment with **learning rate** and **initialization** and log **training curves**.

---

## Further Reading

- Lundberg & Lee, “A Unified Approach to Interpreting Model Predictions”, [NeurIPS 2017](https://arxiv.org/abs/1705.07874) (SHAP).
- Ribeiro, Singh, Guestrin, “Why Should I Trust You?”, [KDD 2016](https://arxiv.org/abs/1602.04938) (LIME).
- Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks”, [IJCV 2020](https://arxiv.org/abs/1610.02391).
- [Explainable artificial intelligence, Wikipedia](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence)
- [Gradient descent, Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent)
- [Xavier Glorot initialization, Wikipedia](https://en.wikipedia.org/wiki/Xavier_Glorot_initialization)
