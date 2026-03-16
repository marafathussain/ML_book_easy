# Chapter 6: From Linear Models to Multilayer Perceptrons (Deep Learning)

## Introduction

In Chapters 3, 4, and 5, you learned **conventional machine learning**: linear regression, logistic regression, decision trees, SVMs, clustering, and PCA. All of these either stay linear (e.g. a weighted sum of features) or use hand-designed structure (e.g. trees, kernels). In this chapter we build a **smooth bridge** from that world to **deep learning**: we start from the linear equation and the idea of a decision boundary, add non-linearity and the **perceptron**, then connect to **PCA** and show how stacking a linear projection with a non-linear activation leads to **hidden layers** and, with more than one such layer, to **deep neural networks**. By the end, you will see where the **multilayer perceptron (MLP)** comes from and how **backpropagation** trains it.

The path is: **linear regression (vector form)** → **binary classifier (line as boundary)** → **hint of non-linearity (logistic loss)** → **perceptron** → **multiclass (softmax)** → **PCA + classifier (still linear)** → **training together (still linear)** → **activation and hidden layer** → **deep network** → **backpropagation**. Each step reuses the previous one so the transition feels continuous.

---

## 6.1 Linear Regression: From 1D to Many Features and Vector Notation

You have already seen **linear regression** with one feature: predict $y$ from $x$ using $y = w_0 + w_1 x$. With **multiple features** $x_1, x_2, \ldots, x_n$, the natural extension is:

$$
y = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_n x_n.
$$

**Example (2D input):** If we have two features $x_1$ and $x_2$, then:

$$
y = w_0 + w_1 x_1 + w_2 x_2.
$$

Here $w_0$ is the **bias** (intercept), and $w_1$, $w_2$ are the **weights** for each feature. For many features, writing the sum is cumbersome. We switch to **vector notation**.

**Vector notation:** Pack the weights and the features into vectors. Define $\mathbf{w} = (w_0, w_1, \ldots, w_n)^\top$ and $\mathbf{x} = (1, x_1, x_2, \ldots, x_n)^\top$ (the leading $1$ is for the bias). Then:

$$
y = \mathbf{w}^\top \mathbf{x} = w_0 \cdot 1 + w_1 x_1 + w_2 x_2 + \cdots + w_n x_n.
$$

So **one linear equation** in vector form is $y = \mathbf{w}^\top \mathbf{x}$. For a whole dataset, we have one such equation per sample: $y_i = \mathbf{w}^\top \mathbf{x}_i$ for each $i$. This is the same linear model you saw in Chapter 3; we are just writing it in a form that will generalize to many outputs and, later, to layers.

---

## 6.2 Binary Classification: A Line as Decision Boundary

In **binary classification**, we have two classes, say C1 and C2. We can use the **same linear function** $y(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$ (including bias via a 1 in $\mathbf{x}$) and interpret it as a **score**. Define the **decision boundary** as the set of points where the score is zero:

$$
y(\mathbf{x}_i) = \mathbf{w}^\top \mathbf{x}_i = 0.
$$

That is a **line** in 2D (or a **hyperplane** in higher dimensions). We then assign:

- If $\mathbf{w}^\top \mathbf{x}_i \geq 0$, then $\mathbf{x}_i$ is classified as **C1**.
- If $\mathbf{w}^\top \mathbf{x}_i < 0$, then $\mathbf{x}_i$ is classified as **C2**.

So the **predicted label** can be written as:

$$
\hat{\ell}_i = \mathrm{sign}(y_i) = \mathrm{sign}(\mathbf{w}^\top \mathbf{x}_i),
$$

where we treat $y_i = \mathbf{w}^\top \mathbf{x}_i$ as the score for sample $i$. The **objective** in the simplest binary linear classifier is to choose $\mathbf{w}$ so that $\mathrm{sign}(\mathbf{w}^\top \mathbf{x}_i)$ matches the true label (e.g. +1 for C1, −1 for C2). This is the **linear binary classifier**: one linear function, one boundary, and predictions by the sign of that function.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter6/figure1.jpg" alt="Perceptron and axon-dendrite analogy" />
  <p class="caption"><strong>Figure 6.1.</strong> A simple linear binary classifier.</p>
</div>

---

## 6.3 A Hint of Non-Linearity: Logistic Regression and Negative Log Loss

The classifier above uses **sign** and gives a hard boundary. In **logistic regression**, we keep the same linear part $\mathbf{w}^\top \mathbf{x}$ but pass it through a **non-linear function** so that we get a **probability** instead of a raw score. The standard choice is the **sigmoid** (logistic function):

$$
\sigma(\mathbf{w}^\top \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^\top \mathbf{x}}}.
$$

So we still use $\mathbf{w}^\top \mathbf{x}$, but the output is squeezed into $(0, 1)$ and interpreted as the probability of class C1. The decision boundary is still where $\mathbf{w}^\top \mathbf{x} = 0$ (where $\sigma(0) = 0.5$), but now we can define a **loss** that is smooth and good for training.

If the true label is $y_i \in \{0, 1\}$ and the predicted probability is $p_i = \sigma(\mathbf{w}^\top \mathbf{x}_i)$, then the **negative log-likelihood** (cross-entropy) for one sample is:

$$
\mathcal{L}_i = - \bigl( y_i \log p_i + (1 - y_i) \log(1 - p_i) \bigr).
$$

Minimizing this over the dataset pushes $p_i$ toward 1 when $y_i = 1$ and toward 0 when $y_i = 0$. So we get a **smooth, differentiable objective** built from the same linear combination $\mathbf{w}^\top \mathbf{x}$. This is the first hint that we can add **non-linearity** (here, the sigmoid) on top of a linear map and train with gradient-based methods. The next step is to give that linear map a name and a picture: the **perceptron**.

---

## 6.4 The Perceptron: One Neuron

The **perceptron** is the simplest model of a **neuron**: it takes several inputs (features), forms their **linear combination** (with weights), and produces an output. In the binary case, that output is often passed through a **threshold** (e.g. sign) or a **sigmoid** (as in logistic regression). So the computation is still:

$$
\text{output} = \sigma(\mathbf{w}^\top \mathbf{x}) \quad \text{or} \quad \text{output} = \mathrm{sign}(\mathbf{w}^\top \mathbf{x}).
$$

Biologically, we can think of the inputs as signals arriving at **dendrites**, the weights as **synaptic strengths**, and the weighted sum as the **cell body** combining them; the output (after a threshold or activation) travels along the **axon**. The figure below is a placeholder for an illustration of this analogy (axon, dendrites, and the flow of signals).

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter6/figure2.jpg" alt="Perceptron and axon-dendrite analogy" />
  <p class="caption"><strong>Figure 6.2.</strong> Perceptron and axon-dendrite analogy. A single neuron: inputs (dendrites), weights (synapses), linear combination (cell body), and output (axon).</p>
</div>

So the **perceptron** is exactly the same as our linear model plus an activation (sign or sigmoid): one “neuron” that computes $\mathbf{w}^\top \mathbf{x}$ and then applies a non-linear function. The next step is to go from **one output** (binary) to **many outputs** (multiclass).

---

## 6.5 Multiclass Classification: Several Weights and Softmax

For **multiclass** classification with $C$ classes, we use **one weight vector per class**. Let $\mathbf{W}$ be the matrix whose rows are $\mathbf{w}_1^\top, \mathbf{w}_2^\top, \ldots, \mathbf{w}_C^\top$. The **score** for class $c$ for input $\mathbf{x}_i$ is:

$$
s_c(\mathbf{x}_i) = \mathbf{w}_c^\top \mathbf{x}_i.
$$

A natural rule is to assign the class with the **largest score**:

$$
\hat{\ell}_i = \arg\max_c \, \mathbf{w}_c^\top \mathbf{x}_i.
$$

To get **probabilities** (like logistic regression but for many classes), we use the **softmax** of these scores:

$$
p(y = c \mid \mathbf{x}_i) = \frac{e^{\mathbf{w}_c^\top \mathbf{x}_i}}{\sum_{k=1}^{C} e^{\mathbf{w}_k^\top \mathbf{x}_i}}.
$$

So the **likelihood** of the correct class $c_i$ for sample $i$ is $p(y = c_i \mid \mathbf{x}_i)$. Taking the **negative log** gives the **cross-entropy loss**:

$$
\mathcal{L}_i = -\log p(y = c_i \mid \mathbf{x}_i) = -\log \frac{e^{\mathbf{w}_{c_i}^\top \mathbf{x}_i}}{\sum_{k=1}^{C} e^{\mathbf{w}_k^\top \mathbf{x}_i}}.
$$

**Why this equals the logistic regression loss when $C = 2$**

For two classes we have two weight vectors $\mathbf{w}_0$ and $\mathbf{w}_1$. The softmax gives:

$$
p(y = 1 \mid \mathbf{x}_i) = \frac{e^{\mathbf{w}_1^\top \mathbf{x}_i}}{e^{\mathbf{w}_0^\top \mathbf{x}_i} + e^{\mathbf{w}_1^\top \mathbf{x}_i}}, \qquad p(y = 0 \mid \mathbf{x}_i) = \frac{e^{\mathbf{w}_0^\top \mathbf{x}_i}}{e^{\mathbf{w}_0^\top \mathbf{x}_i} + e^{\mathbf{w}_1^\top \mathbf{x}_i}}.
$$

Divide numerator and denominator of $p(y = 1 \mid \mathbf{x}_i)$ by $e^{\mathbf{w}_0^\top \mathbf{x}_i}$:

$$
p(y = 1 \mid \mathbf{x}_i) = \frac{e^{(\mathbf{w}_1 - \mathbf{w}_0)^\top \mathbf{x}_i}}{1 + e^{(\mathbf{w}_1 - \mathbf{w}_0)^\top \mathbf{x}_i}} = \frac{1}{1 + e^{-(\mathbf{w}_1 - \mathbf{w}_0)^\top \mathbf{x}_i}} = \sigma\bigl((\mathbf{w}_1 - \mathbf{w}_0)^\top \mathbf{x}_i\bigr).
$$

So with a single vector $\mathbf{w} = \mathbf{w}_1 - \mathbf{w}_0$, we get $p(y = 1 \mid \mathbf{x}_i) = \sigma(\mathbf{w}^\top \mathbf{x}_i)$, which is exactly the logistic regression model. For the loss, when the true label is $c_i \in \{0, 1\}$, the negative log of the correct class probability is $-\log p(y = c_i \mid \mathbf{x}_i)$. For $c_i = 1$ that is $-\log \sigma(\mathbf{w}^\top \mathbf{x}_i)$; for $c_i = 0$ it is $-\log(1 - \sigma(\mathbf{w}^\top \mathbf{x}_i))$. So we get:

$$
\mathcal{L}_i = -\bigl( c_i \log \sigma(\mathbf{w}^\top \mathbf{x}_i) + (1 - c_i) \log\bigl(1 - \sigma(\mathbf{w}^\top \mathbf{x}_i)\bigr) \bigr),
$$

which is the **binary cross-entropy (negative log-likelihood)** used in logistic regression. So **softmax + negative log** for $C = 2$ is the same as logistic regression loss; for $C > 2$ it is the natural multiclass extension, and we still only have linear combinations $\mathbf{w}_c^\top \mathbf{x}_i$ plus one non-linear function (softmax).

---

## 6.6 Connecting to PCA: Classification in Reduced Space

In **Chapter 5** we saw **PCA**: we project the data onto a lower-dimensional space. Let $\mathbf{Z}$ be the **reduced** representation of the data (e.g. the first few principal components). We can write the **reconstruction** of the original data (approximately) as $\mathbf{X} \approx \mathbf{Z} \mathbf{W}^\top$ (or in the form $\mathbf{X} = \mathbf{W}^\top \mathbf{Z}$ depending on how we define $\mathbf{W}$ and $\mathbf{Z}$). In other words, we have a **linear map** from the reduced space back to the feature space. Conversely, the data in reduced space is $\mathbf{Z} = \mathbf{X} \mathbf{W}$ (each row of $\mathbf{Z}$ is a sample in reduced dimensions).

We can do **classification in the reduced space**. For each sample we have a reduced vector $\mathbf{z}_i$. We use the same linear classifier as before, but on $\mathbf{z}_i$:

$$
y_i = \mathbf{w}^\top \mathbf{z}_i, \qquad \hat{\ell}_i = \mathrm{sign}(y_i) = \mathrm{sign}(\mathbf{w}^\top \mathbf{z}_i).
$$

So the flow is: **original features** $\mathbf{x}_i$ → **reduced features** $\mathbf{z}_i$ (via PCA or any linear projection) → **score** $y_i = \mathbf{w}^\top \mathbf{z}_i$ → **predicted label** $\hat{\ell}_i = \mathrm{sign}(y_i)$. The **decision boundary** is still linear, but now it lives in the space of $\mathbf{z}$, not $\mathbf{x}$.

---

## 6.7 Training Together: Still One Linear Model

We have two steps: $\mathbf{x} \to \mathbf{z}$ (reduction) and $\mathbf{z} \to y$ (classification). We could **train them separately**: first fit PCA (or fix $\mathbf{W}$), then fit $\mathbf{w}$ on $\mathbf{z}$. But we could also **train them together**. If $\mathbf{z}_i = \mathbf{W} \mathbf{x}_i$ (so $\mathbf{Z} = \mathbf{X} \mathbf{W}^\top$ in matrix form), then:

$$
y_i = \mathbf{w}^\top \mathbf{z}_i = \mathbf{w}^\top (\mathbf{W} \mathbf{x}_i) = (\mathbf{W}^\top \mathbf{w})^\top \mathbf{x}_i = \hat{\mathbf{w}}^\top \mathbf{x}_i,
$$

where $\hat{\mathbf{w}} = \mathbf{W}^\top \mathbf{w}$. So the **combined** map from $\mathbf{x}$ to $y$ is still **one linear function** $\hat{\mathbf{w}}^\top \mathbf{x}$. Training $\mathbf{W}$ and $\mathbf{w}$ together is equivalent to training a single linear model in the original space: we do not get any extra expressiveness. The figure below is a placeholder for a diagram that shows: (1) two stages trained separately (x → z → y), and (2) the equivalent single linear map (x → y).

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter6/figure3.jpg" alt="Two-stage linear model equivalent to one linear map" />
  <p class="caption"><strong>Figure 6.3.</strong> Two-stage linear model (x → z → y) and the fact that $\mathbf{Y} = \mathbf{w}^T$ $\mathbf{Z} = \mathbf{w}^T$ (\mathbf{W} \mathbf{X}) = (\mathbf{W}^T \mathbf{w})^T \mathbf{X} is still a single linear map in \mathbf{x}.</p>
</div>

To get something **more powerful** than a single linear model, we need a **non-linearity** between the two stages so that the overall function is no longer linear in $\mathbf{x}$. That is exactly what an **activation function** does.

---

## 6.8 Neurons Need an Activation: Introducing $h(\mathbf{z})$

So far we had $\mathbf{z} = \mathbf{W} \mathbf{x}$ and then $y = \mathbf{w}^\top \mathbf{z}$. The composition is linear. In a **neuron**, the output of the linear part is usually passed through a **non-linear activation** $h$:

$$
\text{neuron output} = h(\mathbf{w}^\top \mathbf{x}) \quad \text{or} \quad \text{hidden unit} = h(\mathbf{W} \mathbf{x}).
$$

So we replace the raw $\mathbf{z}$ by **activated** values $h(\mathbf{z})$ (element-wise: each component $z_j$ becomes $h(z_j)$). Common choices for $h$ are **ReLU** $h(z) = \max(0, z)$, **sigmoid** $h(z) = 1/(1+e^{-z})$, or **tanh**. Once we do this, the overall map from $\mathbf{x}$ to the next layer is **no longer linear** in $\mathbf{x}$, so we can represent more complicated boundaries and functions. So: **neurons (perceptrons) require an activation** $h$ to move beyond linearity.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter6/figure4.jpg" alt="activation" />
  <p class="caption"><strong>Figure 6.4.</strong> Introducing non-linear function as activation.</p>
</div>

---

## 6.9 Hidden Layers: From One Layer to Deep Neural Networks

We now combine the ideas: **linear map** $\mathbf{z} = \mathbf{W} \mathbf{x}$, then **activation** $\mathbf{a} = h(\mathbf{z})$ (applied to each component). The vector $\mathbf{a}$ is the output of a **hidden layer**: it is “hidden” because we do not observe it directly; we only observe the final output (e.g. class scores or regression value). If we then add another linear map on top, e.g. $y = \mathbf{w}^\top \mathbf{a}$, we get:

$$
y = \mathbf{w}^\top \mathbf{a} = \mathbf{w}^\top h(\mathbf{W} \mathbf{x}).
$$

This is a **one-hidden-layer** network: one layer of hidden units $\mathbf{a} = h(\mathbf{W} \mathbf{x})$, then one output $y = \mathbf{w}^\top \mathbf{a}$. If we stack **more** such layers (linear map → activation → linear map → activation → …), we get a **multilayer perceptron (MLP)**. When there is **more than one hidden layer**, we usually call it a **deep neural network** or **deep learning**. So:

- **One hidden layer:** $\mathbf{a} = h(\mathbf{W} \mathbf{x})$, $y = \mathbf{w}^\top \mathbf{a}$.
- **More than one hidden layer:** deep neural network / deep learning.

The depth (number of layers) and the width (number of units per layer) determine the capacity of the model. Training such a network requires a way to compute gradients with respect to all weights; that is **backpropagation**.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter6/figure5.jpg" alt="deep-learning" />
  <p class="caption"><strong>Figure 6.5.</strong> Deep Neural Network.</p>
</div>

---

## 6.10 Backpropagation for a Single Hidden Layer Network

We write the **forward pass** for one hidden layer and one output (e.g. regression or one class score). Let:

- $\mathbf{x}$ = input vector (including bias 1 if needed).
- $\mathbf{W}^{(1)}$ = weight matrix of the hidden layer (input → hidden).
- $\mathbf{z} = \mathbf{W}^{(1)} \mathbf{x}$ = pre-activation of the hidden layer.
- $\mathbf{a} = h(\mathbf{z})$ = hidden layer activations (element-wise $h$).
- $\mathbf{w}^{(2)}$ = weight vector from hidden to output.
- $y = \mathbf{w}^{(2)\top} \mathbf{a}$ = output (scalar).

We have a **loss** $\mathcal{L}$ (e.g. squared error $(y - t)^2$ or cross-entropy). **Backpropagation** computes the gradient of $\mathcal{L}$ with respect to every weight by the **chain rule**.

**Output layer:** The gradient of $\mathcal{L}$ with respect to $\mathbf{w}^{(2)}$ is:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}^{(2)}} = \frac{\partial \mathcal{L}}{\partial y} \cdot \mathbf{a}.
$$

So we need $\delta_{\mathrm{out}} = \frac{\partial \mathcal{L}}{\partial y}$ (e.g. for squared error $\mathcal{L} = \frac{1}{2}(y - t)^2$, we get $\frac{\partial \mathcal{L}}{\partial y} = (y - t)$). Then $\frac{\partial \mathcal{L}}{\partial \mathbf{w}^{(2)}} = \delta_{\mathrm{out}} \, \mathbf{a}$.

**Hidden layer:** The gradient of $\mathcal{L}$ with respect to $\mathbf{W}^{(1)}$ flows back through the output and through the activation. Let $\delta_{\mathrm{out}} = \frac{\partial \mathcal{L}}{\partial y}$. The gradient of $\mathcal{L}$ with respect to $\mathbf{a}$ is:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{a}} = \delta_{\mathrm{out}} \, \mathbf{w}^{(2)}.
$$

The gradient with respect to $\mathbf{z}$ uses the **derivative of the activation** $h'(z_j)$ (element-wise):

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}} \odot h'(\mathbf{z}),
$$

where $\odot$ is element-wise product. Then:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}} \cdot \mathbf{x}^\top.
$$

So we get **update rules** for $\mathbf{w}^{(2)}$ and $\mathbf{W}^{(1)}$ by descending along these gradients (e.g. with gradient descent or SGD). This is **backpropagation** for a single hidden layer. For more layers, the same idea repeats: the gradient at each layer is computed from the gradient at the next layer and the local Jacobian (activation derivative and weight matrix).

---

## Summary

- **Linear regression** with many features is $y = \mathbf{w}^\top \mathbf{x}$. Binary classification uses the **decision boundary** $\mathbf{w}^\top \mathbf{x} = 0$ and $\hat{\ell} = \mathrm{sign}(\mathbf{w}^\top \mathbf{x})$.
- **Logistic regression** adds a **sigmoid** on $\mathbf{w}^\top \mathbf{x}$ and trains with **negative log-likelihood**.
- The **perceptron** is one neuron: linear combination + activation (sign or sigmoid). It connects to the biological picture of axon and dendrites.
- **Multiclass** uses one weight vector per class, **argmax** for the predicted class, and **softmax** for probabilities; **negative log** of softmax gives the cross-entropy loss.
- **PCA** gives a reduced representation $\mathbf{z}$. Classification in reduced space is $\hat{\ell} = \mathrm{sign}(\mathbf{w}^\top \mathbf{z})$. Training $\mathbf{W}$ (reduction) and $\mathbf{w}$ (classifier) **together** still yields one linear model $\hat{\mathbf{w}}^\top \mathbf{x}$.
- Adding an **activation** $h(\mathbf{z})$ makes the combined map **non-linear**. A **hidden layer** is $\mathbf{a} = h(\mathbf{W} \mathbf{x})$; the output is $y = \mathbf{w}^\top \mathbf{a}$. **More than one hidden layer** gives a **deep neural network** (deep learning).
- **Backpropagation** computes gradients for all weights by the chain rule: output layer first, then hidden layer(s), using the derivative of the activation.

---

## Topics for the Next Class

In the next class we will study:

- **Convolutions, kernels, and feature maps** , how convolutional layers work.
- **CNN architectures** , typical designs for image (and other grid) data.
- **Transfer learning for biological images** , reusing pretrained networks.
- **Data augmentation** , increasing effective data size for better generalization.

---

## Further Reading

- [Neural Networks and Deep Learning (Nielsen)](http://neuralnetworksanddeeplearning.com/) — online book with step-by-step derivation of backpropagation.
- [Deep Learning (Goodfellow, Bengio, Courville)](https://www.deeplearningbook.org/) — Chapter 6 (Deep Feedforward Networks) and Chapter 8 (Optimization).
- [3Blue1Brown: Neural networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — visual introduction to the structure of neural nets.
- [PyTorch: Neural Networks](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html) — building and training MLPs in code.
- [Scikit-learn: MLPClassifier / MLPRegressor](https://scikit-learn.org/stable/modules/neural_networks_supervised.html) — simple MLP without writing backprop by hand.
