# Chapter 7: Convolutions, CNNs, Transfer Learning, and Data Augmentation

## Introduction

In Chapter 6 we built from linear models to **multilayer perceptrons (MLPs)**. MLPs treat each input as a flat vector: they do not explicitly use the **grid structure** of images (rows and columns, nearby pixels). For images, time series, or any data that lives on a **grid**, **convolutional neural networks (CNNs)** are the standard tool. They use **convolutions**: a small **kernel** (filter) slides over the input and produces a **feature map**. This chapter explains **discrete 1D and 2D convolution** with concrete examples, then **kernels and feature maps**, **typical CNN architectures**, **transfer learning** for biological images, and **data augmentation**.

---

## 7.1 Discrete 1D Convolution (Step by Step)

**Convolution** combines two sequences by sliding one (the **kernel**) over the other (the **signal**) and computing, at each position, a weighted sum of overlaps.

**Notation:** Let the **signal** be a 1D array of length $n$, e.g. $\mathbf{x} = [x_0, x_1, x_2, x_3, x_4]$. Let the **kernel** (filter) be a shorter array of length $k$, e.g. $\mathbf{w} = [w_0, w_1, w_2]$. The **output** at position $i$ is the sum of products where the kernel is aligned starting at $i$.

**Formula (discrete 1D convolution):**

$$
(y * w)[i] = \sum_{j=0}^{k-1} x_{i+j} \, w_j
$$

So at each output index $i$, we take a window of the signal $[x_i, x_{i+1}, \ldots, x_{i+k-1}]$, multiply it element-wise with the kernel $[w_0, w_1, \ldots, w_{k-1}]$, and sum. The output length depends on whether we **pad** the signal and how we handle the edges (see below).

---

### Example: 1D convolution with a tiny signal and kernel

**Signal** (length 5): $\mathbf{x} = [1, 2, 3, 4, 5]$

**Kernel** (length 3): $\mathbf{w} = [1, 0, -1]$

We will compute the output for every position where the kernel fully overlaps the signal (no padding). So the kernel “starts” at index 0, 1, 2 of the signal. That gives 3 output values.

**Step 1 — kernel at position 0:**  
Window = $[x_0, x_1, x_2] = [1, 2, 3]$, kernel = $[1, 0, -1]$

$$
y[0] = 1\cdot1 + 2\cdot0 + 3\cdot(-1) = 1 + 0 - 3 = -2
$$

**Step 2 — kernel at position 1:**  
Window = $[x_1, x_2, x_3] = [2, 3, 4]$, kernel = $[1, 0, -1]$

$$
y[1] = 2\cdot1 + 3\cdot0 + 4\cdot(-1) = 2 + 0 - 4 = -2
$$

**Step 3 — kernel at position 2:**  
Window = $[x_2, x_3, x_4] = [3, 4, 5]$, kernel = $[1, 0, -1]$

$$
y[2] = 3\cdot1 + 4\cdot0 + 5\cdot(-1) = 3 + 0 - 5 = -2
$$

**Result:** $\mathbf{y} = [-2, -2, -2]$. So the output length is $n - k + 1 = 5 - 3 + 1 = 3$.

This kernel $[1, 0, -1]$ is a simple **finite-difference** filter: it approximates the derivative (difference between “next” and “previous” sample). So 1D convolution can detect **edges** or **changes** in a 1D signal (e.g. a time series or a single row of an image).

The figure below illustrates 1D convolution: a signal, a kernel, and the sliding window producing one output value.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter7/1D_convolution.gif" alt="1D convolution: signal, kernel, and sliding window" />
  <p class="caption"><strong>Figure 7.1.</strong> Discrete 1D convolution. The kernel slides over the signal; at each position the overlapping elements are multiplied and summed to give one output value.</p>
</div>

**Padding:** If we **zero-pad** the signal (e.g. add zeros at the ends) we can get an output that has the same length as the input, or control the output size. For example, with one zero on each side, $\mathbf{x}_{\text{padded}} = [0, 1, 2, 3, 4, 5, 0]$, we can compute more output positions. **Stride** is the step by which we move the kernel (stride 1 = move one position each time; stride 2 = skip one position).

The figure below illustrates 1D convolution with **zero-pad**.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter7/1D_convolution_w_padding.gif" alt="1D convolution: signal, kernel, and sliding window" />
  <p class="caption"><strong>Figure 7.2.</strong> Discrete 1D convolution with zero padding.</p>
</div>


---

## 7.2 Discrete 2D Convolution (Step by Step)

In 2D we have a **grid** (e.g. an image) and a **2D kernel**. The kernel slides in both rows and columns; at each position we take the element-wise product of the kernel with the patch of the image and sum.

**Notation:** Let the **input** be a 2D array (matrix) of size $H \times W$. Let the **kernel** be $K_h \times K_w$ (e.g. $3 \times 3$). The **output** at position $(i, j)$ is:

$$
(y * w)[i,j] = \sum_{a=0}^{K_h-1} \sum_{b=0}^{K_w-1} x[i+a,\, j+b] \; w[a,b]
$$

So we take a $K_h \times K_w$ patch of the input starting at $(i, j)$, multiply it element-wise with the kernel, and sum. The output size is $(H - K_h + 1) \times (W - K_w + 1)$ if we do not pad.

---

### Example: 2D convolution with a small “image” and kernel

**Input** (e.g. a $4 \times 4$ patch of an image):

$$
\mathbf{X} = \begin{bmatrix}
1 & 2 & 1 & 0 \\
0 & 1 & 2 & 1 \\
1 & 0 & 1 & 2 \\
2 & 1 & 0 & 1
\end{bmatrix}
$$

**Kernel** ($3 \times 3$, a simple vertical-edge detector: negative left column, positive right column):

$$
\mathbf{W} = \begin{bmatrix}
-1 & 0 & 1 \\
-1 & 0 & 1 \\
-1 & 0 & 1
\end{bmatrix}
$$

**Step 1 — kernel at top-left (0,0):**  
Patch of $\mathbf{X}$ at (0,0) of size $3\times3$:

$$
\begin{bmatrix} 1 & 2 & 1 \\ 0 & 1 & 2 \\ 1 & 0 & 1 \end{bmatrix}
$$

Element-wise with $\mathbf{W}$ and sum:

$$
y[0,0] = 1\cdot(-1)+2\cdot0+1\cdot1 + 0\cdot(-1)+1\cdot0+2\cdot1 + 1\cdot(-1)+0\cdot0+1\cdot1 = -1+0+1+0+0+2-1+0+1 = 2
$$

**Step 2 — kernel at (0,1):**  
Patch:

$$
\begin{bmatrix} 2 & 1 & 0 \\ 1 & 2 & 1 \\ 0 & 1 & 2 \end{bmatrix}
$$

$$
y[0,1] = 2\cdot(-1)+1\cdot0+0\cdot1 + 1\cdot(-1)+2\cdot0+1\cdot1 + 0\cdot(-1)+1\cdot0+2\cdot1 = -2+0+0-1+0+1+0+0+2 = 0
$$

We continue in the same way for positions $(1,0)$, $(1,1)$. The **output** is a $2 \times 2$ map (because $4-3+1 = 2$ in each dimension):

$$
\mathbf{Y} = \begin{bmatrix} y[0,0] & y[0,1] \\ y[1,0] & y[1,1] \end{bmatrix}
$$

Each entry of $\mathbf{Y}$ is the response of the kernel at one position. So **one kernel** produces **one 2D feature map**. Different kernels (e.g. horizontal edge, blur, sharpen) produce different feature maps. In a CNN, these kernels are **learned**; the network learns which local patterns (edges, textures) to detect.

The figure below illustrates 2D convolution: an input patch, a kernel, and the resulting feature map.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter7/conv2d_demo.png" alt="2D convolution: input patch, kernel, and feature map" />
  <p class="caption"><strong>Figure 7.3.</strong> Discrete 2D convolution. The kernel slides over the image (or feature map); at each position the overlapping patch is multiplied element-wise with the kernel and summed to give one value in the output feature map.</p>
</div>

---

### Padding and stride in 2D

- **Padding:** Add rows/columns of zeros around the input. For example, “same” padding is chosen so that the output has the same height and width as the input (for a $3\times3$ kernel, we often add 1 pixel on each side). Other than zeros, neighboring values can also be used for padding (see Figure 7.4).
- **Stride:** Step size when sliding. Stride 1 moves one pixel at a time; stride 2 halves the output size (approximately). So convolution can **downsample** the spatial dimensions.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter7/2D_convolution_w_padding.gif" alt="2D convolution: input patch, kernel, and feature map" />
  <p class="caption"><strong>Figure 7.4.</strong> Discrete 2D convolution with padding. Image source: https://commons.wikimedia.org/wiki/File:2D_Convolution_Animation.gif</p>
</div>

---

## 7.3 Kernels and Feature Maps

- **Kernel (filter):** The small matrix we slide over the input. In a CNN it is **learned** (parameters of the network). Hand-designed examples: $[1, 0, -1]$ (1D edge), $3\times3$ vertical or horizontal edge detectors, or a box filter for blur.
- **Feature map:** The **output** of one convolution. One kernel → one feature map. If we use 32 kernels, we get 32 feature maps (often thought of as 32 “channels”).
- **Stack of feature maps:** For a colour image we have 3 input channels (R, G, B). Each kernel is 3D (e.g. $3\times3\times3$): it has one $3\times3$ slice per input channel, and we sum over channels to get one number per position. So one kernel still gives **one** feature map; the number of **output** feature maps equals the number of kernels.

After convolution we often apply a **non-linearity** (e.g. ReLU) and sometimes **pooling** (e.g. max-pool: replace each $2\times2$ block by its maximum). That gives the next “layer” of representations.

---

## 7.4 CNN Architectures

A typical **CNN** for images has:

1. **Convolutional layers** — Convolve with learned kernels; add many feature maps (e.g. 32, 64, 128). After each conv we usually apply ReLU and sometimes batch normalization.
2. **Pooling layers** — Reduce spatial size (e.g. max-pool $2\times2$) to get translation invariance and smaller feature maps.
3. **More conv + pool blocks** — Stack several such blocks so that early layers detect edges/textures and deeper layers detect higher-level patterns (parts, objects).
4. **Flatten + fully connected** — At the end we flatten the feature maps and pass them through one or more fully connected (dense) layers for classification or regression.

<div class="figure">
  <img src="https://marafathussain.github.io/ML_book_easy/figures/chapter7/CNN_demo.jpg" alt="2D convolution: input patch, kernel, and feature map" />
  <p class="caption"><strong>Figure 7.5.</strong> An example of a convolutional neural network (CNN), designed for binary (i.e., 2-way) classification. </p>
</div>



**Examples of architectures:** LeNet, AlexNet, VGG, ResNet, EfficientNet. They differ in depth, kernel sizes, and use of skip connections. For **biological images** (microscopy, histology, radiology), the same idea applies: convolutions capture local structure; deeper layers capture more abstract features.

---

## 7.5 Transfer Learning for Biological Images

**Transfer learning** means reusing a network (or part of it) that was **pretrained** on a large dataset (e.g. ImageNet) and **adapting** it to your task (e.g. classifying cell types or tissue from a smaller dataset).

**Why it helps:** Biological image datasets are often small. Training a deep CNN from scratch needs many labelled images. A pretrained network has already learned useful low-level and mid-level features (edges, textures, shapes). We can:

- **Freeze** the early layers (keep their weights fixed) and only train the last few layers and the classifier head, or
- **Fine-tune** the whole network with a small learning rate so we do not destroy the pretrained features but adapt them to the new domain.

**Typical workflow:** Load a pretrained model (e.g. ResNet from PyTorch or TensorFlow), replace the final classifier with one that has the right number of classes for your problem, then train on your biological images (optionally with data augmentation). This is standard in microscopy, histology, and medical imaging.

---

## 7.6 Data Augmentation

**Data augmentation** creates extra training examples by applying **label-preserving** transformations to the existing data. For images, common operations are:

- **Geometric:** Rotation, flip (horizontal/vertical), crop, zoom, shift.
- **Photometric:** Brightness/contrast change, slight blur, colour jitter.

**Why it helps:** The model sees more varied examples; it tends to generalize better and overfit less. For biological images, augmentations should stay **realistic** (e.g. small rotations and flips are usually safe; strong distortions might not correspond to real variation).

**In code:** Libraries like Keras (`ImageDataGenerator`), PyTorch (e.g. `torchvision.transforms`), or specialized packages for microscopy/histology let you define a pipeline of random augmentations applied on-the-fly during training.

---

## Summary

- **1D convolution:** Kernel slides over a 1D signal; at each position we compute the inner product of the kernel with the overlapping segment. Output length = signal length − kernel length + 1 (without padding).
- **2D convolution:** Kernel slides over a 2D grid (image or feature map); at each position we compute the sum of element-wise products over the overlapping patch. One kernel produces one **feature map**.
- **Kernels** in CNNs are **learned**; they act as local feature detectors. **Padding** and **stride** control output size.
- **CNN architectures** stack conv + activation + pooling, then flatten + dense layers for classification. Deeper layers capture more abstract features.
- **Transfer learning** reuses pretrained networks and fine-tunes on biological (or other) images to get good performance with limited data.
- **Data augmentation** increases effective dataset size with label-preserving transforms (rotation, flip, brightness, etc.) and improves generalization.

---

## Topics for the Next Class

- Recurrent and attention-based models for sequences (e.g. RNA, time series).
- Or further practice with CNNs and biological image analysis.

---

## Further Reading

- [A guide to convolution arithmetic (e.g. padding, stride)](https://github.com/vdumoulin/conv_arithmetic)
- [CS231n: Convolutional Neural Networks](http://cs231n.github.io/convolutional-networks/)
- [PyTorch: Training a classifier (with CNNs and transfer learning)](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [TensorFlow: Image classification and transfer learning](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Survey: Deep learning in microscopy and medical imaging](https://www.nature.com/articles/s41592-021-01205-4) (Nature Methods, 2021)
