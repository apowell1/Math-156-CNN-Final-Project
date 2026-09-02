# Math 156: Custom Convolutional Neural Network from Scratch with GPU Acceleration and LIME Explainability

A complete, framework-free implementation of a Convolutional Neural Network (CNN) built entirely from scratch with low-level matrix calculus, forward/backward backpropagation passes, and GPU acceleration via CuPy, featuring custom `im2col`/`col2im` operations, Response Normalization, Stochastic Pooling, and Local Interpretable Model-agnostic Explanations (LIME).

---

## Purpose & Overview

Deep learning frameworks (such as PyTorch and TensorFlow) abstract away tensor operations, memory layout transformations, gradient accumulation, and stochastic layer dynamics.

This project was developed for **UCLA Math 156 (Machine Learning and Data Science)** to construct and train a deep convolutional network from fundamental mathematical and numerical principles:
- **Modular Layer Architecture**: Object-oriented `Module` and `Composition` abstraction implementing explicit analytical forward and backward passes.
- **Vectorized GPU Convolutions (`im2col` & `col2im`)**: Reformulates 4D spatial tensor convolutions into 2D General Matrix Multiply (GEMM) operations executed on CUDA via CuPy.
- **Advanced Regularization & Activation Layers**: Implements He (Kaiming) weight initialization, Local Response Normalization, and Stochastic Pooling (sampling activations according to multinomial distributions proportional to feature intensities).
- **Numerically Stable Loss**: Combined Softmax and Categorical Cross-Entropy loss with automatic log-sum-exp stabilization.
- **Model Explainability**: Interprets model predictions on MNIST digits using LIME (Local Interpretable Model-agnostic Explanations) and Quickshift superpixel segmentations.

---

## Mathematical Foundations & Tensor Calculus

The implementation derives exact analytical gradients across all tensor layers without automated differentiation frameworks:

### 1. Vectorized Convolution Gradient Formulation (`im2col` / `col2im`)
Let \(\mathbf{X} \in \mathbb{R}^{B \times C_{\text{in}} \times H \times W}\) be the input tensor and \(\mathbf{W} \in \mathbb{R}^{C_{\text{out}} \times (C_{\text{in}} K_h K_w)}\) the filter weight matrix.
The spatial sliding window operation is linearized via the `im2col` operator \(\Phi: \mathbb{R}^{B \times C_{\text{in}} \times H \times W} \to \mathbb{R}^{(B H_{\text{out}} W_{\text{out}}) \times (C_{\text{in}} K_h K_w)}\).
- **Forward GEMM**:
  \[
  \mathbf{Y}_{\text{flat}} = \Phi(\mathbf{X}) \mathbf{W}^T + \mathbf{b}^T
  \]
- **Backward Gradients via Matrix Adjoints**:
  Given upstream gradient \(\nabla_{\mathbf{Y}} \mathcal{L} \in \mathbb{R}^{(B H_{\text{out}} W_{\text{out}}) \times C_{\text{out}}}\):
  \[
  \nabla_{\mathbf{W}} \mathcal{L} = (\nabla_{\mathbf{Y}} \mathcal{L})^T \Phi(\mathbf{X}), \quad \nabla_{\mathbf{b}} \mathcal{L} = \sum_{i} (\nabla_{\mathbf{Y}} \mathcal{L})_{i, :}
  \]
  \[
  \nabla_{\mathbf{X}} \mathcal{L} = \Phi^*\left(\nabla_{\mathbf{Y}} \mathcal{L} \cdot \mathbf{W}\right) = \text{col2im}\left(\nabla_{\mathbf{Y}} \mathcal{L} \cdot \mathbf{W}\right)
  \]
  where \(\Phi^*\) is the adjoint un-rolling operator accumulating overlapping window gradients.

### 2. He (Kaiming) Variance-Preserving Initialization
To prevent exponential gradient vanishing or explosion across deep ReLU networks, initial weights are sampled from a zero-mean normal distribution with variance calibrated to the layer fan-in:
\[
W_{ij} \sim \mathcal{N}\left(0, \frac{2}{C_{\text{in}} \cdot K_h \cdot K_w}\right)
\]
justified by maintaining \(\text{Var}(y_l) = \text{Var}(y_{l-1})\) across rectifier units \(\mathbb{E}[\max(0, x)^2] = \frac{1}{2} \text{Var}(x)\).

### 3. Local Response Normalization (LRN) Calculus
Given activation map \(x_{b, c, h, w}\), the normalized output is:
\[
y_{b, c, h, w} = \frac{x_{b, c, h, w}}{\mathcal{S}_{b, h, w}^\beta}, \quad \text{where } \mathcal{S}_{b, h, w} = k + \frac{\alpha}{2} \sum_{c'} x_{b, c', h, w}^2
\]
Applying the multivariate chain rule yields the exact analytical backward pass:
\[
\frac{\partial \mathcal{L}}{\partial x_{b, c, h, w}} = \frac{1}{\mathcal{S}_{b, h, w}^\beta} \frac{\partial \mathcal{L}}{\partial y_{b, c, h, w}} - \frac{\alpha \beta x_{b, c, h, w}}{\mathcal{S}_{b, h, w}^{\beta + 1}} \sum_{c'} x_{b, c', h, w} \frac{\partial \mathcal{L}}{\partial y_{b, c', h, w}}
\]

### 4. Stochastic Multinomial Pooling Dynamics
Unlike deterministic maximum pooling \(\max_{i \in R} a_i\), Stochastic Pooling converts activations in pooling region \(R\) into a multinomial probability simplex:
\[
p_i = \frac{|a_i|}{\sum_{j \in R} |a_j| + \epsilon}, \quad i \in R
\]
A location \(k \sim \text{Multinomial}(p_1, \dots, p_{|R|})\) is sampled during forward evaluation. The backward pass routes the full subgradient exclusively through the active sampled index:
\[
\frac{\partial \mathcal{L}}{\partial a_i} = \frac{\partial \mathcal{L}}{\partial y} \cdot \mathbf{1}_{\{i = k\}}
\]
acting as a continuous spatial dropout regularizer that prevents feature co-adaptation.

### 5. Combined Softmax Cross-Entropy Loss & Jacobian Cancellation
For logits \(\mathbf{z} \in \mathbb{R}^K\) and one-hot target \(\mathbf{y} \in \{0, 1\}^K\), the loss is:
\[
\mathcal{L} = -\sum_{k=1}^K y_k \ln \left( \frac{\exp(z_k - \max_j z_j)}{\sum_j \exp(z_j - \max_j z_j)} \right)
\]
Direct calculation of the loss gradient w.r.t logits reveals the Jacobian cancellation:
\[
\nabla_{\mathbf{z}} \mathcal{L} = \mathbf{p} - \mathbf{y}
\]
avoiding separate computation of the \(K \times K\) Softmax Jacobian matrix \(\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)\).

---

## Network Architecture & Layer Specifications

The benchmark architecture constructed in `create_cnn_model()` processes $28 \times 28$ grayscale MNIST digits:

| Layer | Type | Configuration / Dimensions | Output Shape |
|---|---|---|---|
| **Conv1** | `Convolution` | 16 filters, $4 \times 4$ kernel, stride=1, padding=0 | $(B, 16, 25, 25)$ |
| **ReLU1** | `ReLU` | Element-wise max activation \(\max(0, x)\) | $(B, 16, 25, 25)$ |
| **Conv2** | `Convolution` | 32 filters, $2 \times 2$ kernel, stride=1, padding=0 | $(B, 32, 24, 24)$ |
| **Norm1** | `ResponseNormalization` | \(\alpha=0.05, \beta=0.5, k=1.0\) across channels | $(B, 32, 24, 24)$ |
| **Pool1** | `StochasticPooling` | Stochastic multinomial sampling, $2 \times 2$ pool size | $(B, 32, 12, 12)$ |
| **Pool2** | `StochasticPooling` | Stochastic multinomial sampling, $2 \times 2$ pool size | $(B, 32, 6, 6)$ |
| **Conv3** | `Convolution` | 64 filters, $3 \times 3$ kernel, stride=1, padding=0 | $(B, 64, 4, 4)$ |
| **Flatten1** | `Flatten` | Flattens spatial features $(64 \times 4 \times 4 \to 1024)$ | $(B, 1024)$ |
| **FC1** | `Linear` | Fully connected projection $(1024 \to 10)$ | $(B, 10)$ |
| **Loss** | `SoftmaxCrossEntropy` | Softmax probability transformation + Cross-Entropy | Scalar |

---

## Setup & Running the Notebook

### 1. Requirements

Running with GPU acceleration requires an NVIDIA GPU and matching CUDA libraries (Google Colab with T4 GPU runtime recommended).

```bash
# Install required dependencies
pip install cupy-cuda12x lime scikit-image matplotlib tqdm
# Or CPU-only development
pip install lime scikit-image matplotlib numpy tqdm
```

### 2. Execution

Open `Math_156_project.ipynb` in Google Colab or local JupyterLab, and execute all cells:
- Downloads and normalizes MNIST dataset to $[0, 1]$.
- Trains the CNN across hyperparameter configurations (`lr`, `batch_size`, `num_epochs`).
- Evaluates test accuracy and renders LIME visual explanation boundaries highlighting superpixel digit features.

---

## Limitations & Project Status

- Developed for CUDA-enabled Python environments using `cupy` for GPU tensor math; running on CPU requires replacing `cupy` imports with `numpy`.
- Dataset loader uses `tensorflow.keras.datasets.mnist` purely as an array fetcher for the benchmark dataset.
