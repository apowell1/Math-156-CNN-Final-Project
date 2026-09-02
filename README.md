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

## Tech Stack & Dependencies

- **Language & Runtime**: Python 3 (designed for GPU Colab environments with CUDA support)
- **GPU Array Computing**:
  - `cupy` (CUDA-accelerated array computing and GPU memory management)
- **Dataset Loading & Utilities**:
  - `tensorflow.keras.datasets` (used solely to download raw MNIST benchmark arrays)
  - `numpy`, `tqdm`
- **Interpretability & Image Processing**:
  - `lime` (`lime_image`, `LimeImageExplainer`)
  - `scikit-image` (`quickshift`, `mark_boundaries`, `resize`)
  - `matplotlib` (visualizing LIME explanation masks and boundaries)

---

## Repository Structure

```
.
└── Math_156_project.ipynb   # Main Jupyter notebook containing layer implementations, training, and LIME evaluation
```

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

## Key Mathematical & Algorithmic Modules

### 1. `im2col` & `col2im` (Convolution via Matrix Multiplication)
Transforms 4D input tensors $(N, C, H, W)$ into 2D matrices $(N \cdot H_{out} \cdot W_{out}, C \cdot K_h \cdot K_w)$. Forward pass computes:
\[
\text{Out} = \text{Cols} \cdot W^T + b
\]
Backward pass accumulates parameter gradients \(\nabla W = \nabla \text{Out}^T \cdot \text{Cols}\) and maps input gradients back to spatial coordinates via `col2im`.

### 2. `StochasticPooling`
Instead of deterministic max or average pooling, creates a probability distribution across each $2 \times 2$ pooling window:
\[
p_i = \frac{|a_i|}{\sum_{j} |a_j| + \epsilon}
\]
Samples a single activation location per window during forward pass, and routes the full gradient back to the sampled coordinate during backpropagation.

### 3. `ResponseNormalization`
Normalizes each feature map across adjacent channel depths:
\[
y = \frac{x}{\left(k + \frac{\alpha}{2} \sum_{c} x_c^2\right)^\beta}
\]

### 4. `MiniBatchSGD`
Stochastic gradient descent optimizer updating trainable weights and biases across all composite submodules:
\[
W \leftarrow W - \eta \cdot \nabla W
\]

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
