# 🧠 Small Language Model (SLM) from Scratch

An end-to-end, PyTorch implementation of a decoder-only Transformer Small Language Model (SLM) built entirely from scratch. This project demystifies how modern generative language models work under the hood spanning raw text tokenization, memory-mapped binary datasets, causal multi-head self-attention, pre-norm Transformer blocks, mixed-precision training, and autoregressive decoding.

---

## 📖 Table of Contents
1. [Why Build a Small Language Model?](#-why-build-a-small-language-model)
2. [End-to-End Architecture Overview](#-end-to-end-architecture-overview)
3. [Core Concepts & Theoretical Workings](#-core-concepts--theoretical-workings)
   - [1. Subword Tokenization (BPE)](#1-subword-tokenization-byte-pair-encoding)
   - [2. Disk-Backed Memory Mapping (`np.memmap`)](#2-disk-backed-memory-mapping-npmemmap)
   - [3. Token & Positional Embeddings](#3-token--positional-embeddings)
   - [4. Causal Multi-Head Self-Attention](#4-causal-multi-head-self-attention)
   - [5. Feed-Forward Network & Residual Pre-Norms](#5-feed-forward-network--residual-pre-norms)
   - [6. Weight Tying](#6-weight-tying)
4. [Training Pipeline & Optimization](#-training-pipeline--optimization)
5. [Autoregressive Inference & Sampling](#-autoregressive-inference--sampling)
6. [Model Specifications](#-model-specifications)
7. [Getting Started & Usage](#-getting-started--usage)

---

## 🎯 Why Build a Small Language Model?

While Large Language Models (LLMs) with hundreds of billions of parameters dominate headlines, **Small Language Models (SLMs)** (typically ranging from 10M to 500M parameters) prove that massive scale is not required to learn grammar, reasoning patterns, and coherent storytelling especially when trained on focused, high-quality datasets like **TinyStories**.

Building an SLM from scratch provides:
* **Full Transparency:** Every tensor transformation from token IDs to output logits is explicit and inspectable.
* **Single-GPU Accessibility:** Trains in hours on consumer hardware (or free cloud GPUs like Google Colab T4) using mixed precision and gradient accumulation.
* **Architectural Foundations:** Implements the exact same mathematical building blocks used in GPT-2, GPT-3, and LLaMA.

---

## 🏗️ End-to-End Architecture Overview

The model follows a **Decoder-Only Transformer** architecture trained on the **Next-Token Prediction** (causal language modeling) objective:

```text
Input Text ("Once upon a time")
       │
       ▼
[ BPE Tokenizer ] ──► Token IDs: [7454, 2402, 257, 640]
       │
       ▼
[ Token Embedding (W_te) ] + [ Positional Encoding (W_pe / RoPE) ]
       │
       ▼
[ Dropout ]
       │
       ▼
┌──────────────────────────────────────────────┐
│            Transformer Block (× N)           │
│                                              │
│  ┌──► [ LayerNorm / RMSNorm ]                │
│  │           │                               │
│  │    [ Causal Multi-Head Self-Attention ]   │
│  │           │                               │
│  └─── ( + ) ◄┘  (Residual Connection 1)      │
│         │                                    │
│  ┌──► [ LayerNorm / RMSNorm ]                │
│  │           │                               │
│  │    [ Feed-Forward Network (MLP / GELU) ]  │
│  │           │                               │
│  └─── ( + ) ◄┘  (Residual Connection 2)      │
└──────────────────────────────────────────────┘
       │
       ▼
[ Final LayerNorm ]
       │
       ▼
[ Linear Language Modeling Head (Weight-Tied with W_te) ]
       │
       ▼
Output Logits (Batch_Size × Context_Length × Vocab_Size)
       │
       ▼
[ Softmax + Temperature / Top-k Sampling ] ──► Next Token
```

---

## 🔬 Core Concepts & Theoretical Workings

### 1. Subword Tokenization (Byte-Pair Encoding)
Neural networks cannot process raw strings directly. Instead of character-level tokenization (which produces overly long sequences with weak semantic density) or word-level tokenization (which suffers from massive vocabularies and out-of-vocabulary `UNK` errors), this project uses **Byte-Pair Encoding (BPE)** (e.g., via `tiktoken` GPT-2 tokenizer with a vocabulary of `50,257` tokens).
* Common words (`" the"`, `"once"`) remain single tokens.
* Rare or complex words (`"unbelievably"`) are decomposed into meaningful subword units (`"un"`, `"believ"`, `"ably"`).

### 2. Disk-Backed Memory Mapping (`np.memmap`)
Tokenizing millions of documents produces hundreds of millions of token IDs. Loading the entire tokenized corpus into system RAM at once causes Out-Of-Memory (OOM) crashes.
* During preprocessing, token IDs are serialized directly to disk as contiguous 16-bit unsigned integers (`uint16`, since vocab size `50,257 < 65,535`) inside `train.bin` and `validation.bin`.
* During training, `numpy.memmap` streams random slices of length `block_size` directly from disk into GPU memory on the fly with near-zero RAM overhead.

### 3. Token & Positional Embeddings
Each token ID is mapped to a dense continuous vector of dimension <i>d</i><sub>model</sub> (`n_embd`). Because self-attention treats input tokens as an unordered bag of vectors, **Positional Embeddings** are injected so the model understands word order:

<p align="center">
  <b><i>h</i><sub>0</sub> = <i>E</i><sub>token</sub>(<i>x</i>) + <i>E</i><sub>pos</sub>(0, 1, …, <i>T</i> − 1)</b>
</p>

### 4. Causal Multi-Head Self-Attention
Self-attention enables every token in a sequence to dynamically gather context from preceding tokens. Given an input representation <i>X</i> ∈ ℝ<sup><i>T</i> × <i>d</i><sub>model</sub></sup>, we project <i>X</i> into three distinct spaces **Queries** (<i>Q</i>), **Keys** (<i>K</i>), and **Values** (<i>V</i>) split across <i>H</i> independent attention heads where <i>d<sub>k</sub></i> = <i>d</i><sub>model</sub> / <i>H</i>:

<p align="center">
  <b>Attention(<i>Q</i>, <i>K</i>, <i>V</i>) = softmax( (<i>QK<sup>T</sup></i>) / √<i>d<sub>k</sub></i> + <i>M</i> ) <i>V</i></b>
</p>

* **Scaling Factor:** Why divide by √<i>d<sub>k</sub></i>? As <i>d<sub>k</sub></i> grows, the dot products <i>QK<sup>T</sup></i> grow large in magnitude, pushing the softmax function into regions with vanishingly small gradients. Scaling by 1 / √<i>d<sub>k</sub></i> keeps the variance at unit scale.
* **Causal Masking:** To prevent the model from looking at future tokens during training, an upper-triangular mask matrix <i>M</i> filled with −∞ is added prior to softmax. After exponentiation inside softmax, future positions become `0`.
* **Multi-Head Intuition:** Running <i>H</i> heads in parallel allows different heads to specialize in distinct linguistic relationships simultaneously (e.g., Head 1 tracks pronoun-noun references, Head 2 tracks verb tense, Head 3 tracks punctuation).

### 5. Feed-Forward Network & Residual Pre-Norms
While Self-Attention routes information *between* token positions, the **Position-wise Feed-Forward Network (FFN)** processes information *within* each token representation independently. It projects the embedding dimension up by a factor of 4 (`4 * n_embd`), applies a non-linear activation (**GELU** or **SwiGLU**), and projects back down to `n_embd`:

<p align="center">
  <b>FFN(<i>x</i>) = <i>W</i><sub>2</sub> · GELU(<i>W</i><sub>1</sub><i>x</i> + <i>b</i><sub>1</sub>) + <i>b</i><sub>2</sub></b>
</p>

**Pre-Layer Normalization & Residual Highways:**
Unlike the original 2017 Transformer which applied LayerNorm *after* each sub-layer (Post-Norm), modern SLMs apply LayerNorm *before* Attention and FFN (**Pre-Norm**), paired with additive residual skip connections:

<p align="center">
  <b><i>x</i><sub><i>l</i>+1</sub> = <i>x<sub>l</sub></i> + SubLayer(LayerNorm(<i>x<sub>l</sub></i>))</b>
</p>

This creates an unimpeded "gradient highway" from the loss function directly back to the earliest layers, preventing vanishing gradients and stabilizing training without complex warm-up tuning.

### 6. Weight Tying
The input Token Embedding matrix of shape (<i>V</i> × <i>d</i><sub>model</sub>) and the final Language Modeling output projection matrix of shape (<i>d</i><sub>model</sub> × <i>V</i>) share the exact same underlying weight tensor. Because semantically similar tokens should both have similar input representations and receive similar output probabilities, tying these weights cuts parameter overhead by 25–30% in small models while accelerating convergence.

---

## ⚙️ Training Pipeline & Optimization

Training a language model from scratch requires careful numerical and optimization engineering:

1. **Objective Function (Cross-Entropy Loss):**
   Given context tokens <i>x</i><sub>1:<i>t</i></sub>, the model outputs unnormalized logits <i>z<sub>t</sub></i> ∈ ℝ<sup><i>V</i></sup>. We minimize the negative log-likelihood of the true next token <i>x</i><sub><i>t</i>+1</sub> across all positions in the context window simultaneously:

<p align="center">
  <b>ℒ = − (1 / <i>T</i>) ∑<sub><i>t</i>=1..<i>T</i></sub> log <i>P</i>(<i>x</i><sub><i>t</i>+1</sub> | <i>x</i><sub>≤<i>t</i></sub>)</b>
</p>

2. **AdamW Optimizer with Decoupled Weight Decay:**
   Uses β<sub>1</sub> = `0.9` and β<sub>2</sub> = `0.95`, applying weight decay (`0.1`) only to 2D weight matrices (linear projections and embeddings) while excluding 1D biases and LayerNorm scales.
3. **Learning Rate Schedule (Linear Warmup + Cosine Decay):**
   Starting with a high learning rate on randomly initialized weights causes immediate divergence. The learning rate warms up linearly over the first few thousand steps, then decays smoothly along a cosine curve down to `min_lr`.
4. **Gradient Accumulation:**
   Simulates large batch sizes (which stabilize gradient estimates) on memory-constrained GPUs by accumulating gradients over <i>K</i> micro-steps before calling `optimizer.step()`.
5. **Automatic Mixed Precision (`bfloat16` / `float16`):**
   Executes matrix multiplications in 16-bit precision via `torch.amp.autocast` while maintaining master weights in `float32`, cutting VRAM usage nearly in half and doubling throughput via Tensor Cores.
6. **Gradient Clipping (`max_norm = 1.0`):**
   Clips the global L2 norm of gradients prior to weight updates to prevent sudden loss spikes from anomalous mini-batches.

---

## 💬 Autoregressive Inference & Sampling

During generation, the model generates text one token at a time in an autoregressive loop:

1. Feed the prompt tokens into the model and extract the logits from the **final time step** only.
2. **Temperature Scaling:** Divide logits by temperature τ before softmax (<i>z<sub>i</sub></i> / τ):
   * τ < `1.0`: Sharpens the distribution, making high-probability tokens more dominant (more deterministic and focused).
   * τ > `1.0`: Flattens the distribution, increasing creativity and diversity.
3. **Top-<i>k</i> / Top-<i>p</i> (Nucleus) Filtering:** Zeroes out the tail of low-probability tokens so the model never samples incoherent words when creativity is high.
4. Sample the next token via `torch.multinomial`, append it to the running sequence, and repeat until `max_new_tokens` is reached.

---

## 📊 Model Specifications

*(Update the values below to match your exact `GPTConfig` in the repo)*

| Hyperparameter | Config Variable | Value | Description |
| :--- | :--- | :--- | :--- |
| **Vocabulary Size** | `vocab_size` | `50,257` | GPT-2 BPE subword vocabulary (`tiktoken`) |
| **Context Window** | `block_size` | `128` – `512` | Maximum sequence length processed per pass |
| **Transformer Layers** | `n_layer` | `6` | Number of stacked decoder blocks |
| **Attention Heads** | `n_head` | `6` | Parallel attention heads per block |
| **Embedding Dimension** | `n_embd` | `384` | Hidden state dimensionality (<i>d</i><sub>model</sub>) |
| **Dropout Rate** | `dropout` | `0.1` | Regularization rate on attention and MLP |
| **Total Parameters** | — | `~10M - 30M` | Compact footprint suitable for single-GPU training |

---

## 🚀 Getting Started & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/GauthamSU/Small-Language-Model-from-Scratch.git
cd Small-Language-Model-from-Scratch
```

### 2. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch numpy tiktoken datasets tqdm matplotlib
```

### 3. Run Training & Generation
* **Using Jupyter Notebook:** Open the `.ipynb` notebook in Jupyter Lab or Google Colab (select a **T4 GPU** runtime) and execute the cells sequentially to download the dataset, build `train.bin` / `validation.bin`, train the Transformer, inspect the training/validation loss curves, and sample generated stories.
* **Using Python Script (if modularized):**
  ```bash
  python train.py
  python generate.py --prompt "Once upon a time, there was a little robot" --max_new_tokens 200
  ```

---

## 📚 References & Acknowledgements
* Vaswani et al. (2017) — [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
* Radford et al. (2019) — [*Language Models are Unsupervised Multitask Learners (GPT-2)*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* Eldan & Li (2023) — [*TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*](https://arxiv.org/abs/2305.07759)
* Andrej Karpathy — [*nanoGPT*](https://github.com/karpathy/nanoGPT)
