# Small Language Model (SLM) From Scratch

A deep, modern, hardware-aligned Decoder-Only Small Language Model implemented in pure PyTorch.
Designed for college research and future scale-up on **NVIDIA Blackwell GPUs**.

---

## 🛠️ Software & Tools Recommended for Development

As a student with **zero budget**, here is the recommended setup:

1. **Code Editor / IDE:**
   - [VS Code (Visual Studio Code)](https://code.visualstudio.com/) — Free, industry standard.
   - Recommended VS Code Extensions: *Python (by Microsoft)* and *Pylance*.
2. **Python Environment:**
   - Python 3.10 or 3.11 installed on your system.
   - Standard built-in `venv` to keep dependencies isolated without extra software.
3. **Hardware / Testing Platforms:**
   - **Local Machine (Your Mac / PC):** For writing code, checking shapes, and running the unit tests using `ModelConfig.tiny()`.
   - **Google Colab (Free Tier):** For running on a free cloud GPU (T4 16GB) when testing CUDA kernels.
   - **Kaggle Notebooks (Free 30 hrs/week):** For dual-GPU testing (2x T4).

---

## 📂 Repository File Structure

| File | Purpose | Key Details |
| :--- | :--- | :--- |
| [`requirements.txt`](./requirements.txt) | Python dependencies | Minimal set: PyTorch, NumPy, TikToken, Safetensors, PyTest. |
| [`config.py`](./config.py) | Hyperparameter management | Stores dimensions. Aligned to multiples of 64/128 for Blackwell Tensor Cores. Includes `.tiny()` and `.slm_1b()` presets. |
| [`norm.py`](./norm.py) | Root Mean Square Normalization | RMSNorm layer. Omits mean subtraction to reduce GPU memory traffic. |
| [`rotary.py`](./rotary.py) | Rotary Positional Embeddings (RoPE) | Rotates Query and Key coordinates to encode relative token distance. |
| [`mlp.py`](./mlp.py) | SwiGLU Feed-Forward Network | Gated MLP projection (`gate_proj`, `up_proj`, `down_proj`) with SiLU activation. |
| [`attention.py`](./attention.py) | Grouped-Query Attention (GQA) | 4x KV compression, FlashAttention integration, and KV cache handling. |
| [`block.py`](./block.py) | Transformer Decoder Block | Pre-RMSNorm residual architecture combining Attention and SwiGLU. |
| [`model.py`](./model.py) | Full SLM Model | Full assembly: Token Embeddings $\rightarrow$ Stacks of Blocks $\rightarrow$ RMSNorm $\rightarrow$ LM Head. |
| [`generation.py`](./generation.py) | Auto-regressive Generation | Text generation with KV caching, greedy decoding, temperature, top-k, and top-p sampling. |
| [`tokenizer.py`](./tokenizer.py) | Tokenizer wrapper | Supports HuggingFace BPE with a zero-dependency character fallback. |
| [`checkpoint.py`](./checkpoint.py) | Checkpoint Manager | Saves and loads model state, optimizer state, step counter, and config dictionary. |
| [`dataset.py`](./dataset.py) | Pre-training Data Loader | In-memory dataset and zero-RAM memory-mapped binary loader (`np.memmap`). |
| [`test_model.py`](./test_model.py) | Verification Test Suite | 7 automated unit tests checking shapes, gradients, causal mask leakage, and KV cache equivalence. |

---

## 🚀 How to Run and Verify (Step-by-Step)

### Step 1: Set Up Python Virtual Environment
Open your terminal in this directory and run:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Run the Architectural Verification Test Suite
To verify that all 7 mathematical and architectural components are working correctly:

```bash
python test_model.py
```

Expected output:
```text
Running SLM Verification Test Suite...

✓ RMSNorm verification passed.
✓ RoPE verification passed.
✓ SwiGLU MLP verification passed.
✓ Grouped-Query Attention verification passed.
✓ Model Forward and Backward passed. (Total params: 1,602,816)
✓ Causal Masking Leakage Test passed: Future tokens cannot leak into past context.
✓ KV-Cache Equivalence verified: Fast generation is numerically identical.

🎉 ALL 7 ARCHITECTURAL TESTS PASSED SUCCESSFULLY!
```

---

## 🧠 Architectural Highlights for Blackwell GPUs

1. **Hardware Dimension Multiples:** All attention head dimensions, hidden dimensions, and MLP intermediate projections are multiples of **64 or 128**, ensuring 100% matrix tile utilization on Blackwell 2nd-Gen Tensor Cores.
2. **FlashAttention Ready:** Uses PyTorch's native `F.scaled_dot_product_attention`, which seamlessly binds to FlashAttention CUDA kernels when running on NVIDIA GPUs.
3. **KV Cache Compression (GQA):** 4x reduction in KV Cache memory consumption, drastically reducing HBM bandwidth pressure during inference.
4. **Pre-RMSNorm:** Prevents vanishing/exploding gradients during deep multi-layer backpropagation.
