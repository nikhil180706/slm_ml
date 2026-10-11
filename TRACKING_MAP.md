# 🗺️ SLM Project: Master Tracking Map & Knowledge Guide

> **Project Context:** College Partnership with **IAIRO**  
> **Target Scale:** Small Language Model (SLM) ~1B to 1.5B parameters  
> **Target Hardware:** NVIDIA Blackwell GPUs (B200 / GB200)  
> **Development Environment:** Local Mac (MPS / CPU) & Zero-Budget Cloud (Colab / Kaggle)

---

## 🧭 Master Project Timeline & Roadmap

### Step 1: Architecture & Model Design (✅ COMPLETED)
- `config.py`: Defined dimensions aligned to multiples of 64/128 for Blackwell Tensor Cores.
- `norm.py`: Custom RMSNorm omitting mean subtraction to save memory bandwidth.
- `rotary.py`: Rotary Positional Embeddings (RoPE) precomputing frequency tables and rotating 2D coordinate planes.
- `mlp.py`: SwiGLU Feed-Forward Network using three zero-bias linear projections.
- `attention.py`: Grouped-Query Attention (GQA) with 4x KV compression and FlashAttention integration.
- `block.py`: Pre-RMSNorm Transformer Decoder Block combining Attention and SwiGLU with residual streams.
- `model.py`: Full assembly of Token Embeddings -> Stack of Blocks -> RMSNorm -> LM Head.

### Step 2: Verification & Test Suite (✅ COMPLETED)
- `test_model.py`: Ran 7 unit tests covering tensor shapes, gradient flow, causal mask leakage, and KV-cache numerical equivalence. All passed.

### Step 3: Training Engine & Optimization (✅ COMPLETED)
- `train.py`: Built AdamW optimizer with weight decay splitting, Cosine LR scheduler with warmup, gradient clipping (1.0), and 100-step overfit verification.
- `checkpoint.py`: Saved trained state to `checkpoints/slm_step_100.pt`.

### Step 4: Interactive Text Generation (⏳ UP NEXT)
- Load saved checkpoint and generate text auto-regressively from custom prompts using KV caching.

### Step 5: Binary Pre-Training Data Pipeline (📅 FUTURE)
- Stream open datasets (FineWeb-Edu, TinyStories) and tokenize into memory-mapped `.bin` files (`dataset.py`).

### Step 6: Blackwell Supercomputing Training (📅 FUTURE)
- Multi-GPU distributed training (FSDP / Megatron-LM) with FP8 Transformer Engine on NVIDIA Blackwell hardware.

---

## 🔍 Exhaustive File-by-File Breakdown

### 1. `config.py` — The Central Blueprint
* **Easy Intuition:** Like the architectural blueprint of a building before buying materials.
* **Deep Mechanics:** Stores dataclass parameters: `dim`, `n_layers`, `n_heads`, `n_kv_heads`, `multiple_of=128`. Aligns dimensions to 128 so Blackwell Tensor Cores achieve peak matrix tile utilization.
* **Pitfalls to Avoid:** Never hard-code dimensions in other files; always import from `config.py`.

### 2. `norm.py` — RMSNorm
* **Easy Intuition:** A volume knob that keeps numbers stable between -1.0 and +1.0.
* **Deep Mechanics:** $\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \odot \gamma$. Skips mean calculation to reduce memory traffic by ~7%.
* **Pitfalls to Avoid:** Add `eps = 1e-5` to prevent division by zero. Compute in FP32 during mixed precision to prevent underflow.

### 3. `rotary.py` — Rotary Positional Embeddings (RoPE)
* **Easy Intuition:** A clock that rotates word vectors in 2D space based on their position in the sentence.
* **Deep Mechanics:** Rotates Query and Key coordinates. The dot product between two tokens depends purely on $(m - n)$, capturing relative distance rather than absolute position.
* **Pitfalls to Avoid:** Apply RoPE ONLY to Query and Key; NEVER apply to Value. Head dimension must be even.

### 4. `mlp.py` — SwiGLU Feed-Forward Network
* **Easy Intuition:** The individual thinking brain of each token.
* **Deep Mechanics:** $\text{SwiGLU}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})) W_{\text{down}}$. Gated non-linear activation with zero bias.
* **Pitfalls to Avoid:** intermediate dimension should be ~$(8/3) \times \text{dim}$ rounded to a multiple of 128.

### 5. `attention.py` — Grouped-Query Attention (GQA) & KV Cache
* **Easy Intuition:** Allows words to look at and communicate with other words.
* **Deep Mechanics:** Shares 1 Key/Value head group across 4 Query heads (4x memory reduction). Uses FlashAttention for fast memory-efficient calculation.
* **Pitfalls to Avoid:** Causal mask leakage (future tokens peeking into past) must be strictly prevented using `is_causal=True`.

### 6. `block.py` — Transformer Decoder Block
* **Easy Intuition:** One repeating floor in the transformer skyscraper.
* **Deep Mechanics:** Pre-RMSNorm: $h = x + \text{Attn}(\text{Norm}(x))$, $\text{out} = h + \text{MLP}(\text{Norm}(h))$. Clean residual highway ensures deep backpropagation without vanishing gradients.

### 7. `model.py` — Full SLM Assembly
* **Easy Intuition:** The complete machine from raw tokens to word predictions.
* **Deep Mechanics:** Token Embeddings $\rightarrow$ $N$ Blocks $\rightarrow$ RMSNorm $\rightarrow$ LM Head $\rightarrow$ Cross-Entropy Loss with `ignore_index=-100`.

### 8. `tokenizer.py` — Tokenizer Wrapper
* **Easy Intuition:** Translates human words into integer IDs.
* **Deep Mechanics:** BPE tokenizer. Falls back to character-level encoding if offline.
* **Pitfalls to Avoid:** The vocab mismatch bug—always ensure `config.vocab_size == tokenizer.vocab_size`.

### 9. `checkpoint.py` — Checkpoint Manager
* **Easy Intuition:** The save game button.
* **Deep Mechanics:** Preserves model weights, optimizer momentum buffers, step counter, and configuration dictionary.

### 10. `dataset.py` — Data Loader
* **Easy Intuition:** Feeds text batches into the model during training.
* **Deep Mechanics:** Auto-regressive shifting ($x = \text{tokens}[:-1]$, $y = \text{tokens}[1:]$). Memory-mapped binary reading (`np.memmap`) for zero RAM overhead.

### 11. `generation.py` — Inference Engine
* **Easy Intuition:** Takes a prompt and predicts words one by one.
* **Deep Mechanics:** Prefill prompt into KV-cache, decode step-by-step with Greedy, Temperature, Top-K, and Top-P (Nucleus) sampling.

### 12. `train.py` — Training Engine
* **Easy Intuition:** The training supervisor that calculates loss and updates weights.
* **Deep Mechanics:** AdamW with weight decay splitting, Cosine LR scheduler with linear warmup, and gradient clipping at 1.0.

### 13. `test_model.py` — Verification Suite
* **Easy Intuition:** The 7-point crash test facility verifying shapes, gradients, causal masking, and KV-cache equivalence.
