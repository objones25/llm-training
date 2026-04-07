# Next Run: 604M Parameter Model

## Architecture

| Hyperparameter         | Value | Notes             |
| ---------------------- | ----- | ----------------- |
| `n_layers`             | 12    |                   |
| `d_model`              | 2048  |                   |
| `n_heads`              | 32    | head_dim = 64     |
| `d_ff`                 | 8192  | 4× d_model        |
| `vocab_size`           | 8192  | unchanged         |
| `seq_len`              | 512   | unchanged         |
| Non-embedding params N | ~589M | see formula below |

**Non-embedding param count formula:**

```
N = n_layers × (4×d_model² + 2×d_model×d_ff + 4×d_model) + 2×d_model
  = 12 × (4×2048² + 2×2048×8192 + 4×2048) + 2×2048
  ≈ 604M
```

---

## Training Hyperparameters

| Hyperparameter  | Value        | Rationale                                                          |
| --------------- | ------------ | ------------------------------------------------------------------ |
| `max_steps`     | 700,000      | D = max_steps × batch_size × seq_len ≈ 11.5B tokens ≈ full dataset |
| `batch_size`    | 32           | keep constant; scale via gradient accum if OOM                     |
| `seq_len`       | 512          | unchanged                                                          |
| `learning_rate` | 2e-4         | Chinchilla recommendation for ~600M scale                          |
| `warmup_steps`  | 7,000        | 1% of max_steps                                                    |
| `weight_decay`  | 0.1          | unchanged                                                          |
| `grad_clip`     | 1.0          | unchanged                                                          |
| `adamw_betas`   | (0.9, 0.999) | unchanged                                                          |
| `ln_lr_mult`    | 3.0          | compensates for bounded LN gradients                               |
| `embed_lr_mult` | 0.1          | embeddings benefit from lower LR                                   |

**Token budget check (Chinchilla-optimal):**

```
D_optimal = 20 × N = 20 × 589M = 11.78B tokens
Available:  train.bin = 11.66B + val.bin = 0.117B = 11.78B tokens  ✓
```

**Compute estimate:**

```
C = 6 × N × D = 6 × 589M × 11.78B ≈ 41.7 EFLOPs
H100 throughput: ~3.4 EFLOP/hr (BF16, ~70% utilization)
Estimated wall time: ≈ 12 hours on 1× H100
```

**Expected loss (Epoch AI corrected Chinchilla):**

```
L(N, D) = 1.8172 + 482.01 / N^0.3478 + 2085.43 / D^0.3658
        = 1.8172 + 482.01 / 589M^0.3478 + 2085.43 / 11.78B^0.3658
        ≈ 2.84
```

---

## Config Changes vs Current Run

```python
# src/config.py — update these defaults (already done)
n_layers: int = 12       # was 6
d_model: int = 2048      # was 512 (or previous default)
n_heads: int = 32        # was 8
d_ff: int = 8192         # was 2048
max_steps: int = 700_000 # was 20_000
warmup_steps: int = 7_000 # was 1_500
learning_rate: float = 2e-4  # was 1.5e-4
```

---

## Cloud GPU Setup

### Recommended Provider: Lambda Labs

Lambda Labs offers the best price/performance for single-node H100 jobs.

**Steps:**

1. **Create account** at `lambda.ai`

2. **Generate SSH key** (if you don't have one):

   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   cat ~/.ssh/id_ed25519.pub  # copy this to Lambda dashboard
   ```

3. **Launch instance**:
   - Instance type: `1× H100 SXM5 80GB` (~$2.49/hr as of 2026-04)
   - OS image: `Ubuntu 22.04 LTS + PyTorch`
   - Add your SSH public key

4. **Connect and set up**:

   ```bash
   ssh ubuntu@<instance_ip>

   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.cargo/env

   # Clone repo
   git clone https://github.com/<your_username>/llm-training.git
   cd llm-training

   # Install dependencies
   uv sync
   ```

5. **Upload pre-tokenized data** (avoids re-tokenizing on the GPU instance):

   ```bash
   # From your local machine
   rsync -avz --progress data/train.bin data/val.bin ubuntu@<instance_ip>:llm-training/data/
   ```

   > train.bin is 23.3 GB — rsync will resume if interrupted.

6. **Configure for GPU**:

   ```bash
   # Edit src/config.py or pass overrides via environment / script
   # Key changes:
   #   device = "cuda"
   #   use_amp = True      # BF16 mixed precision on H100
   #   use_compile = True  # torch.compile for ~20% throughput gain
   ```

7. **Start training** (inside tmux/screen so it survives disconnect):

   ```bash
   tmux new -s train
   uv run python src/train.py
   ```

8. **Monitor**:
   ```bash
   # In a second tmux pane
   tail -f train.log
   watch -n 5 nvidia-smi
   ```

### Alternative: RunPod

RunPod is useful when Lambda has low H100 availability.

1. Go to `runpod.io` → **Secure Cloud** → filter for H100 SXM
2. Select **RunPod PyTorch** template (comes with CUDA + PyTorch pre-installed)
3. Set **Volume** to 50 GB (for data + checkpoints)
4. Expose port 22 for SSH
5. Follow the same setup steps as Lambda above

---

## Checkpoint Strategy

With 700k steps at `checkpoint_every=1000`, you'll produce 700 checkpoints. Disk space:

```
~604M params × 4 bytes × ~3 tensors (model + optimizer state) ≈ 7 GB/checkpoint
700 checkpoints × 7 GB = 4.9 TB  ← DO NOT keep all of them
```

**Recommended:** checkpoint every 5,000 steps and keep only the last 3:

```python
# In src/config.py
checkpoint_every: int = 5_000
```

Add cleanup logic to `src/checkpoint.py` to delete checkpoints older than the last 3 when saving.

---

## Pre-Training Checklist

- [ ] Confirm step-0 loss ≈ `log(8192) ≈ 9.01`
- [ ] Verify `model.n_params ≈ 589M` in training startup log
- [ ] Confirm cosine schedule reaches near-zero at step 700,000
- [ ] Expected loss target: **2.84** — treat >3.0 at convergence as a signal to investigate
- [ ] `plots/` directory exists and writable before launch
- [ ] `data/train.bin` and `data/val.bin` present on instance

---

## Before This Run: Muon Prerequisite

The bimodal gradient distribution (LN ~10⁻², weight matrices ~10⁻¹) is **structurally irreducible** — it reflects a mathematical difference in how gradients propagate through normalization vs. linear ops.

The three-group AdamW (already implemented) handles the optimizer mismatch by giving each group an appropriate LR. For the 604M run, consider also implementing **Muon** for weight matrix parameters before training:

- Muon applies Newton-Schulz orthogonalization to gradient updates
- Eliminates bimodality by construction — update norms are uniform regardless of raw gradient magnitude
- Particularly valuable at 604M scale where training stability matters more
- Recommended reference: Keller Jordan's Muon implementation (modular-diffusion or muon repo)

Muon is not required for the run to succeed, but it is the most principled fix and worth the implementation cost at this scale.
