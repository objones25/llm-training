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

### Current Run: RunPod A100

**Pod details (2026-04-07 run):**

- Provider: RunPod Secure Cloud
- GPU: 1× A100 (80 GB)
- Base image: RunPod PyTorch (Lambda Stack 24.04)
- SSH: `root@195.26.233.65 -p 19914`
- Workspace: `/workspace/llm-training/`

---

### Reproducing This Setup Step by Step

1. **Provision pod** at `runpod.io` → Secure Cloud
   - Choose GPU (A100 or H100 depending on availability)
   - Template: **RunPod PyTorch** (or Lambda Stack — comes with CUDA pre-installed)
   - Volume: **50 GB** (data ~23.3 GB + checkpoints ~7 GB + overhead)
   - No network volume needed; no encryption needed
   - Add your SSH public key under SSH settings

2. **Connect**:

   ```bash
   ssh -p <port> root@<ip>
   ```

   Type `yes` to accept the host key on first connect.

3. **Install uv and clone repo**:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
   git clone https://github.com/objones25/llm-training.git /workspace/llm-training
   cd /workspace/llm-training
   uv sync
   ```

4. **Fix PyTorch CUDA compatibility** (RunPod containers have CUDA 12.4 driver; torch 2.11+ requires newer):

   ```bash
   uv pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
   ```

   > This is only needed if the pod driver reports version `12040` (`nvidia-smi` shows `CUDA Version: 12.4`).
   > pyproject.toml already pins `torch>=2.6.0,<2.7.0` so `uv sync` won't override this.

5. **Upload pre-tokenized data and .env** (run from your local Mac — not inside the pod):

   ```bash
   scp -P <port> /Users/theelusivegerbilfish/Python_Projects/llm-training/.env root@<ip>:/workspace/llm-training/
   scp -P <port> /Users/theelusivegerbilfish/Python_Projects/llm-training/tokenizer.json root@<ip>:/workspace/llm-training/
   scp -P <port> /Users/theelusivegerbilfish/Python_Projects/llm-training/data/train.bin root@<ip>:/workspace/llm-training/data/
   scp -P <port> /Users/theelusivegerbilfish/Python_Projects/llm-training/data/val.bin root@<ip>:/workspace/llm-training/data/
   ```

   > train.bin is 23.3 GB — allow ~1–2 hours over a typical connection.
   > `scp` uses `-P` (capital P) for port, unlike `ssh` which uses `-p`.

6. **Start training** inside tmux (survives disconnect):

   ```bash
   tmux new -s train
   uv run python scripts/run_training.py --train-bin data/train.bin --val-bin data/val.bin --device cuda --use-muon --early-stopping-patience 10
   ```

7. **Monitor**:

   ```bash
   tail -f train.log
   watch -n 5 nvidia-smi
   ```

---

### Downloading Results After Training

Run these from your **local Mac** (not inside the pod):

```bash
scp -P <port> root@<ip>:/workspace/llm-training/checkpoints/best.pt /Users/theelusivegerbilfish/Python_Projects/llm-training/checkpoints/
scp -P <port> -r root@<ip>:/workspace/llm-training/plots /Users/theelusivegerbilfish/Python_Projects/llm-training/
scp -P <port> root@<ip>:/workspace/llm-training/train.log /Users/theelusivegerbilfish/Python_Projects/llm-training/
```

For the current run (pod IP `195.26.233.65`, port `19914`):

```bash
scp -P 19914 root@195.26.233.65:/workspace/llm-training/checkpoints/best.pt /Users/theelusivegerbilfish/Python_Projects/llm-training/checkpoints/
scp -P 19914 -r root@195.26.233.65:/workspace/llm-training/plots /Users/theelusivegerbilfish/Python_Projects/llm-training/
scp -P 19914 root@195.26.233.65:/workspace/llm-training/train.log /Users/theelusivegerbilfish/Python_Projects/llm-training/
```

---

### Lambda Labs (Alternative)

Lambda Labs is preferred when H100 availability is good — typically better price/performance than RunPod for long runs.

1. Create account at `lambda.ai`
2. Launch `1× H100 SXM5 80GB` with Ubuntu 22.04 + PyTorch image
3. SSH user is `ubuntu` (not `root`); standard port 22
4. Follow the same steps above — no CUDA compatibility fix needed on Lambda (drivers are kept current)

---

## Checkpoint Strategy

Training saves a single `checkpoints/best.pt` whenever validation loss improves. No numbered checkpoints are written. Disk footprint is bounded:

```text
~604M params × 4 bytes × ~3 tensors (model + optimizer state) ≈ 7 GB (one file)
```

`best.pt` is overwritten in place on each improvement — safe to download at any point during training.

To resume from a checkpoint:

```bash
uv run python scripts/run_training.py --train-bin data/train.bin --val-bin data/val.bin --device cuda --use-muon --resume checkpoints/best.pt
```

---

## Pre-Training Checklist

- [ ] Confirm step-0 loss ≈ `log(8192) ≈ 9.01`
- [ ] Verify `model.n_params ≈ 589M` in training startup log
- [ ] Confirm cosine schedule reaches near-zero at step 700,000
- [ ] Expected loss target: **2.84** — treat >3.0 at convergence as a signal to investigate
- [ ] `plots/` directory exists and writable before launch
- [ ] `data/train.bin` and `data/val.bin` present on instance

