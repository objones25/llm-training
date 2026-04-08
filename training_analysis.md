> **[ARCHIVED — from 6-layer experiment (d_model=512, n_heads=8, vocab=8192), superseded by 604M run (d_model=2048, n_heads=32, vocab=32768)]**

# Training Run Analysis — 20,000 Steps (Final)

**Checkpoint:** `checkpoints/checkpoint_0020000.pt`
**Config:** 6-layer, d_model=512, n_heads=8, vocab=8192
**Evaluation:** `val loss=3.7050`, `perplexity=40.65` (50 batches, 819,200 tokens)

---

## Loss Curve (`loss.png`)

- Initialized at ~9.23 — correct for log(8192) ≈ 9.01, confirming GPT-2 style weight initialization is working.
- Strong decay during warmup (steps 0–1500).
- Brief plateau at steps 1500–1750, resolved on its own.
- Steady progress through step 10,000; continued descent to ~3.5–3.6 by step 20,000.
- Persistent high-frequency noise (±1.0 loss unit) throughout — batch-level variation, not instability.
- Validation loss tracks training closely throughout — no overfitting. Final val loss 3.7050.
- **Best loss recorded: 2.8216** (mid-run, transient).
- Loss is still declining at step 20,000 — not fully converged at this compute budget.

---

## Learning Rate Schedule (`lr.png`)

- Linear warmup to ~1.45e-4 over 1,500 steps.
- Smooth cosine decay to ~1e-6 by step 20,000.
- Full cosine cycle completed exactly at max_steps — schedule configured correctly.

---

## Gradient Norms (`grad_norm.png`)

Overall healthy, with one notable early instability window.

- Steps 0–2500: High variance, grad_norm_max reaching ~2.0–2.2 (structural embedding noise).
- **Steps 2500–5500:** Spike zone — major event at step ~4200 (grad_norm up to ~12.0). Clipping fired; model recovered.
- **Steps 7800–9000:** Dense spike cluster (17 consecutive spike-dump events) during warmup/cosine transition. All self-correcting.
- **Steps 10,000–20,000:** Significantly tightened. Total norm converges to ~1.5–2.0 band, max rarely exceeds 3.0. Only 4 isolated spikes (steps 12843, 13166, 17548, 18389) — sparse and self-correcting.
- `grad_norm_min` flat at ~0.002–0.005 throughout — expected for LayerNorm parameters.

**Anomalies logged:**

- 1 WARNING: `step=4687 layer=token_embedding.weight grad_norm=10.5137` (threshold 10.0) — one-time event, no cascade.
- 285 total spike-dump events (threshold crossed). 123 before step 10k (structural), 162 after (mild, isolated). Zero NaN/Inf events. Zero RuntimeErrors.

---

## Gradient Heatmap (`grad_heatmap.png`)

- Early phase (0–5000): Scattered dark blue patches in attention layers — mild vanishing in specific heads.
- Dark stripe at ~step 4200 (spike recovery region).
- **Steps 10,000–20,000:** More uniform yellow-green across all layers. Fewer dark stripes than 10k baseline — gradient flow improved in second half.
- Final steps (18k–20k): Uniformly bright yellow, all layers actively receiving gradient signal.

The persistent mild darkness in early attention layers (layers 0–2, q_proj/v_proj) is structural — these layers have smaller gradient contributions relative to feed-forward layers. Not a training failure.

---

## Weight Norms (`weight_norm.png`)

Healthy throughout the full run. Uniformly bright yellow across all layers and all 20,000 steps. No weight collapse, no pathological growth. Weight decay functioning correctly. Cleanest plot in the run — no action needed.

---

## Gradient Histogram (`grad_hist.png`)

**Bimodal distribution persists:**

- Primary mode: ~10⁻² (dominant cluster, ~26 parameters) — LayerNorm and attention Q/K/V layers
- Secondary mode: ~10⁻¹ (broader tail, ~9 parameters) — feed-forward and output projection layers

This ~10× magnitude difference between parameter populations is a meaningful inefficiency. The primary peak sharpened vs. the 10k baseline (more parameters concentrating in the small-gradient region), but the bimodality did not resolve. See **Bimodal Gradient Distribution** section below.

---

## Summary

Training is **stable and complete**. Loss improved from 3.7–3.9 at step 10k to 3.5–3.6 at step 20k. No divergence. No numerical instability. The model is not converged — it is compute-limited, not data-limited. The data budget is far larger than what was consumed.

---

## Chinchilla Scaling Analysis

**Actual dataset size (measured from binary files):**

| File      | Size    | Tokens             |
| --------- | ------- | ------------------ |
| train.bin | 23.3 GB | 11,658,045,440     |
| val.bin   | 0.24 GB | 117,919,744        |
| **Total** | 23.6 GB | **11,775,965,184** |

**Current model vs. data budget:**

| Metric                                | Value               |
| ------------------------------------- | ------------------- |
| Non-embedding params (current)        | 18,887,680 (~18.9M) |
| Tokens trained (20k × 32 × 512)       | 327,680,000 (~328M) |
| Chinchilla-optimal tokens for 18.9M N | ~378M               |
| % of optimal token budget consumed    | **87%**             |
| % of available data consumed          | **2.8%**            |

The current model is effectively at Chinchilla-optimal compute allocation (87% of the D=20N target). However, **97.2% of the available data is unused**.

**Chinchilla-optimal model for the full dataset:**

To train compute-optimally on all 11.78B tokens: `N_optimal = D / 20 = 589M non-embedding params`.

Candidate architectures:

| Config      | d_model | n_heads | d_ff | N (non-emb) | Notes                                  |
| ----------- | ------- | ------- | ---- | ----------- | -------------------------------------- |
| 12L × 2048d | 2048    | 32      | 8192 | ~604M       | Closest match; standard GPT-2-XL shape |
| 24L × 1408d | 1408    | 16      | 5632 | ~571M       | Deeper; better gradient flow           |
| 36L × 1152d | 1152    | 16      | 4608 | ~573M       | Very deep; diminishing returns vs. 24L |

**Recommended next config:** `n_layers=12, d_model=2048, n_heads=32, d_ff=8192` (604M params). This is the closest match to Chinchilla-optimal, uses a standard shape (GPT-2 XL architecture), and maps cleanly onto tensor parallelism if scaling to multiple GPUs. Requires cloud GPU (Lambda Labs, RunPod, or similar) — this config is not practical on MPS.

**Compute estimate for the full run:**

```text
C = 6 × N × D = 6 × 604M × 11.78B ≈ 4.27 × 10¹⁹ FLOPs ≈ 42.7 EFLOPs
```

At H100 throughput (~1,000 TFLOPS effective): ~42,700 GPU-seconds ≈ **~12 GPU-hours**.

---

## Bimodal Gradient Distribution

The persistent bimodal distribution (10⁻² vs. 10⁻¹) indicates two parameter populations receiving systematically different gradient signal. This reduces effective learning efficiency — the optimizer applies the same LR to both groups, understepping for the large-gradient group and overstepping for the small-gradient group.

**Root cause:** Mixed parameter types with different gradient scales in a single parameter group:

- Small-gradient group: LayerNorm params, attention Q/K/V weights (sparse, low-magnitude updates)
- Large-gradient group: Feed-forward W1/W2, attention output projection (dense, high-magnitude updates)

**Techniques to address, in order of implementation effort:**

| Technique                            | Effort | Expected Impact | Notes                                                                                      |
| ------------------------------------ | ------ | --------------- | ------------------------------------------------------------------------------------------ |
| Per-group learning rates             | Low    | Medium          | Give embeddings and LN lower LR; FF layers higher LR                                       |
| Scaled output projection init        | Low    | Medium          | Init `out_proj` and `ff.w2` with `std / sqrt(2 * n_layers)`                                |
| Global → per-layer gradient clipping | Medium | Medium          | Prevents large-gradient layers from dominating the clip budget                             |
| Muon optimizer (matrix params only)  | High   | High            | Orthogonalizes gradient updates for weight matrices; eliminates bimodality by construction |

**Recommended immediate fix (lowest effort, meaningful impact):** Add `output_projection_scale` to GPT-2 style init — scale the residual-path output projections (`attn.out_proj` and `ff.w2`) by `1 / sqrt(2 * n_layers)`. This is the original GPT-2 paper's residual init prescription and directly reduces the large-gradient population at the source. The small-gradient population (LayerNorm) is structural and benign — no action needed there.

---

## Recommended Actions for Next Run

| Priority | Action                                             | Rationale                                                                                  |
| -------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| High     | Scale to ~604M params, train on full 11.78B tokens | Chinchilla-optimal; current 18.9M model is massively underparameterized for available data |
| High     | Apply residual projection scaling at init          | Reduces bimodal gradient distribution; improves learning efficiency                        |
| Medium   | Per-parameter-group learning rates                 | Further reduces bimodality; gives embeddings lower LR independently                        |
| Medium   | Cloud GPU (H100 or A100)                           | ~12 GPU-hours for full Chinchilla-optimal run; not feasible on MPS                         |
| Low      | Extend current model to 30k–50k steps              | Marginal gain only — loss still declining but compute-inefficient at this N                |
