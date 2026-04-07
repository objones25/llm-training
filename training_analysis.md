# Training Run Analysis — 10,000 Steps

**Checkpoint:** `checkpoints/checkpoint_0010000.pt`
**Config:** 6-layer, d_model=512, n_heads=8, vocab=8192

---

## Loss Curve (`loss.png`)

- Initialized at ~9.0 — correct for log(8192) ≈ 9.0, confirming GPT-2 style weight initialization is working.
- Strong decay during steps 0–500 (warmup phase).
- Steady progress through step 5000.
- Plateau with increasing noise in steps 7000–10000. Final loss is approximately 3.7–3.9.
- No divergence or NaN events. Overall a healthy trajectory, though the plateau suggests diminishing returns at this compute budget.

---

## Learning Rate Schedule (`lr.png`)

- Linear warmup ramp to ~3e-4 by step 500.
- Smooth cosine decay to near-zero by step 10000.
- Schedule is correctly configured and executing as expected. No issues.

---

## Gradient Norms (`grad_norm.png`)

**Concerning.** The total gradient norm spikes repeatedly (reaching 5.0–7.0) during steps 2500–5500, with a major spike near step 4200. Gradient clipping is clearly firing — the rapid norm collapse after each spike confirms this. The frequency and magnitude suggest either layer-specific instability or outlier batches driving large updates.

The min/max spread widens significantly during the spike region, meaning some layers are contributing disproportionately to the total norm.

---

## Gradient Heatmap (`grad_heatmap.png`)

Most layers show moderate norms (yellow-green). A few horizontal stripes — consistent with early attention layers — are noticeably darker (lower norms) throughout training. This indicates uneven gradient flow: some layers are consistently underutilized. No complete layer death, but the imbalance reduces effective model capacity.

---

## Weight Norms (`weight_norm.png`)

Healthy. Almost entirely bright yellow across all layers and all steps, indicating steady, uniform weight growth. Weight decay is functioning correctly — no explosion or collapse. This is the cleanest-looking plot in the run.

---

## Gradient Histogram (`grad_hist.png`)

**Problematic.** The distribution is bimodal:
- A large peak at ~10^-2 (roughly 26 parameters)
- A secondary peak at ~10^-1 (roughly 9 parameters)

That is approximately a 10x difference in gradient magnitude between two parameter populations. This indicates some layers are receiving significantly weaker gradient signal than others — consistent with the darker stripes in the heatmap. Healthy training typically shows a unimodal, roughly log-normal distribution.

---

## Summary

Training is **stable but suboptimal**. Loss decreases monotonically with no divergence, but the gradient instability window (steps 2500–5500) and bimodal gradient distribution suggest the model is not learning as efficiently as it could.

### Recommended actions

| Priority | Action | Rationale |
| -------- | ------ | --------- |
| High | Increase warmup steps to 1000–2000 | Current 500-step warmup may be too short; larger LR earlier coincides with the spike region |
| High | Lower peak learning rate to 1.5e-4 | Reduces spike frequency; the current 3e-4 is aggressive for this model size |
| Medium | Extend `max_steps` to 20000 | Loss has not fully converged — the plateau is compute-limited, not data-limited |
| Medium | Investigate per-layer gradient statistics around step 4200 | Identify which layer triggers the largest spike |
| Low | Review initialization for attention projection layers | The consistently dark heatmap stripes suggest those layers may benefit from scaled init |
