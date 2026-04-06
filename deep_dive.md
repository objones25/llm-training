# Deep Dive: Architecture, Scaling Laws, Training & Inference

Now working directly from the five papers you uploaded. This is a comprehensive synthesis grounded purely in the source texts.

---

## 1. What the Scaling Laws Actually Measure

All five papers converge on cross-entropy loss as the primary metric. As stated directly in the Kaplan paper: the test loss of a Transformer trained to autoregressively model language can be predicted using a power-law when performance is limited by only either the number of non-embedding parameters N, the dataset size D, or the optimally allocated compute budget C_min.

The three independent power laws from Kaplan are:

- **Parameters only (trained to convergence):** L(N) = (N_c/N)^αN, with αN ≈ 0.076
- **Dataset only (early stopped):** L(D) = (D_c/D)^αD, with αD ≈ 0.095
- **Compute (optimally allocated):** L(C_min) = (C_min_c/C_min)^α_min_C, with α_min_C ≈ 0.050

---

## 2. Transformer Architecture — What Matters and What Doesn't

This is one of the most important and underappreciated findings in Kaplan. Model performance depends most strongly on scale, which consists of three factors: the number of model parameters N (excluding embeddings), the size of the dataset D, and the amount of compute C used for training. Within reasonable limits, performance depends very weakly on other architectural hyperparameters such as depth vs. width.

More precisely, from the same paper: Transformer performance depends very weakly on the shape parameters n_layer, n_heads, and d_ff when we hold the total non-embedding parameter count N fixed. Aspect ratio in particular can vary by a factor of 40 while only slightly impacting performance; an (n_layer, d_model) = (6, 4288) reaches a loss within 3% of the (48, 1600) model.

This has profound implications: **you can choose depth vs. width almost freely for engineering reasons (parallelism, hardware efficiency), and it barely matters for loss.**

### Embeddings — a Critical Subtlety

Kaplan deliberately excluded embedding parameters from N, and this turns out to be important. The Reconciling paper (2406.12907) directly addresses this: Kaplan studied relationships in terms of non-embedding parameters (N\E) and non-embedding compute (C\E), excluding the linear layers embedding the vocabulary and position indices. By contrast, Chinchilla studied total parameters (N_T) and total compute (C_T).

This is not a minor detail — it's the primary explanation for why the two studies produced different scaling exponents (0.73 vs 0.50). This note finds that much of this discrepancy can be attributed to Kaplan counting non-embedding rather than total parameters, combined with their analysis being performed at small scale. Simulating the Chinchilla study under these conditions produces biased scaling coefficients close to Kaplan's.

From the experimental results in the same paper: when coefficients are fitted to N*T, we find N_T ∝ C^0.49_T and for N\E, we find N\E ∝ C^0.74*\E. These match closely with the Chinchilla and Kaplan coefficients.

---

## 3. The Compute Equation: Where the FLOPs Come From

Kaplan provides the fundamental FLOPs accounting that all subsequent papers rely on. From the paper directly: The total amount of non-embedding compute used during training can be estimated as C = 6NBS, where B is the batch size, S is the number of parameter updates, and the factor of 6 accounts for the forward and backward passes.

The factor of 6 comes from: forward pass ≈ 2N FLOPs per token (matrix multiply factor of 2), backward pass ≈ 2× the forward pass, giving 6N total per training token. For inference, the Sardana paper uses: the standard approximation of FLOPs for transformer models with N parameters: 6N per training token and 2N per inference token.

This 3:1 ratio between training and inference compute per token is fundamental to the "overtrain for inference efficiency" argument.

### Per-Layer Breakdown

From Kaplan's Table 1 (reproduced in the document): the major contributors to compute per token are QKV projections (2 × 3 × n_layer × d_model × d_attn), feedforward blocks (2 × n_layer × 2 × d_model × d_ff), and the attention mask (2 × n_layer × n_ctx × d_attn). Note that for contexts and models with d_model > n_ctx/12, the context-dependent computational cost per token is a relatively small fraction of the total compute. This is why the standard 6N approximation holds well in practice.

---

## 4. Normalization & Training Stability — The AdamW Finding

Chinchilla made a notable architectural/optimization change versus its predecessor Gopher. From the paper: We use AdamW for Chinchilla rather than Adam as this improves the language modelling loss and the downstream task performance after finetuning. A model trained with AdamW only passes the training performance of a model trained with Adam around 80% of the way through the cosine cycle, though the ending performance is notably better.

This is significant: AdamW trains _worse_ for most of the run but ends better — indicating that decoupled weight decay regularization is especially important at large scale.

### Learning Rate Schedules

Both Kaplan and Chinchilla converge on cosine decay schedules. From Kaplan: the choice of learning rate schedule is mostly irrelevant, as long as the total summed learning rate is sufficiently large, and the schedule includes a warmup period and a final decay to near-vanishing learning rate.

Chinchilla adds a critical practical constraint: setting the cosine cycle length too much longer than the target number of training tokens results in sub-optimally trained models. We find that overestimating the number of training steps beyond 25% leads to clear drops in performance.

This means you must **decide your total training duration before you start**, or accept a performance penalty — a significant practical constraint for practitioners.

---

## 5. The Parametric Loss Function (Chinchilla's Most Useful Formula)

The most practically useful output of Chinchilla is its parametric loss model. From the paper: we propose the following functional form: L̂(N, D) ≜ E + A/N^α + B/D^β. The first term captures the loss for an ideal generative process on the data distribution. The second term captures the fact that a perfectly trained transformer with N parameters underperforms the ideal generative process. The final term captures the fact that the transformer is not trained to convergence.

The fitted values from Chinchilla: E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28.

However, the Besiroglu replication (2404.10102) reveals these values are problematic. The parameter values reported in the paper are rounded, and especially for the data exponent this rounding is significant enough to make the fit much worse than it should be. The true value of the data exponent β should be around 0.2849 according to the comments in the TeX source code of Hoffmann et al.; using the value 0.28 instead introduces a positive bias on the order of ≈ (10^11)^0.0049 − 1 ≈ 13% for a typical dataset size of D = 10^11 tokens.

The Epoch AI corrected estimates are: E = 1.8172, A = 482.01, B = 2085.43, α = 0.3478, β = 0.3658 ⟹ N\*\_T ∝ C^0.51_T.

---

## 6. Training-Time Scaling: The Optimal Allocation Problem

### Kaplan's View

When training within a fixed compute budget C, but with no other constraints, the optimal model size N, optimal batch size B, optimal number of steps S, and dataset size D should grow as: N ∝ C^0.73, B ∝ C^0.24, S ∝ C^0.03. As the computational budget C increases, it should be spent primarily on larger models, without dramatic increases in training time or dataset size.

The S ∝ C^0.03 result is remarkable — the number of serial training steps barely grows with compute. Most of the compute increase should go to **model size** and **batch size parallelism**.

### Chinchilla's Revision

For compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled. All three approaches suggest that as compute budget increases, model size and the amount of training data should be increased in approximately equal proportions.

The table from Chinchilla's paper is directly in your uploaded document — for a 67B parameter model, the compute-optimal token count is 1.5 trillion, and for a 175B model (GPT-3's size), it should have been trained on 3.7 trillion tokens, not 300 billion.

### What "Compute-Optimal" Actually Means

From Kaplan: for compute-efficient training, we should train to a fixed percentage αN/αS ≈ 10% above the converged loss. Compute-efficient training uses 7.7× fewer parameter updates, 2.7× more parameters, and 65% less compute to reach the same loss compared to training to convergence.

This is the key insight often missed: **compute-optimal does not mean trained to convergence**. You stop well before the model has learned everything it could from the data.

---

## 7. Inference-Time Scaling: The "Beyond Chinchilla" Argument

The Sardana paper (2401.00448) introduces the critical extension. The core math: we are interested in minimizing the sum of our training and inference FLOPs under the constraint L(N, D_tr) = ℓ: minimize 6ND_tr + 2ND_inf, where 6N is the cost per training token and 2N per inference token.

The implication: LLM researchers expecting reasonably large inference demand (~1B requests) should train models smaller and longer than Chinchilla-optimal. Furthermore, model quality continues to improve as we scale tokens per parameter to extreme ranges (up to 10,000).

A concrete example from the paper: An LLM developer seeking to train a 13B model who expects 2 trillion tokens of inference demand during the model's lifetime can reduce their total compute by 1.7 × 10^22 FLOPs (17%) by instead training a 7B model on more data.

### Hardware Utilization Asymmetry

The paper also identifies a crucial real-world asymmetry: inference hardware utilization can be much lower than training utilization, since small batch size computation can result in low Model FLOPs Utilization (MFU). MFU can be as low as ~1% for inference but is typically 40-60% during training. During generation, output tokens must be produced sequentially, resulting in low utilization due to memory bandwidth constraints.

This 50× MFU gap between training and auto-regressive inference generation means the FLOP cost of inference is far worse in wall-clock terms than the raw number suggests — making the case for smaller models even stronger.

---

## 8. Batch Size: The Critical Batch Size Concept

Kaplan introduces the concept of a "critical batch size" that determines the optimal parallelism tradeoff. The ideal batch size for training these models is roughly a power of the loss only, and continues to be determinable by measuring the gradient noise scale; it is roughly 1-2 million tokens at convergence for the largest models we can train. The critical batch size Bcrit(L) is independent of model size, and only depends on the loss L.

The formula: B_crit(L) = B*/L^(1/αB), with B* ≈ 2×10^8 tokens and αB ≈ 0.21. As training progresses and loss decreases, the optimal batch size grows. This has direct implications for distributed training: training at B ≪ B_crit minimizes the number of training steps, while B ≫ B_crit minimizes the use of compute. Training at the critical batch size makes a roughly optimal time/compute tradeoff, requiring 2S_min training steps and processing 2E_min data examples.

---

## 9. The Data Exponent Problem: Scaling Laws at Extreme Ratios

The Sardana paper reveals a critical limitation of applying Chinchilla to "overtrained" models. When fitting Chinchilla coefficients using only typical token ratio runs, this formula overestimates the impact of additional training data as we move to the extreme ratio regime. As we use more extreme data in our fitting procedure, our scaling curves become flatter.

Their empirical table of refitted coefficients (Table 1 in the paper) shows the data exponent β rising from 0.08 (at ≤100 tok/param) to 0.24 (all data), compared to Chinchilla's 0.28 — meaning the Chinchilla formula **overpredicts** the benefit of additional tokens when you're far beyond the 20:1 regime. The loss improvements are real but smaller than the formula predicts.

---

## 10. Summary: The Integrated Picture

| Question                             | Answer from Papers                                                                           |
| ------------------------------------ | -------------------------------------------------------------------------------------------- |
| What governs loss?                   | N, D, C as power laws. Architecture shape (depth/width) barely matters                       |
| Kaplan vs. Chinchilla disagreement?  | Mostly explained by embedding parameter counting + small-scale analysis bias                 |
| Optimal N/D split?                   | Equal scaling (Chinchilla); not 73/27 (Kaplan)                                               |
| Should you train to convergence?     | No — stop ~10% above converged loss for compute efficiency                                   |
| What about inference?                | Smaller model + more tokens wins once inference demand is large (~1B requests)               |
| Does loss keep decreasing with data? | Yes, up to 10,000 tokens/param, but slower than Chinchilla predicts                          |
| Is the Chinchilla formula accurate?  | Parametric fit (Approach 3) has known numerical errors; Epoch AI corrected version preferred |
| Cosine schedule constraint?          | Must match training duration — >25% overshoot degrades performance                           |
| Batch size?                          | Critical batch size grows as loss decreases; ~1-2M tokens at convergence                     |
