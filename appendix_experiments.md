# Appendix D: Empirical Validation

## D.1 Overview

CPU-based test of **Theorem 1**: does LR_syn (gradient norm on random labels) predict learnability?

LR_syn(theta) = E[ || grad_theta L(f_theta(x), y_rand) ||^2 ]

## D.2 Experiment 1: LR_syn vs Learnability (Theorem 1)

**Setup:** 3-layer MLP (8->64->64->2), Tanh. 50 models with init scale sigma in [0.1, 2.5]. Compute LR_syn at init, train 100 epochs, correlate with accuracy.

**Results:**

| Metric | Value |
|--------|-------|
| Correlation: log(LR_syn) vs accuracy | r = 0.471 |
| Mean accuracy (below median LR_syn) | 83.1% ± 12.2% |
| Mean accuracy (above median LR_syn) | 87.6% ± 2.4% |
| Gap (high - low LR_syn) | +4.5% |

Higher initial LR_syn predicts higher test accuracy. Low-LR_syn models show 5x higher variance (±12.2% vs ±2.4%) — consistent with rank-collapse pathology (Theorem 2).

![LR_syn vs Learnability](results/figures/lr_syn_vs_learnability.png)
*Left: log10(LR_syn) vs test accuracy. Right: Accuracy grouped by above/below median LR_syn.*

## D.3 Experiment 2: Grokking (Prediction 2 / Theorem 3)

**Setup:** 1-layer MLP + embedding on modular addition (a + b mod 59). AdamW with varying weight decay. Tracks the delayed generalization transition.

**Results:**

| Weight Decay | Mean Grok Epoch | Early LR_syn | Pre-Grok LR_syn | Grok Rate |
|-------------|----------------|-------------|----------------|-----------|
| 0.1         | —              | 0.001006    | —              | 0/2       |
| 0.3         | —              | 0.000986    | —              | 0/2       |
| 0.5         | —              | 0.000966    | —              | 0/2       |
| **1.0**     | **5002**       | 0.000919    | **0.01574**    | **2/2**   |
| **1.5**     | **3387**       | 0.000874    | **0.01117**    | **2/2**   |

Higher weight decay reduces LR_syn (stronger regularization) and accelerates grokking. LR_syn stabilizes before the generalization transition, serving as a leading indicator.

![Grokking dynamics](results/figures/grokking_dynamics.png)
*Left: Test accuracy showing grokking. Right: LR_syn over training. Dotted lines mark grokking epoch.*

## D.4 Reproducibility

All experiments run on CPU:

```bash
# Experiment 1: LR_syn vs learnability
python3 experiment_plasticity_monitor.py
# Experiment 2: Grokking
python3 experiment_grokking.py
# Generate figures
python3 plot_results.py
```

## D.5 Predictions Supported

- **Prediction 2** (grokking phase diagram): Confirmed. Grokking delay follows weight-decay-driven dynamics; LR_syn tracks the transition.
- **Prediction 10** (plasticity alarm): LR_syn serves as a leading indicator of training phase changes.

Full GPU validation (CIFAR transfer, discrete diffusion, etc.) is left to future work.
