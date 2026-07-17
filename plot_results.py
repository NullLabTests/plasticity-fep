"""Generate figures for the PLASTIC experiment."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results/figures", exist_ok=True)

with open("results/lr_syn_learnability.json") as f:
    data = json.load(f)

lr = np.array([r["lr_syn"] for r in data["results"]])
acc = np.array([r["test_acc"] for r in data["results"]])
scale = np.array([r["scale"] for r in data["results"]])

# Fig 1: LR_syn vs test accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(np.log10(lr + 1e-12), acc, c=scale, cmap="viridis", alpha=0.8, s=30)
ax1.set_xlabel("log10(LR_syn)")
ax1.set_ylabel("Test Accuracy")
ax1.set_title("LR_syn vs Learnability (r={:.3f})".format(data["correlation"]))
ax1.grid(alpha=0.3)
cbar = plt.colorbar(ax1.collections[0], ax=ax1)
cbar.set_label("Init Scale")

# Median split bar chart
median = np.median(lr)
low = acc[lr < median]
high = acc[lr >= median]
ax2.bar(["Below Median LR_syn", "Above Median LR_syn"],
        [low.mean(), high.mean()],
        yerr=[low.std(), high.std()], capsize=5, color=["coral", "steelblue"])
ax2.set_ylabel("Mean Test Accuracy")
ax2.set_title("Plasticity-Based Group Split")
ax2.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("results/figures/lr_syn_vs_learnability.png", dpi=150)
print("Saved results/figures/lr_syn_vs_learnability.png")

# Fig 2: Training dynamics for extreme models
fig, ax = plt.subplots(figsize=(8, 5))
idx_low = np.argmin(lr)
idx_high = np.argmax(lr)
ax.scatter([0, 1], [lr[idx_low], lr[idx_high]], s=100, c=["coral", "steelblue"])
ax.set_xticks([0, 1])
ax.set_xticklabels(["Lowest LR_syn\n(init scale={:.2f})".format(scale[idx_low]),
                     "Highest LR_syn\n(init scale={:.2f})".format(scale[idx_high])])
ax.set_ylabel("LR_syn (grad norm on random labels)")
ax.set_title("Dynamic Range of LR_syn Across Initializations")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/lr_syn_range.png", dpi=150)
print("Saved results/figures/lr_syn_range.png")
