"""
PLASTIC Experiment: LR_syn Predicts Plasticity (Theorem 1)

Tests the core claim: local redundancy on synthetic memorization
LR_syn(θ) = E[||∇_θ ℓ(f_θ(x), y_rand)||²] predicts learnability.

Simple CPU experiment: train N models with different initializations,
measure LR_syn at init, then measure how fast they learn a real task.
"""

import torch
import torch.nn as nn
import numpy as np
import json, os

torch.manual_seed(0)
np.random.seed(0)

DEVICE = "cpu"
N_MODELS = 50
N_EPOCHS = 100
D_IN = 8
D_HIDDEN = 64


class MLP(nn.Module):
    def __init__(self, init_scale=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_IN, D_HIDDEN), nn.Tanh(),
            nn.Linear(D_HIDDEN, D_HIDDEN), nn.Tanh(),
            nn.Linear(D_HIDDEN, 2),
        )
        # Scale initialization to vary initial LR_syn across models
        with torch.no_grad():
            for p in self.parameters():
                p.mul_(init_scale)

    def forward(self, x):
        return self.net(x)


def make_data(n=2000):
    x = torch.randn(n, D_IN)
    # Clear non-linear structure: parity-like XOR boundary
    y = ((x[:, 0] * x[:, 1] + x[:, 2] * x[:, 3] - x[:, 4] * x[:, 5]) > 0).long()
    return x, y


def compute_lr_syn(model):
    """LR_syn = E[||∇_θ ℓ(f_θ(x), y_rand)||²] on random labels."""
    model.eval()
    x_syn = torch.randn(256, D_IN, device=DEVICE)
    y_syn = torch.randint(0, 2, (256,), device=DEVICE)
    logits = model(x_syn)
    loss = nn.CrossEntropyLoss()(logits, y_syn)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    grad_norm_sq = sum(g.reshape(-1).norm().item() ** 2 for g in grads)
    model.train()
    return grad_norm_sq / 256  # normalize by batch


def train(model, x, y, n_epochs=N_EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    x, y = x.to(DEVICE), y.to(DEVICE)
    for ep in range(n_epochs):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
    # Return test accuracy
    model.eval()
    with torch.no_grad():
        acc = (model(x).argmax(1) == y).float().mean().item()
    return acc


def run():
    print("=" * 70)
    print("PLASTIC Experiment: LR_syn Predicts Learnability")
    print("=" * 70)

    x_train, y_train = make_data(2000)
    x_test, y_test = make_data(1000)

    init_scales = np.linspace(0.1, 2.5, N_MODELS)
    results = []

    for i, scale in enumerate(init_scales):
        model = MLP(init_scale=scale).to(DEVICE)
        lr_syn = compute_lr_syn(model)
        acc = train(model, x_train, y_train)
        test_acc = ((model(x_test.to(DEVICE)).argmax(1) == y_test.to(DEVICE))
                    .float().mean().item())
        results.append({"scale": float(scale), "lr_syn": float(lr_syn),
                        "train_acc": float(acc), "test_acc": float(test_acc)})

        if i % 10 == 0 or i == N_MODELS - 1:
            print(f"  Model {i:2d}/{N_MODELS} | scale={scale:.2f} | "
                  f"LR_syn={lr_syn:.6f} | test={test_acc:.3f}")

    # ---- Analysis ----
    lr_syn_vals = np.array([r["lr_syn"] for r in results])
    acc_vals = np.array([r["test_acc"] for r in results])

    # Split into low / high LR_syn groups
    median_lr = np.median(lr_syn_vals)
    low_mask = lr_syn_vals < median_lr
    high_mask = lr_syn_vals >= median_lr

    low_acc = acc_vals[low_mask].mean()
    high_acc = acc_vals[high_mask].mean()
    low_std = acc_vals[low_mask].std()
    high_std = acc_vals[high_mask].std()

    # Correlation
    valid = lr_syn_vals > 0
    corr = np.corrcoef(np.log(lr_syn_vals[valid]), acc_vals[valid])[0, 1]

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Models below median LR_syn:  test acc = {low_acc:.3f} ± {low_std:.3f}")
    print(f"  Models above median LR_syn:  test acc = {high_acc:.3f} ± {high_std:.3f}")
    print(f"  Gap (high - low):            {high_acc - low_acc:+.3f}")
    print(f"  log(LR_syn) vs accuracy:     r = {corr:.3f}")

    if corr > 0.3:
        print("  ✓ Supports Theorem 1: higher LR_syn → better learning")
    if high_acc > low_acc:
        print("  ✓ LR_syn predicts plasticity at initialization")

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/lr_syn_learnability.json", "w") as f:
        json.dump({"results": results, "correlation": float(corr)}, f, indent=2)
    print(f"\n  Results saved to results/lr_syn_learnability.json")
    print("Done.")


if __name__ == "__main__":
    run()
