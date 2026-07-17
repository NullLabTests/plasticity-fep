"""
PLASTIC Experiment: LR_syn Predicts Grokking (Prediction 2, Theorem 3)

Standard grokking setup: modular addition, embedding + MLP, AdamW.
Tracks LR_syn and shows it serves as an early warning signal.

CPU-friendly: p=59, 5000 epochs, single seed per WD.
"""

import torch
import torch.nn as nn
import numpy as np, json, os, time

torch.manual_seed(0)
np.random.seed(0)
DEVICE = "cpu"


def make_data(p, seed=0, train_frac=0.5):
    rng = np.random.RandomState(seed)
    a = np.arange(p)
    b = np.arange(p)
    aa, bb = np.meshgrid(a, b)
    pairs = np.stack([aa.ravel(), bb.ravel()], axis=1)
    targets = (pairs[:, 0] + pairs[:, 1]) % p

    idx = rng.permutation(len(pairs))
    split = int(train_frac * len(idx))
    a_train, b_train = pairs[idx[:split], 0], pairs[idx[:split], 1]
    a_test, b_test = pairs[idx[split:], 0], pairs[idx[split:], 1]
    y_train, y_test = targets[idx[:split]], targets[idx[split:]]

    return (torch.tensor(a_train), torch.tensor(b_train), torch.tensor(y_train)), \
           (torch.tensor(a_test), torch.tensor(b_test), torch.tensor(y_test))


class GrokNet(nn.Module):
    def __init__(self, p, d_embed=128, d_hidden=256):
        super().__init__()
        self.embed = nn.Embedding(p, d_embed)
        self.net = nn.Sequential(
            nn.Linear(2 * d_embed, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, p),
        )

    def forward(self, a, b):
        e = torch.cat([self.embed(a), self.embed(b)], dim=-1)
        return self.net(e)


@torch.no_grad()
def accuracy(model, a, b, y):
    model.eval()
    return (model(a, b).argmax(1) == y).float().mean().item()


def compute_lr_syn(model, p, n=256):
    model.eval()
    a = torch.randint(0, p, (n,))
    b = torch.randint(0, p, (n,))
    y = torch.randint(0, p, (n,))
    logits = model(a, b)
    loss = nn.CrossEntropyLoss()(logits, y)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    gn = sum(g.reshape(-1).norm().item() ** 2 for g in grads)
    model.train()
    return gn / n


def run_seed(p, d_embed, d_hidden, wd, n_epochs, tr_frac, seed):
    torch.manual_seed(seed)
    (a_tr, b_tr, y_tr), (a_te, b_te, y_te) = make_data(p, seed, tr_frac)
    a_tr, b_tr, y_tr = a_tr.to(DEVICE), b_tr.to(DEVICE), y_tr.to(DEVICE)
    a_te, b_te, y_te = a_te.to(DEVICE), b_te.to(DEVICE), y_te.to(DEVICE)

    model = GrokNet(p, d_embed, d_hidden).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
    crit = nn.CrossEntropyLoss()

    hist = {"epoch": [], "loss": [], "train_acc": [], "test_acc": [],
            "lr_syn": [], "grokked": False, "grok_epoch": None, "wd": wd}

    for ep in range(n_epochs):
        model.train(); opt.zero_grad()
        loss = crit(model(a_tr, b_tr), y_tr)
        loss.backward(); opt.step()

        tr_acc = accuracy(model, a_tr, b_tr, y_tr)
        te_acc = accuracy(model, a_te, b_te, y_te)
        lr_syn = compute_lr_syn(model, p)

        hist["epoch"].append(ep)
        hist["loss"].append(loss.item())
        hist["train_acc"].append(tr_acc)
        hist["test_acc"].append(te_acc)
        hist["lr_syn"].append(lr_syn)

        if te_acc >= 0.95 and not hist["grokked"]:
            hist["grokked"] = True
            hist["grok_epoch"] = ep

        if ep % 500 == 0 or (hist["grokked"] and ep == hist["grok_epoch"]):
            m = " <-- GROKKED" if (hist["grokked"] and ep == hist["grok_epoch"]) else ""
            name = f"wd={wd}"
            print(f"  {name:>6s} ep {ep:4d} | loss {loss.item():.3f} "
                  f"| train {tr_acc:.3f} | test {te_acc:.3f} | LR_syn {lr_syn:.6f}{m}")

    return hist


def run():
    print("=" * 70)
    print("PLASTIC: Grokking Experiment (Prediction 2 / Theorem 3)")
    print("=" * 70)

    P = 59
    D_EMBED = 64
    D_HIDDEN = 128
    N_EPOCHS = 8000
    TRAIN_FRAC = 0.5
    WEIGHT_DECAYS = [1.0, 1.5]
    SEEDS = [0, 1]

    all_results = {}

    for wd in WEIGHT_DECAYS:
        key = f"wd={wd}"
        all_results[key] = []
        print(f"\n=== Weight Decay = {wd} ===")
        for s in SEEDS:
            t0 = time.time()
            h = run_seed(P, D_EMBED, D_HIDDEN, wd, N_EPOCHS, TRAIN_FRAC, s)
            dt = time.time() - t0
            s_str = f"grokked ep {h['grok_epoch']}" if h["grokked"] else "DID NOT grok"
            print(f"  --> Seed {s}: {s_str} ({dt:.0f}s)")
            all_results[key].append(h)

    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for wd in WEIGHT_DECAYS:
        key = f"wd={wd}"
        for h in all_results[key]:
            ge = h["grok_epoch"] if h["grokked"] else None
            lr_early = np.mean(h["lr_syn"][:50])
            lr_pre = None
            if h["grokked"] and ge > 100:
                lr_pre = np.mean(h["lr_syn"][ge - 100:ge])
            print(f"  {key:>6s} seed={h['epoch'][0] if 'epoch' in h else 0}: "
                  f"grok={ge}, early_LR={lr_early:.6f}" +
                  (f", pre_grok_LR={lr_pre:.6f}" if lr_pre else ""))

    # Cross-WD: does lower LR_syn predict longer/no grokking?
    print(f"\n  Cross-WD Summary:")
    for wd in WEIGHT_DECAYS:
        key = f"wd={wd}"
        grokked = sum(1 for h in all_results[key] if h["grokked"])
        mean_lr = np.mean([np.mean(h["lr_syn"][:50]) for h in all_results[key]])
        print(f"    wd={wd:.1f}: grokked {grokked}/{len(SEEDS)}, "
              f"mean early LR_syn={mean_lr:.6f}")

    os.makedirs("results", exist_ok=True)
    with open("results/grokking_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to results/grokking_results.json")
    print("Done.")


if __name__ == "__main__":
    run()
