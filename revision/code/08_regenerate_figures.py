"""
Regenerate all paper figures including LOR-MIM model.
Output: revision/paper/figures/
"""
import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (precision_recall_curve, roc_curve, roc_auc_score, average_precision_score)
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (PROCESSED_DIR, MODEL_DIR, FIGURE_DIR, DEVICE, FEATURE_NAMES, INPUT_DIM)
from models.ft_transformer import FTTransformer
from models.lor_mim import LORMIM, build_affinity_groups

PAPER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figures")
os.makedirs(PAPER_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "savefig.dpi": 200, "savefig.bbox": "tight"})

C_SOFTMAX = "#e67e22"
C_SPARSEMAX = "#2980b9"
C_BCE = "#27ae60"
C_LOR_MIM = "#8e44ad"
C_LOR_SPARSE = "#16a085"

# Load data
print("Loading data...")
test_X, test_y = torch.load(os.path.join(PROCESSED_DIR, "test.pt"), weights_only=True)
train_X, train_y = torch.load(os.path.join(PROCESSED_DIR, "train.pt"), weights_only=True)

# Build groups for LOR-MIM
ga = build_affinity_groups(train_X[:, :28].numpy(), train_X[:, 28].numpy(), n_groups=5, random_state=42)

# Load models
class LORMIM_FTT(nn.Module):
    def __init__(self, d_in, ga, use_sm=False):
        super().__init__()
        self.lor_mim = LORMIM(d_in-1, ga)
        self.ftt = FTTransformer(d_in, use_sparsemax=use_sm)
    def forward(self, x):
        x_t, _ = self.lor_mim(x); return self.ftt(x_t)
    def get_attention_weights(self, x):
        x_t, _ = self.lor_mim(x); return self.ftt.get_attention_weights(x_t)

print("Loading models...")
models = {}
for name, ckpt, use_sm in [("FTT+Softmax", "ftt_softmax_best.pt", False),
                             ("FTT+Sparsemax", "ftt_sparsemax_best.pt", True),
                             ("FTT+BCE", "ftt_softmax_bce_best.pt", False)]:
    m = FTTransformer(INPUT_DIM, use_sparsemax=use_sm).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(MODEL_DIR, ckpt), map_location=DEVICE, weights_only=True))
    m.eval()
    models[name] = m

m_lor = LORMIM_FTT(INPUT_DIM, ga, use_sm=False).to(DEVICE)
m_lor.load_state_dict(torch.load(os.path.join(MODEL_DIR, "LOR-MIM-FTT_best.pt"), map_location=DEVICE, weights_only=True))
m_lor.eval()
models["FTT+LORMIM"] = m_lor

m_ls = LORMIM_FTT(INPUT_DIM, ga, use_sm=True).to(DEVICE)
m_ls.load_state_dict(torch.load(os.path.join(MODEL_DIR, "lor_mim_sparsemax_best.pt"), map_location=DEVICE, weights_only=True))
m_ls.eval()
models["FTT+LORMIM+Sparsemax"] = m_ls

colors = {"FTT+Softmax": C_SOFTMAX, "FTT+Sparsemax": C_SPARSEMAX,
          "FTT+BCE": C_BCE, "FTT+LORMIM": C_LOR_MIM,
          "FTT+LORMIM+Sparsemax": C_LOR_SPARSE}

# Get predictions
@torch.no_grad()
def get_scores(m, X):
    B=2048; s=[]
    for i in range(0,len(X),B):
        s.append(torch.sigmoid(m(X[i:i+B].to(DEVICE)).squeeze(-1)).cpu())
    return torch.cat(s).numpy()

print("Computing scores...")
all_scores = {}
for name, m in models.items():
    all_scores[name] = get_scores(m, test_X)
    auprc = average_precision_score(test_y.numpy(), all_scores[name])
    print(f"  {name}: AUPRC={auprc:.4f}")

# ================================================================
# Figure: Attention Weight Distribution (4 models)
# ================================================================
print("\n[Fig] Attention weights...")
@torch.no_grad()
def get_attn(m, X, n=200):
    idx = np.random.RandomState(42).choice(len(X), min(n, len(X)), replace=False)
    return m.get_attention_weights(X[idx].to(DEVICE)).cpu().numpy()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plot_order = ["FTT+Softmax", "FTT+Sparsemax", "FTT+LORMIM", "FTT+LORMIM+Sparsemax"]
for ax, name in zip(axes.flat, plot_order):
    attn = get_attn(models[name], test_X)
    sorted_idx = np.argsort(attn)[::-1]
    c = colors[name]
    bar_colors = [c if attn[i] > 1e-4 else "#cccccc" for i in sorted_idx]
    ax.bar(range(29), attn[sorted_idx], color=bar_colors, edgecolor="white", linewidth=0.3)
    ax.axhline(y=1/29, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(29))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in sorted_idx], rotation=90, fontsize=5)
    ax.set_ylabel("Weight", fontsize=9)
    asd = np.std(attn); ub = 1/29
    ax.set_title(f"{name}\nstd={asd:.4f}, std/uniform={asd/ub:.3f}", fontsize=10, fontweight="bold")

fig.suptitle("Attention Weight Distribution: CLS → 29 Features", fontsize=13, fontweight="bold")
plt.tight_layout()
for fmt in ["pdf","png"]:
    fig.savefig(os.path.join(FIGURE_DIR, f"ftt_attention.{fmt}"))
    shutil.copy(os.path.join(FIGURE_DIR, f"ftt_attention.{fmt}"), os.path.join(PAPER_DIR, f"ftt_attention.{fmt}"))
plt.close()
print("  -> ftt_attention.pdf")

# ================================================================
# Figure: PR Curves (Softmax vs Sparsemax vs LORMIM)
# ================================================================
print("[Fig] PR curves...")
fig, ax = plt.subplots(figsize=(8, 6))
for name in ["FTT+Softmax", "FTT+Sparsemax", "FTT+BCE", "FTT+LORMIM", "FTT+LORMIM+Sparsemax"]:
    s = all_scores[name]
    p, r, _ = precision_recall_curve(test_y.numpy(), s)
    auprc = average_precision_score(test_y.numpy(), s)
    ax.plot(r, p, color=colors[name], linewidth=1.5, label=f"{name} (AUPRC={auprc:.4f})")
ax.axhline(y=test_y.numpy().mean(), color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves", fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
for fmt in ["pdf","png"]:
    fig.savefig(os.path.join(FIGURE_DIR, f"pr_curves.{fmt}"))
    shutil.copy(os.path.join(FIGURE_DIR, f"pr_curves.{fmt}"), os.path.join(PAPER_DIR, f"pr_curves.{fmt}"))
plt.close()
print("  -> pr_curves.pdf")

# ================================================================
# Figure: ROC Curves
# ================================================================
print("[Fig] ROC curves...")
fig, ax = plt.subplots(figsize=(8, 6))
for name in ["FTT+Softmax", "FTT+Sparsemax", "FTT+BCE", "FTT+LORMIM", "FTT+LORMIM+Sparsemax"]:
    s = all_scores[name]
    fpr, tpr, _ = roc_curve(test_y.numpy(), s)
    auc = roc_auc_score(test_y.numpy(), s)
    ax.plot(fpr, tpr, color=colors[name], linewidth=1.5, label=f"{name} (AUC={auc:.4f})")
ax.plot([0,1],[0,1],"k--",linewidth=0.8,alpha=0.5)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC Curves", fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
for fmt in ["pdf","png"]:
    fig.savefig(os.path.join(FIGURE_DIR, f"roc_curves.{fmt}"))
    shutil.copy(os.path.join(FIGURE_DIR, f"roc_curves.{fmt}"), os.path.join(PAPER_DIR, f"roc_curves.{fmt}"))
plt.close()
print("  -> roc_curves.pdf")

# ================================================================
# Figure: Calibration (Softmax vs Sparsemax)
# ================================================================
print("[Fig] Calibration...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, c) in zip(axes, [("FTT+Softmax", C_SOFTMAX), ("FTT+Sparsemax", C_SPARSEMAX)]):
    s = all_scores[name]
    pt, pp = calibration_curve(test_y.numpy(), s, n_bins=10)
    ece = np.average(np.abs(pt-pp), weights=np.histogram(s,bins=10,range=(0,1))[0]/len(s))
    ax.bar(pp, pt, width=0.08, color=c, alpha=0.7, edgecolor="white")
    ax.plot([0,1],[0,1],"k--",linewidth=0.8)
    ax.set_title(f"{name} (ECE={ece:.4f})", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
    ax.grid(True, alpha=0.2)
fig.suptitle("Reliability Diagrams", fontweight="bold")
plt.tight_layout()
for fmt in ["pdf","png"]:
    fig.savefig(os.path.join(FIGURE_DIR, f"calibration_curves.{fmt}"))
    shutil.copy(os.path.join(FIGURE_DIR, f"calibration_curves.{fmt}"), os.path.join(PAPER_DIR, f"calibration_curves.{fmt}"))
plt.close()
print("  -> calibration_curves.pdf")

# ================================================================
# Figure: Permutation Importance (Softmax vs Sparsemax)
# ================================================================
print("[Fig] Permutation importance...")
@torch.no_grad()
def perm_imp(m, X, y, n_repeats=5):
    base_s = get_scores(m, X)
    base_auprc = average_precision_score(y.numpy(), base_s)
    imps = np.zeros((INPUT_DIM, n_repeats))
    Xn = X.numpy()
    for fi in tqdm(range(INPUT_DIM), desc="  Permuting"):
        for r in range(n_repeats):
            Xp = Xn.copy(); np.random.shuffle(Xp[:, fi])
            sp = get_scores(m, torch.from_numpy(Xp).float())
            imps[fi, r] = base_auprc - average_precision_score(y.numpy(), sp)
    return imps.mean(1), imps.std(1)

imp_sm, std_sm = perm_imp(models["FTT+Softmax"], test_X, test_y)
imp_sp, std_sp = perm_imp(models["FTT+Sparsemax"], test_X, test_y)
imp_lm, std_lm = perm_imp(models["FTT+LORMIM"], test_X, test_y)

fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(INPUT_DIM); w = 0.28
ax.bar(x-w, imp_sm, w, color=C_SOFTMAX, alpha=0.85, label="FTT+Softmax")
ax.bar(x, imp_sp, w, color=C_SPARSEMAX, alpha=0.85, label="FTT+Sparsemax")
ax.bar(x+w, imp_lm, w, color=C_LOR_MIM, alpha=0.85, label="FTT+LORMIM")
ax.set_xticks(x); ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=7)
ax.set_ylabel("AUPRC Decrease"); ax.legend(fontsize=9)
r12=np.corrcoef(imp_sm,imp_sp)[0,1]; r13=np.corrcoef(imp_sm,imp_lm)[0,1]
ax.set_title(f"Permutation Importance (r(Soft,Sparse)={r12:.3f}, r(Soft,LORMIM)={r13:.3f})", fontweight="bold")
ax.grid(True, alpha=0.2, axis="y")
plt.tight_layout()
for fmt in ["pdf","png"]:
    fig.savefig(os.path.join(FIGURE_DIR, f"permutation_importance.{fmt}"))
    shutil.copy(os.path.join(FIGURE_DIR, f"permutation_importance.{fmt}"), os.path.join(PAPER_DIR, f"permutation_importance.{fmt}"))
plt.close()
print("  -> permutation_importance.pdf")

print("\nAll figures saved to:", PAPER_DIR)
