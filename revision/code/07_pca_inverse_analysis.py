"""
PCA逆正交化验证：Amount作为语义锚点。
假设：如果注意力塌缩来自PCA→那么非PCA特征(Amount)的嵌入应具有更高方向多样性。

分析维度：
1. 嵌入范数对比：Amount vs V1-V28
2. 余弦相似度：Amount与各V特征 vs V特征之间
3. 注意力权重：Amount被分配了多少注意力
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PROCESSED_DIR, MODEL_DIR, FIGURE_DIR, DEVICE, FEATURE_NAMES
from models.ft_transformer import FTTransformer

# ── 加载模型 ──
input_dim = 29
model = FTTransformer(input_dim, use_sparsemax=False).to(DEVICE)
model.load_state_dict(torch.load(
    os.path.join(MODEL_DIR, "ftt_softmax_best.pt"), map_location=DEVICE, weights_only=True
))
model.eval()

# ── 提取嵌入权重 ──
W = model.impl.tokenizer.W.detach().cpu().numpy()  # (29, 64)
bias = model.impl.tokenizer.bias.detach().cpu().numpy()

# ── 1. 嵌入范数分析 ──
norms = np.linalg.norm(W, axis=1)  # (29,)
amount_idx = FEATURE_NAMES.index("Amount")
v_indices = [i for i, name in enumerate(FEATURE_NAMES) if name.startswith("V")]

print("=" * 60)
print("1. 嵌入范数对比")
print("=" * 60)
print(f"  Amount (非PCA):  ||W|| = {norms[amount_idx]:.4f}")
print(f"  V1-V28 均值:     ||W|| = {np.mean(norms[v_indices]):.4f} ± {np.std(norms[v_indices]):.4f}")
print(f"  V1-V28 范围:     [{np.min(norms[v_indices]):.4f}, {np.max(norms[v_indices]):.4f}]")
print(f"  Amount排名:       {np.argsort(norms)[::-1].tolist().index(amount_idx) + 1}/29")

# ── 2. 余弦相似度分析 ──
# 归一化嵌入方向
W_norm = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
cos_sim = W_norm @ W_norm.T  # (29, 29) 余弦相似度矩阵

# Amount vs V features
amount_v_cos = cos_sim[amount_idx, v_indices]  # Amount <-> 各V特征的余弦相似度
# V features pairwise (upper triangle excluding diagonal)
v_cos_pairs = []
for i in range(len(v_indices)):
    for j in range(i + 1, len(v_indices)):
        v_cos_pairs.append(cos_sim[v_indices[i], v_indices[j]])

print(f"\n2. 余弦相似度对比")
print(f"  Amount<->V特征 均值: {np.mean(amount_v_cos):.4f} ± {np.std(amount_v_cos):.4f}")
print(f"  V<->V特征之间 均值: {np.mean(v_cos_pairs):.4f} ± {np.std(v_cos_pairs):.4f}")
print(f"  Amount最相似于:   {FEATURE_NAMES[v_indices[np.argmax(amount_v_cos)]]} (cos={np.max(amount_v_cos):.4f})")
print(f"  Amount最不相似于: {FEATURE_NAMES[v_indices[np.argmin(amount_v_cos)]]} (cos={np.min(amount_v_cos):.4f})")

# ── 3. 注意力权重提取 ──
test_X, test_y = torch.load(os.path.join(PROCESSED_DIR, "test.pt"), weights_only=True)
X_sub = test_X[:200].to(DEVICE)
attn = model.get_attention_weights(X_sub)
if attn is not None:
    attn_np = attn.cpu().numpy()
    amount_attn = attn_np[amount_idx]
    v_attns = attn_np[v_indices]
    print(f"\n3. 注意力权重对比")
    print(f"  Amount注意力:  {amount_attn:.5f}")
    print(f"  V特征注意力均值: {np.mean(v_attns):.5f} ± {np.std(v_attns):.5f}")
    print(f"  Amount排名: {np.argsort(attn_np)[::-1].tolist().index(amount_idx) + 1}/29")

# ── 4. 综合判断 ──
print(f"\n{'='*60}")
print("PCA逆正交化验证结论")
print(f"{'='*60}")

amount_norm_rank = np.argsort(norms)[::-1].tolist().index(amount_idx)
amount_cos_diff = np.mean(v_cos_pairs) - np.mean(amount_v_cos)

print(f"  Amount嵌入范数排名: {amount_norm_rank + 1}/29")
print(f"  V<->V平均余弦 - Amount<->V平均余弦 = {amount_cos_diff:.4f}")

if amount_cos_diff > 0.02:
    print(f"  ✅ Amount嵌入方向确实比V特征更独立——支撑'PCA是塌缩根因'")
elif amount_cos_diff > 0:
    print(f"  ⚠️ Amount略有独立趋势但效应微弱——弱支撑")
else:
    print(f"  ❌ Amount嵌入方向并无更独立——不支持'PCA是塌缩根因'")

# ── 5. 可视化 ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# 左：嵌入余弦相似度热力图
ax = axes[0]
# 重排索引：Amount放在第一个
reorder = [amount_idx] + [i for i in range(29) if i != amount_idx]
reorder_labels = [FEATURE_NAMES[i] for i in reorder]
sub_matrix = cos_sim[np.ix_(reorder, reorder)]
im = ax.imshow(sub_matrix, cmap="RdBu_r", vmin=-0.5, vmax=1.0, aspect="auto")
ax.set_xticks(range(29))
ax.set_yticks(range(29))
ax.set_xticklabels(reorder_labels, rotation=90, fontsize=6)
ax.set_yticklabels(reorder_labels, fontsize=6)
ax.set_title("Feature Embedding Cosine Similarity\n(Amount at position 0)", fontsize=11, fontweight="bold")
# 高亮Amount行/列
ax.axhline(y=0.5, color="orange", linewidth=2, linestyle="--", alpha=0.7)
ax.axvline(x=0.5, color="orange", linewidth=2, linestyle="--", alpha=0.7)
plt.colorbar(im, ax=ax, label="Cosine Similarity", shrink=0.8)

# 右：嵌入范数条形图
ax = axes[1]
sorted_idx = np.argsort(norms)[::-1]
colors = ["#e74c3c" if FEATURE_NAMES[i] == "Amount" else "#3498db" for i in sorted_idx]
ax.bar(range(29), norms[sorted_idx], color=colors, edgecolor="white")
ax.set_xticks(range(29))
ax.set_xticklabels([FEATURE_NAMES[i] for i in sorted_idx], rotation=90, fontsize=7)
ax.set_ylabel("Embedding L2 Norm", fontsize=11)
ax.set_title("Feature Embedding Norms\n(Amount highlighted)", fontsize=11, fontweight="bold")
ax.axhline(y=np.mean(norms[v_indices]), color="blue", linestyle="--", alpha=0.5, label=f"V mean={np.mean(norms[v_indices]):.3f}")
ax.legend(fontsize=9)

fig.suptitle("PCA Inverse Orthogonalization: Amount as Semantic Anchor", fontsize=13, fontweight="bold")
plt.tight_layout()
outpath = os.path.join(FIGURE_DIR, "pca_inverse_amount_anchor.pdf")
fig.savefig(outpath, dpi=200, bbox_inches="tight")
print(f"\nFigure saved to: {outpath}")
plt.close()
