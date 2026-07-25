"""
LOR-MIM: Local Orthogonality Relaxation via Mutual Information-gated Mixup

打破 PCA 匿名化特征空间的全局对称性：
1. 基于 V 特征与 Amount 的互信息构建亲和图 → 谱聚类分组
2. 门控 MLP (Amount → K 组权重) 输出样本级 group weight α_k
3. 每组一个可学习混合矩阵 M_k（初始化为近单位阵），组内特征线性混合
4. 混合后还原原始特征顺序，拼接 Amount 送入 FT-Transformer

直观：同一 V 特征在不同 Amount 区间被卷入不同的混合模式，
     打破 PCA 全局对称 → 注意力被迫学习差异化权重。
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import SpectralClustering
from sklearn.metrics import mutual_info_score
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# 特征亲和图构建（离线，训练前执行一次）
# ═══════════════════════════════════════════════════════════════

def build_affinity_groups(v_features: np.ndarray, amount: np.ndarray,
                          n_groups: int = 4, random_state: int = 42):
    """
    基于 V 特征与 Amount 的互信息构建亲和图，谱聚类分组。

    Args:
        v_features: (N, 28) V1-V28
        amount: (N,) Amount
        n_groups: 分组数 K
        random_state: 随机种子

    Returns:
        group_assignments: list of lists, 每个子列表包含该组的特征索引 (0-based)
    """
    n_v = v_features.shape[1]

    # 计算每个 V 特征与 Amount 的互信息（各分 20 个等频箱离散化）
    affinity = np.zeros(n_v)
    for j in range(n_v):
        v_binned = pd.cut(v_features[:, j], bins=20, labels=False)
        a_binned = pd.cut(amount, bins=20, labels=False)
        mask = ~(np.isnan(v_binned) | np.isnan(a_binned))
        affinity[j] = mutual_info_score(v_binned[mask], a_binned[mask]) if mask.sum() > 100 else 0.0

    # 归一化
    if affinity.max() > affinity.min():
        affinity_norm = (affinity - affinity.min()) / (affinity.max() - affinity.min())
    else:
        affinity_norm = np.ones(n_v) / n_v

    # 亲和矩阵：A[i,j] = 1 - |affinity[i] - affinity[j]|
    A = np.zeros((n_v, n_v))
    for i in range(n_v):
        for j in range(n_v):
            A[i, j] = 1.0 - abs(affinity_norm[i] - affinity_norm[j])

    # 谱聚类
    clustering = SpectralClustering(
        n_clusters=n_groups, affinity='precomputed',
        random_state=random_state, assign_labels='kmeans'
    )
    labels = clustering.fit_predict(A)

    group_assignments = []
    for k in range(n_groups):
        members = [int(j) for j in range(n_v) if labels[j] == k]
        if members:
            group_assignments.append(members)
    group_assignments.sort(key=len, reverse=True)

    print(f"  Affinity groups: {[len(g) for g in group_assignments]} features per group")
    print(f"  Affinity range: [{affinity.min():.4f}, {affinity.max():.4f}]")
    return group_assignments


# ═══════════════════════════════════════════════════════════════
# LOR-MIM 模块：动态特征混合
# ═══════════════════════════════════════════════════════════════

class LORMIM(nn.Module):
    """
    局部正交松弛——互信息门控混合。

    Args:
        n_v_features: V 特征数量 (28)
        group_assignments: 谱聚类分组结果
        gate_hidden: 门控 MLP 隐层维度
        init_noise: 混合矩阵初始化的非对角噪声标准差
    """

    def __init__(self, n_v_features: int, group_assignments: list,
                 gate_hidden: int = 16, init_noise: float = 0.05):
        super().__init__()
        self.n_v = n_v_features
        self.group_assignments = group_assignments
        self.n_groups = len(group_assignments)

        # ── 逆排列：拼接后还原原始 V1..V28 顺序 ──
        flat = []
        for g in group_assignments:
            flat.extend(g)
        self.register_buffer('_inv_perm',
                             torch.tensor(sorted(range(len(flat)), key=lambda i: flat[i])))

        # ── 门控网络: Amount → hidden → K (softmax) ──
        self.gate = nn.Sequential(
            nn.Linear(1, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, self.n_groups),
            nn.Softmax(dim=-1),
        )

        # ── 每组一个混合矩阵 M_k（初始化为 I + 小噪声）──
        self.mixing_matrices = nn.ParameterList()
        for group_members in group_assignments:
            d_k = len(group_members)
            I = torch.eye(d_k)
            noise = torch.randn(d_k, d_k) * init_noise
            self.mixing_matrices.append(nn.Parameter(I + noise))

    def forward(self, x):
        """
        Args:
            x: (B, 29) [V1..V28, Amount]
        Returns:
            x_out: (B, 29) 变换后特征（原始顺序），Amount 不变
            gate_weights: (B, K) 门控权重
        """
        v_feat = x[:, :self.n_v]     # (B, 28)
        amount = x[:, self.n_v:]     # (B, 1)

        alpha = self.gate(amount)    # (B, K)

        parts = []
        for k, (group_idx, M_k) in enumerate(zip(self.group_assignments, self.mixing_matrices)):
            v_group = v_feat[:, group_idx]       # (B, d_k)
            v_mixed = v_group @ M_k.T             # (B, d_k)
            alpha_k = alpha[:, k:k+1]             # (B, 1)
            parts.append(alpha_k * v_mixed)

        v_cat = torch.cat(parts, dim=-1)          # (B, 28) 分组序
        v_out = v_cat[:, self._inv_perm]           # (B, 28) 还原 V1..V28 序
        x_out = torch.cat([v_out, amount], dim=-1) # (B, 29)

        return x_out, alpha

    def get_group_sizes(self):
        return [len(g) for g in self.group_assignments]

    def get_mixing_divergence(self):
        divs = {}
        for k, (group, M) in enumerate(zip(self.group_assignments, self.mixing_matrices)):
            d_k = len(group)
            I = torch.eye(d_k, device=M.device)
            divs[f"group_{k}"] = torch.norm(M - I, p='fro').item()
        return divs
