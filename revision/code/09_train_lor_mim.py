"""
09_train_lor_mim.py — LOR-MIM + FT-Transformer (Softmax, FocalLoss) 训练

完整流程：
  1. 加载训练/验证/测试数据
  2. 在训练集上构建 V-Amount 亲和图 → 谱聚类分组
  3. 初始化 LOR-MIM + FTT+Softmax 联合模型
  4. 端到端训练（LOR-MIM gate + mixing + FTT 全部可训练）
  5. 评估：AUPRC, 注意力std, 门控分布, 混合矩阵偏离度
  6. 与 baseline (FTT+Softmax, std=0.00125, AUPRC=0.9126) 对比
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score

from config import (
    PROCESSED_DIR, MODEL_DIR, FIGURE_DIR, RESULT_DIR, DEVICE,
    FEATURE_NAMES, INPUT_DIM, SEED,
    BATCH_SIZE, MAX_EPOCHS, EARLY_STOP_PATIENCE,
    FOCAL_ALPHA, FOCAL_GAMMA, FTT_D_MODEL, FTT_N_HEADS, FTT_N_LAYERS, FTT_FFN_DIM,
)
from models.ft_transformer import FTTransformer
from models.lor_mim import LORMIM, build_affinity_groups
from training.losses import FocalLoss
from training.metrics import compute_all_metrics, find_best_threshold
from training.trainer import Trainer

# ── 可复现 ──
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── LOR-MIM 超参 ──
N_GROUPS = 5          # 谱聚类分组数
GATE_HIDDEN = 16      # 门控 MLP 隐层
INIT_NOISE = 0.02     # 混合矩阵初始化噪声

# ═══════════════════════════════════════════════════════════════
# 联合模型: LOR-MIM → FT-Transformer
# ═══════════════════════════════════════════════════════════════

class LORMIM_FTT(nn.Module):
    """LOR-MIM 特征混合 + FT-Transformer 分类。"""
    def __init__(self, input_dim, group_assignments, gate_hidden=16, init_noise=0.02):
        super().__init__()
        self.lor_mim = LORMIM(input_dim - 1, group_assignments, gate_hidden, init_noise)
        self.ftt = FTTransformer(input_dim, use_sparsemax=False)
        self._last_gate = None  # 缓存最近一次门控值

    def forward(self, x):
        x_transformed, gate = self.lor_mim(x)
        self._last_gate = gate.detach()
        return self.ftt(x_transformed)

    def get_attention_weights(self, x):
        x_transformed, _ = self.lor_mim(x)
        return self.ftt.get_attention_weights(x_transformed)

    def get_last_gate(self):
        return self._last_gate


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("LOR-MIM + FT-Transformer 训练")
    print("=" * 60)

    # ── 加载数据 ──
    print("\n[1/5] Loading data...")
    train_X, train_y = torch.load(os.path.join(PROCESSED_DIR, "train.pt"), weights_only=True)
    val_X, val_y = torch.load(os.path.join(PROCESSED_DIR, "val.pt"), weights_only=True)
    test_X, test_y = torch.load(os.path.join(PROCESSED_DIR, "test.pt"), weights_only=True)
    print(f"  Train: {train_X.shape}, Val: {val_X.shape}, Test: {test_X.shape}")

    # ── 构建亲和图 ──
    print(f"\n[2/5] Building affinity graph (K={N_GROUPS})...")
    v_train = train_X[:, :28].numpy()
    amount_train = train_X[:, 28].numpy()

    group_assignments = build_affinity_groups(
        v_train, amount_train, n_groups=N_GROUPS, random_state=SEED
    )

    # ── 初始化模型 ──
    print("\n[3/5] Building LOR-MIM + FTT model...")
    model = LORMIM_FTT(
        INPUT_DIM, group_assignments,
        gate_hidden=GATE_HIDDEN, init_noise=INIT_NOISE
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    lor_mim_params = sum(p.numel() for p in model.lor_mim.parameters())
    ftt_params = sum(p.numel() for p in model.ftt.parameters())
    print(f"  Total params: {total_params:,}")
    print(f"  LOR-MIM params: {lor_mim_params:,} ({lor_mim_params/total_params*100:.2f}%)")
    print(f"  FTT params: {ftt_params:,}")
    print(f"  Group sizes: {model.lor_mim.get_group_sizes()}")

    # ── 训练 ──
    print("\n[4/5] Training...")
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8
    )
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

    trainer = Trainer(
        model=model,
        loss_fn=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        model_name="LOR-MIM-FTT",
        patience=EARLY_STOP_PATIENCE,
    )

    trainer.fit(train_loader, val_loader, max_epochs=MAX_EPOCHS)
    trainer.load_best()

    # ── 评估 ──
    print("\n[5/5] Evaluating...")
    model.eval()

    # 预测
    @torch.no_grad()
    def get_scores(m, X):
        B = 2048
        scores_list = []
        for i in range(0, len(X), B):
            batch = X[i:i+B].to(DEVICE)
            s = torch.sigmoid(m(batch).squeeze(-1)).cpu()
            scores_list.append(s)
        return torch.cat(scores_list).numpy()

    test_scores = get_scores(model, test_X)
    val_scores = get_scores(model, val_X)
    test_y_np = test_y.numpy()
    val_y_np = val_y.numpy()

    auprc = average_precision_score(test_y_np, test_scores)
    best_threshold, _ = find_best_threshold(val_y_np, val_scores)
    metrics = compute_all_metrics(test_y_np, test_scores, best_threshold)

    # 注意力权重
    n_attn_samples = 200
    idx = np.random.RandomState(SEED).choice(len(test_X), n_attn_samples, replace=False)
    X_sub = test_X[idx].to(DEVICE)
    attn = model.get_attention_weights(X_sub).cpu().numpy()
    attn_std = float(np.std(attn))
    attn_mean = float(np.mean(attn))
    uniform_baseline = 1.0 / 29

    # 门控分布
    @torch.no_grad()
    def get_gate_stats(m, X):
        B = 2048
        gates = []
        for i in range(0, len(X), B):
            batch = X[i:i+B].to(DEVICE)
            _, g = m.lor_mim(batch)
            gates.append(g.cpu())
        return torch.cat(gates, dim=0).numpy()

    test_gates = get_gate_stats(model, test_X)
    gate_means = test_gates.mean(axis=0)

    # 混合矩阵偏离度
    mixing_div = model.lor_mim.get_mixing_divergence()

    # ── 打印结果 ──
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  AUPRC:        {auprc:.4f}")
    print(f"  F1:           {metrics['F1']:.4f}")
    print(f"  AttnStd:       {attn_std:.6f}")
    print(f"  AttnMean:      {attn_mean:.6f}")
    print(f"  Std/Uniform:   {attn_std / uniform_baseline:.4f}")
    print(f"\n  Baseline (FTT+Softmax):")
    print(f"    AUPRC=0.9126, AttnStd=0.00125, Std/Uniform=0.036")
    print(f"\n  Gate means:    {gate_means.round(4)}")
    print(f"  Mixing div:    { {k: round(v, 4) for k, v in mixing_div.items()} }")

    # ── 保存 ──
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "lor_mim_ftt_best.pt"))
    print(f"\n  Model saved to: lor_mim_ftt_best.pt")

    return auprc, attn_std, attn_std / uniform_baseline


if __name__ == "__main__":
    main()
