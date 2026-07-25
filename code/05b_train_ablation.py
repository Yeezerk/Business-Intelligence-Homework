"""
不平衡处理策略消融实验——在相同 TabNet 骨架上对比 4 种方案。
答辩要点：
  4 种策略（控制变量：相同架构、优化器、调度器、种子）：
    1. Focal Loss γ=2.0（论文主方案）：α 按架构定制
    2. ClassWeight：BCEWithLogitsLoss 的 pos_weight = 反频率加权
    3. SMOTE 1:1 过采样 + Plain CE：生成合成欺诈样本至正负平衡
    4. Plain CE（不做任何处理下限对照）

  实验结果预期：
    - Focal Loss > ClassWeight > SMOTE > Plain CE
    - 原因：Focal Loss 的 γ 能动态降低易分类样本权重，比固定权重更精细
    - SMOTE 的问题：合成样本可能不真实（PCA 空间中插值可能产生不合理样本）
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE

from config import (
    PROCESSED_DIR, MODEL_DIR, RESULT_DIR, DEVICE, SEED,
    BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    FOCAL_GAMMA, FOCAL_ALPHA,
)
from training.trainer import Trainer
from training.losses import FocalLoss
from training.metrics import compute_all_metrics, find_best_threshold
from models.tabnet import TabNet

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_processed():
    train_X, train_y = torch.load(os.path.join(PROCESSED_DIR, "train.pt"), weights_only=True)
    val_X, val_y = torch.load(os.path.join(PROCESSED_DIR, "val.pt"), weights_only=True)
    test_X, test_y = torch.load(os.path.join(PROCESSED_DIR, "test.pt"), weights_only=True)
    return train_X, train_y, val_X, val_y, test_X, test_y


def create_loaders(train_X, train_y, val_X, val_y, test_X, test_y):
    train_ds = TensorDataset(train_X, train_y)
    val_ds = TensorDataset(val_X, val_y)
    test_ds = TensorDataset(test_X, test_y)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False)
    return train_loader, val_loader, test_loader


def apply_smote(train_X, train_y):
    """Apply SMOTE oversampling to balance training set (1:1 ratio)."""
    sm = SMOTE(random_state=SEED)
    X_resampled, y_resampled = sm.fit_resample(train_X.numpy(), train_y.numpy())
    print(f"  SMOTE: {train_y.sum().item():.0f} fraud → {y_resampled.sum():.0f} fraud "
          f"(total: {len(y_resampled):,} samples)")
    return torch.tensor(X_resampled, dtype=torch.float32), torch.tensor(y_resampled, dtype=torch.float32)


def train_ablation(name, train_X, train_y, val_X, val_y, test_X, test_y, loss_fn):
    """Train TabNet with the given loss function and evaluate."""
    print(f"\n{'='*60}")
    print(f"Ablation: {name}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader = create_loaders(
        train_X, train_y, val_X, val_y, test_X, test_y
    )

    model = TabNet(train_X.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        model_name=f"ablation_{name}",
    )
    trainer.fit(train_loader, val_loader, max_epochs=MAX_EPOCHS)

    # Evaluate on test set
    trainer.load_best()
    test_scores, test_labels = trainer.predict(test_loader)
    val_scores, val_labels = trainer.predict(val_loader)
    best_threshold, _ = find_best_threshold(val_labels, val_scores)

    metrics = compute_all_metrics(test_labels, test_scores, threshold=best_threshold)
    metrics["Strategy"] = name

    print(f"\n{name} Test Results (threshold={best_threshold:.3f}):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics


def main():
    print(f"Device: {DEVICE}")
    train_X, train_y, val_X, val_y, test_X, test_y = load_processed()
    print(f"Train: {len(train_y):,} samples, Val: {len(val_y):,}, Test: {len(test_y):,} (input_dim={train_X.shape[1]})")
    print(f"Fraud ratio: train={train_y.sum().item()/len(train_y):.4%}, "
          f"val={val_y.sum().item()/len(val_y):.4%}, test={test_y.sum().item()/len(test_y):.4%}")

    # Compute class-balance statistics for loss functions
    n_pos = int(train_y.sum().item())
    n_neg = len(train_y) - n_pos
    fraud_ratio = n_pos / (n_pos + n_neg)
    alpha = FOCAL_ALPHA  # Use default 0.25 (not class-balance derived)
    pos_weight = n_neg / n_pos

    print(f"\nClass stats: {n_pos} fraud, {n_neg} legit")
    print(f"Focal alpha={alpha:.4f}, pos_weight={pos_weight:.2f}")

    results = []

    # ── 1. Focal Loss (γ=2.0) ──────────────────────────────────────
    focal_loss = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA)
    results.append(train_ablation(
        "FocalLoss", train_X, train_y, val_X, val_y, test_X, test_y, focal_loss
    ))

    # ── 2. Class Weight (inverse-frequency pos_weight) ──────────────
    class_weight_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))
    results.append(train_ablation(
        "ClassWeight", train_X, train_y, val_X, val_y, test_X, test_y, class_weight_loss
    ))

    # ── 3. SMOTE (1:1 oversampling) + Plain CE ─────────────────────
    smote_X, smote_y = apply_smote(train_X, train_y)
    plain_ce = nn.BCEWithLogitsLoss()
    results.append(train_ablation(
        "SMOTE", smote_X, smote_y, val_X, val_y, test_X, test_y, plain_ce
    ))

    # ── 4. Plain CE (no imbalance handling — baseline) ─────────────
    results.append(train_ablation(
        "PlainCE", train_X, train_y, val_X, val_y, test_X, test_y, plain_ce
    ))

    # ── Save results ──────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULT_DIR, "ablation_imbalance.csv"), index=False)
    print(f"\nAblation results saved to {RESULT_DIR}/ablation_imbalance.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
