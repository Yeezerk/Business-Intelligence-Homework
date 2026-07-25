"""
训练神经网络基线：TabNet → FT-Transformer → DeepFM。

关键设计决策（答辩必问）：
  1. 种子卫生 reset_seed(SEED)：每个模型训练前独立重置所有随机状态
     → 踩了 pytorch-tabnet 种子污染 bug 后的教训
     → 确保模型间比较公平——差异只来自架构
  2. Focal Loss α 按架构定制：
     → FTT (Sparsemax) α=0.25：注意力已有特征选择，梯度压力小，需均衡
     → DeepFM α=0.995（类别反比计算）：无内置稀疏，需强加权拉向正类
  3. TabNet 用官方库（被种子 bug 坑过后决定不用手搓版）
  4. FTT 和 DeepFM 用手搓实现 + 统一 Trainer
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_tabnet.tab_model import TabNetClassifier

from config import (
    PROCESSED_DIR, MODEL_DIR, RESULT_DIR, DEVICE, SEED,
    BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    FOCAL_ALPHA, FOCAL_GAMMA,
)
from training.trainer import Trainer
from training.losses import FocalLoss
from training.metrics import compute_all_metrics, find_best_threshold
from models.ft_transformer import FTTransformer
from models.deepfm import DeepFM


def reset_seed(seed=SEED):
    """独立重置所有随机种子——每个模型训练前调用一次，确保公平比较。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_processed():
    """加载预处理后的 .pt 张量。"""
    train_X, train_y = torch.load(os.path.join(PROCESSED_DIR, "train.pt"), weights_only=True)
    val_X, val_y = torch.load(os.path.join(PROCESSED_DIR, "val.pt"), weights_only=True)
    test_X, test_y = torch.load(os.path.join(PROCESSED_DIR, "test.pt"), weights_only=True)
    return train_X, train_y, val_X, val_y, test_X, test_y


def create_loaders(train_X, train_y, val_X, val_y, test_X, test_y):
    """构建标准的 2-tuple (X, y) DataLoader。"""
    train_ds = TensorDataset(train_X, train_y)
    val_ds = TensorDataset(val_X, val_y)
    test_ds = TensorDataset(test_X, test_y)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False)
    return train_loader, val_loader, test_loader


def train_tabnet_official(train_X, train_y, val_X, val_y, test_X, test_y):
    """
    用官方 pytorch-tabnet 库训练 TabNet。
    注意：TabNet 在论文中已被排除（因种子污染 bug），保留此代码仅为复现完整性。
    """
    print(f"\n{'='*60}")
    print("Training: TabNet (pytorch-tabnet official)")
    print(f"{'='*60}")

    model = TabNetClassifier(
        n_d=32, n_a=32, n_steps=4, gamma=1.3,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=20, gamma=0.5),
        mask_type='sparsemax',       # TabNet 也用 Sparsemax（但和 FTT 的用途不同）
        seed=SEED,                   # ← 这个 seed 在官方库里会被内部 numpy 状态覆盖
        device_name=DEVICE,
    )

    # TabNetClassifier.fit 需要 numpy 数组
    X_train_np = train_X.numpy()
    y_train_np = train_y.numpy().ravel()
    X_val_np = val_X.numpy()
    y_val_np = val_y.numpy().ravel()
    X_test_np = test_X.numpy()
    y_test_np = test_y.numpy().ravel()

    model.fit(
        X_train=X_train_np, y_train=y_train_np,
        eval_set=[(X_val_np, y_val_np)],
        eval_name=['val'], eval_metric=['auc'],
        max_epochs=MAX_EPOCHS, patience=15,
        batch_size=BATCH_SIZE, virtual_batch_size=128,
    )

    test_probs = model.predict_proba(X_test_np)[:, 1]
    val_probs = model.predict_proba(X_val_np)[:, 1]

    best_threshold, _ = find_best_threshold(y_val_np, val_probs)
    metrics = compute_all_metrics(y_test_np, test_probs, threshold=best_threshold)
    metrics["Model"] = "TabNet"

    model.save_model(os.path.join(MODEL_DIR, "tabnet_official"))
    print(f"\nTabNet Test Results (threshold={best_threshold:.3f}):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


def train_pytorch_model(name, model, train_loader, val_loader, test_loader, loss_fn, lr=LEARNING_RATE):
    """
    使用统一 Trainer 训练 PyTorch 模型。
    这个函数是通用的——FTT 和 DeepFM 都通过它训练，只是模型和 loss_fn 不同。
    """
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8
    )

    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        optimizer=optimizer, scheduler=scheduler,
        model_name=name,
    )
    trainer.fit(train_loader, val_loader, max_epochs=MAX_EPOCHS)

    # 恢复最佳模型 → 在测试集上评估
    trainer.load_best()
    test_scores, test_labels = trainer.predict(test_loader)
    val_scores, val_labels = trainer.predict(val_loader)
    best_threshold, _ = find_best_threshold(val_labels, val_scores)

    metrics = compute_all_metrics(test_labels, test_scores, threshold=best_threshold)
    metrics["Model"] = name

    print(f"\n{name} Test Results (threshold={best_threshold:.3f}):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


def main():
    print(f"Device: {DEVICE}")

    train_X, train_y, val_X, val_y, test_X, test_y = load_processed()
    input_dim = train_X.shape[1]

    # 计算类别统计信息，用于 DeepFM 的 Focal α
    n_pos = train_y.sum().item()
    n_neg = len(train_y) - n_pos
    # 类别反比 α = 1 - 正类比例 ≈ 0.995（极度不平衡）
    alpha_balanced = 1.0 - n_pos / (n_pos + n_neg)
    print(f"Train: {len(train_y):,} samples, fraud={n_pos} ({n_pos/len(train_y):.4%})")
    print(f"Focal Loss alpha_balanced={alpha_balanced:.4f}, alpha_default={FOCAL_ALPHA}")

    results = []

    # ── TabNet（官方 pytorch-tabnet）────────────────────────
    # 论文最终排除 TabNet，此处仅为复现完整性保留
    reset_seed(SEED)
    results.append(train_tabnet_official(train_X, train_y, val_X, val_y, test_X, test_y))

    # ── FT-Transformer（Sparsemax + α=0.25）──────────────────
    # α=0.25 的原因：Sparsemax 已在做特征选择，梯度压力小
    # 如果 α 太大（如 0.995），Sparsemax 的稀疏选择会被强正类梯度扭曲
    # lr=5e-4 比默认 1e-3 更小——因为 Sparsemax 的梯度分布不同
    reset_seed(SEED)
    ftt_train_loader, ftt_val_loader, ftt_test_loader = create_loaders(
        train_X, train_y, val_X, val_y, test_X, test_y
    )
    loss_fn_ftt = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)  # α=0.25
    ftt = FTTransformer(input_dim)
    results.append(train_pytorch_model(
        "FT-Transformer", ftt, ftt_train_loader, ftt_val_loader, ftt_test_loader,
        loss_fn_ftt, lr=5e-4
    ))

    # ── DeepFM（α=0.995, 类别反比）──────────────────────────
    # α=0.995 的原因：DeepFM 无任何内置稀疏机制
    # FM 对所有特征对做同等的内积，梯度均匀分布在所有特征上
    # 所以需要强加权 (199:1) 把梯度拉到正类
    reset_seed(SEED)
    dfm_train_loader, dfm_val_loader, dfm_test_loader = create_loaders(
        train_X, train_y, val_X, val_y, test_X, test_y
    )
    loss_fn_dfm = FocalLoss(alpha=alpha_balanced, gamma=FOCAL_GAMMA)  # α≈0.995
    deepfm = DeepFM(input_dim)
    results.append(train_pytorch_model(
        "DeepFM", deepfm, dfm_train_loader, dfm_val_loader, dfm_test_loader, loss_fn_dfm
    ))

    # ── 保存结果 ──────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULT_DIR, "neural_metrics.csv"), index=False)
    print(f"\nNeural model results saved to {RESULT_DIR}/neural_metrics.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
