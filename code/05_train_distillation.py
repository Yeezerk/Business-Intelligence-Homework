"""
知识蒸馏训练（方案 3A + 3B）——核心实验脚本。

A/B 消融设计的 3 个实验组：
  实验组 1: FTT + KD + TreeReg（3A+3B）→ AUPRC=0.8974（断崖下降）
  实验组 2: FTT + KD 3A only         → AUPRC=0.9347（KD 有效！+1.68pp）
  实验组 3: DeepFM + KD 3A only      → AUPRC=0.9495（全局最优）

没有实验组 1 vs 2，结论只能是模糊的"蒸馏对 FTT 不好"。
有了消融，才能精准定位→KD 跨架构有效，TreeReg 才是毒药。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score

from config import (
    PROCESSED_DIR, MODEL_DIR, RESULT_DIR, DEVICE, SEED,
    BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    FOCAL_ALPHA, FOCAL_GAMMA, TEMPERATURE, KD_ALPHA, TREE_REG_LAMBDA,
    USE_DISTILLATION,
)
from training.trainer import Trainer
from training.losses import FocalLoss
from training.metrics import compute_all_metrics, find_best_threshold
from training.distillation import (
    get_teacher_soft_labels,
    extract_xgb_feature_weights,
    DistillationLoss,
    TreeFeatureRegularization,
)
from models.ft_transformer import FTTransformer
from models.deepfm import DeepFM


# ═══════════════════════════════════════════════════════════════
# 种子卫生：每个模型训练前独立重置
# ═══════════════════════════════════════════════════════════════
def reset_seed(seed=SEED):
    """独立重置所有随机种子——确保模型间比较公平。"""
    random.seed(seed)            # Python 内置随机
    np.random.seed(seed)         # NumPy 随机
    torch.manual_seed(seed)      # PyTorch CPU
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU


reset_seed(SEED)

# FTT 蒸馏时使用更小的架构（减少参数量，观察蒸馏效果是否独立于模型容量）
# 蒸馏 FTT: d_model=48, n_heads=4, n_layers=2
# 基线 FTT:  d_model=64, n_heads=8, n_layers=3
FTT_KD_D_MODEL = 48
FTT_KD_N_HEADS = 4
FTT_KD_N_LAYERS = 2
FTT_KD_LR = 2e-4  # 蒸馏时学习率更小（因为教师信号更稳定，不需要大 lr）


# ═══════════════════════════════════════════════════════════════
# 数据加载与 KD Loader 构建
# ═══════════════════════════════════════════════════════════════

def load_processed():
    """加载预处理后的 .pt 张量。"""
    train_X, train_y = torch.load(os.path.join(PROCESSED_DIR, "train.pt"), weights_only=True)
    val_X, val_y = torch.load(os.path.join(PROCESSED_DIR, "val.pt"), weights_only=True)
    test_X, test_y = torch.load(os.path.join(PROCESSED_DIR, "test.pt"), weights_only=True)
    return train_X, train_y, val_X, val_y, test_X, test_y


def create_kd_loaders(train_X, train_y, val_X, val_y, test_X, test_y,
                      teacher_probs_train, teacher_probs_val, teacher_probs_test):
    """
    构建带教师软标签的 DataLoader。
    每个样本是 (X, y, teacher_prob) 三元组 → KDTrainer 才能工作。
    普通 DataLoader 是 (X, y) 二元组 → Trainer 也能用。
    """
    train_ds = TensorDataset(
        train_X, train_y,
        torch.tensor(teacher_probs_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        val_X, val_y,
        torch.tensor(teacher_probs_val, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        test_X, test_y,
        torch.tensor(teacher_probs_test, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False)
    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════════
# KDTrainer：继承 Trainer，处理 3-tuple
# ═══════════════════════════════════════════════════════════════

class KDTrainer(Trainer):
    """
    KDTrainer 继承 Trainer，重写 train_epoch/validate/predict。
    父类处理 2-tuple (X, y)，子类处理 3-tuple (X, y, teacher_prob)。
    利用了 Python 的解包机制——多一个 teacher_batch 参数。
    """

    def __init__(self, *args, distillation_loss_fn=None, tree_reg=None,
                 lambda_reg=TREE_REG_LAMBDA, **kwargs):
        super().__init__(*args, **kwargs)
        self.distillation_loss_fn = distillation_loss_fn  # DistillationLoss 实例（3A）
        self.tree_reg = tree_reg                          # TreeFeatureRegularization 实例（3B，可选）
        self.lambda_reg = lambda_reg

    def train_epoch(self, loader):
        """训练 epoch：解包 3-tuple，加蒸馏 loss + 可选 TreeReg。"""
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch, teacher_batch in loader:
            X_batch, y_batch = self._to_device(X_batch, y_batch)
            teacher_batch = teacher_batch.to(DEVICE)

            self.optimizer.zero_grad()
            logits = self.model(X_batch).squeeze(-1)

            # 3A 蒸馏损失（若提供了蒸馏 loss 函数）
            if self.distillation_loss_fn is not None:
                loss = self.distillation_loss_fn(logits, y_batch, teacher_batch)
            else:
                loss = self.loss_fn(logits, y_batch)

            # 3B TreeReg（若提供了正则化器且模型有 tree_regularization_loss 方法）
            if self.tree_reg is not None and hasattr(self.model, 'tree_regularization_loss'):
                loss = loss + self.lambda_reg * self.model.tree_regularization_loss()

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
            total_loss += loss.item() * len(y_batch)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def validate(self, loader):
        """验证 epoch：解包 3-tuple，计算 loss 和 AUPRC。"""
        self.model.eval()
        total_loss, all_scores, all_labels = 0.0, [], []
        for X_batch, y_batch, teacher_batch in loader:
            X_batch, y_batch = self._to_device(X_batch, y_batch)
            teacher_batch = teacher_batch.to(DEVICE)
            logits = self.model(X_batch).squeeze(-1)

            if self.distillation_loss_fn is not None:
                loss = self.distillation_loss_fn(logits, y_batch, teacher_batch)
            else:
                loss = self.loss_fn(logits, y_batch)

            total_loss += loss.item() * len(y_batch)
            all_scores.append(torch.sigmoid(logits).cpu())
            all_labels.append(y_batch.cpu())

        scores = torch.cat(all_scores).numpy()
        labels = torch.cat(all_labels).numpy()
        auprc = average_precision_score(labels, scores)
        return total_loss / len(loader.dataset), auprc, scores, labels

    @torch.no_grad()
    def predict(self, loader):
        """预测：解包 3-tuple，只取 X（忽略 teacher_prob）。"""
        self.model.eval()
        all_scores, all_labels = [], []
        for X_batch, y_batch, _ in loader:
            X_batch, y_batch = self._to_device(X_batch, y_batch)
            logits = self.model(X_batch).squeeze(-1)
            all_scores.append(torch.sigmoid(logits).cpu())
            all_labels.append(y_batch.cpu())
        return torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy()


# ═══════════════════════════════════════════════════════════════
# Main：3 个实验组的 A/B 消融
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"Device: {DEVICE}  |  Distillation: {USE_DISTILLATION}")
    print(f"KD config: T={TEMPERATURE}  α={KD_ALPHA}  λ_reg={TREE_REG_LAMBDA}")

    train_X, train_y, val_X, val_y, test_X, test_y = load_processed()
    input_dim = train_X.shape[1]

    # ── 第 1 步：提取教师软标签和特征权重 ──────────────
    # 这一步在 3 个实验组之间共享——确保教师信号一致
    print("\nExtracting XGBoost teacher soft labels ...")
    teacher_probs_train = get_teacher_soft_labels(train_X.numpy())
    teacher_probs_val = get_teacher_soft_labels(val_X.numpy())
    teacher_probs_test = get_teacher_soft_labels(test_X.numpy())

    print("\nExtracting XGBoost feature split importance ...")
    feature_weights = extract_xgb_feature_weights(train_X.numpy())

    # ── 第 2 步：构建 KD DataLoader ─────────────────────
    train_loader, val_loader, test_loader = create_kd_loaders(
        train_X, train_y, val_X, val_y, test_X, test_y,
        teacher_probs_train, teacher_probs_val, teacher_probs_test,
    )

    # ── 第 3 步：共享损失函数 ──────────────────────────
    focal_loss = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    kd_loss = DistillationLoss(focal_loss, T=TEMPERATURE, alpha=KD_ALPHA)
    tree_reg = TreeFeatureRegularization(lambda_reg=TREE_REG_LAMBDA)

    results = []

    # ── 实验组 1：FTT + KD + TreeReg（3A+3B）────────────
    # 假设：TreeReg 的轴对齐约束会让 Sparsemax 的稀疏注意力退化
    # 结果验证：AUPRC=0.8974（-3.73pp）→ 假设成立
    reset_seed(SEED)
    print(f"\n{'='*60}")
    print("Training: FT-Transformer + Tree-Supervised (3A + 3B)")
    print(f"{'='*60}")

    ftt = FTTransformer(
        input_dim, d_model=FTT_KD_D_MODEL, n_heads=FTT_KD_N_HEADS,
        n_layers=FTT_KD_N_LAYERS, feature_weights=feature_weights,  # 传入 3B 特征权重
    )
    optimizer = torch.optim.AdamW(ftt.parameters(), lr=FTT_KD_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)

    trainer = KDTrainer(
        model=ftt, loss_fn=focal_loss,
        optimizer=optimizer, scheduler=scheduler,
        model_name="ftt_kd_treereg",
        distillation_loss_fn=kd_loss,  # 3A
        tree_reg=tree_reg,              # 3B
        lambda_reg=TREE_REG_LAMBDA,
    )
    trainer.fit(train_loader, val_loader, max_epochs=MAX_EPOCHS)
    trainer.load_best()
    test_scores, test_labels = trainer.predict(test_loader)
    val_scores, val_labels = trainer.predict(val_loader)
    best_threshold, _ = find_best_threshold(val_labels, val_scores)
    metrics = compute_all_metrics(test_labels, test_scores, threshold=best_threshold)
    metrics["Model"] = "FT-Transformer + KD + TreeReg"
    results.append(metrics)

    # ── 实验组 2：FTT + KD 3A only（不含 TreeReg）───
    # 这是消融实验的核心——只去掉 TreeReg 一个变量
    # 结果：AUPRC=0.9347（反证 TreeReg 是唯一元凶）
    reset_seed(SEED)
    print(f"\n{'='*60}")
    print("Training: FT-Transformer + KD (3A only, ablation — no TreeReg)")
    print(f"{'='*60}")

    ftt_kd = FTTransformer(
        input_dim, d_model=FTT_KD_D_MODEL, n_heads=FTT_KD_N_HEADS,
        n_layers=FTT_KD_N_LAYERS, feature_weights=None,  # ← 关键：不加 TreeReg
    )
    opt_ftt_kd = torch.optim.AdamW(ftt_kd.parameters(), lr=FTT_KD_LR, weight_decay=WEIGHT_DECAY)
    sch_ftt_kd = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_ftt_kd, mode="max", factor=0.5, patience=8)

    ftt_kd_trainer = KDTrainer(
        model=ftt_kd, loss_fn=focal_loss,
        optimizer=opt_ftt_kd, scheduler=sch_ftt_kd,
        model_name="ftt_kd_only",
        distillation_loss_fn=kd_loss,  # 3A 仍在
        tree_reg=None,                  # ← 关键：不加 TreeReg
    )
    ftt_kd_trainer.fit(train_loader, val_loader, max_epochs=MAX_EPOCHS)
    ftt_kd_trainer.load_best()
    ftt_kd_test_scores, ftt_kd_test_labels = ftt_kd_trainer.predict(test_loader)
    ftt_kd_val_scores, ftt_kd_val_labels = ftt_kd_trainer.predict(val_loader)
    ftt_kd_threshold, _ = find_best_threshold(ftt_kd_val_labels, ftt_kd_val_scores)
    ftt_kd_metrics = compute_all_metrics(ftt_kd_test_labels, ftt_kd_test_scores, threshold=ftt_kd_threshold)
    ftt_kd_metrics["Model"] = "FT-Transformer + KD"
    results.append(ftt_kd_metrics)

    # ── 实验组 3：DeepFM + KD 3A only ───────────────────
    # 验证 KD 是否跨架构有效（DeepFM 使用 FM 内积，不是注意力）
    # 结果：AUPRC=0.9495（全局最优，KD 对异构架构都有增益）
    reset_seed(SEED)
    print(f"\n{'='*60}")
    print("Training: DeepFM + KD (3A only)")
    print(f"{'='*60}")

    deepfm = DeepFM(input_dim)
    opt_dfm = torch.optim.AdamW(deepfm.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sch_dfm = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_dfm, mode="max", factor=0.5, patience=8)

    dfm_trainer = KDTrainer(
        model=deepfm, loss_fn=focal_loss,
        optimizer=opt_dfm, scheduler=sch_dfm,
        model_name="deepfm_kd",
        distillation_loss_fn=kd_loss,
        tree_reg=None,  # DeepFM 不用 TreeReg（它没有 tokenizer 嵌入，用的是 FM 共享嵌入）
    )
    dfm_trainer.fit(train_loader, val_loader, max_epochs=MAX_EPOCHS)
    dfm_trainer.load_best()
    dfm_test_scores, dfm_test_labels = dfm_trainer.predict(test_loader)
    dfm_val_scores, dfm_val_labels = dfm_trainer.predict(val_loader)
    dfm_threshold, _ = find_best_threshold(dfm_val_labels, dfm_val_scores)
    dfm_metrics = compute_all_metrics(dfm_test_labels, dfm_test_scores, threshold=dfm_threshold)
    dfm_metrics["Model"] = "DeepFM + KD"
    results.append(dfm_metrics)

    # ── 保存结果 ──────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULT_DIR, "distillation_metrics.csv"), index=False)
    print(f"\nDistillation results saved to {RESULT_DIR}/distillation_metrics.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
