"""
FTT+KD 消融实验（3A only）——回答反事实问题：KD 本身有害还是 TreeReg 是元凶？
答辩要点：
  - 这是一个独立的消融脚本，和 05_train_distillation.py 中的 FTT+KD 实验组相同
  - 设计目的：如果只改 TreeReg 这一个变量（加/不加），AUPRC 变化多少？
  - 答案：FTT+KD=0.9347（+1.68pp from 基线 0.9179），FTT+KD+TreeReg=0.8974（-3.73pp）
  - → 因果链：KD 确实跨架构有效，TreeReg 是结构性拮抗的唯一原因
  - 打印了消融结果的解读规则（见脚本末尾的 if-AUPRC-≈ 注释块）
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
    BATCH_SIZE, MAX_EPOCHS, WEIGHT_DECAY,
    FOCAL_ALPHA, FOCAL_GAMMA, TEMPERATURE, KD_ALPHA,
)
from training.distillation import (
    get_teacher_soft_labels,
    DistillationLoss,
)
from training.losses import FocalLoss
from training.metrics import compute_all_metrics, find_best_threshold
from training.trainer import Trainer
from models.ft_transformer import FTTransformer


def reset_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


reset_seed(SEED)

# FTT hyperparameters — match 05_train_distillation.py
FTT_KD_D_MODEL = 48
FTT_KD_N_HEADS = 4
FTT_KD_N_LAYERS = 2
FTT_KD_LR = 2e-4


# ── Data loading ────────────────────────────────────────────────────

def load_processed():
    train_X, train_y = torch.load(
        os.path.join(PROCESSED_DIR, "train.pt"), weights_only=True
    )
    val_X, val_y = torch.load(
        os.path.join(PROCESSED_DIR, "val.pt"), weights_only=True
    )
    test_X, test_y = torch.load(
        os.path.join(PROCESSED_DIR, "test.pt"), weights_only=True
    )
    return train_X, train_y, val_X, val_y, test_X, test_y


def create_kd_loaders(train_X, train_y, val_X, val_y, test_X, test_y,
                       teacher_probs_train, teacher_probs_val, teacher_probs_test):
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


# ── KD Trainer (same as 05_train_distillation.py) ──────────────────

class KDTrainer(Trainer):
    def __init__(self, *args, distillation_loss_fn=None, tree_reg=None,
                 lambda_reg=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.distillation_loss_fn = distillation_loss_fn
        self.tree_reg = tree_reg
        self.lambda_reg = lambda_reg

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch, teacher_batch in loader:
            X_batch, y_batch = self._to_device(X_batch, y_batch)
            teacher_batch = teacher_batch.to(DEVICE)
            self.optimizer.zero_grad()
            logits = self.model(X_batch).squeeze(-1)
            if self.distillation_loss_fn is not None:
                loss = self.distillation_loss_fn(logits, y_batch, teacher_batch)
            else:
                loss = self.loss_fn(logits, y_batch)
            if self.tree_reg is not None and hasattr(self.model, 'tree_regularization_loss'):
                loss = loss + self.lambda_reg * self.model.tree_regularization_loss()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(y_batch)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def validate(self, loader):
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
        self.model.eval()
        all_scores, all_labels = [], []
        for X_batch, y_batch, _ in loader:
            X_batch, y_batch = self._to_device(X_batch, y_batch)
            logits = self.model(X_batch).squeeze(-1)
            all_scores.append(torch.sigmoid(logits).cpu())
            all_labels.append(y_batch.cpu())
        return torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"KD config: T={TEMPERATURE}  alpha={KD_ALPHA}")

    train_X, train_y, val_X, val_y, test_X, test_y = load_processed()
    input_dim = train_X.shape[1]

    # Extract teacher soft labels (same as previous distillation runs)
    print("\nExtracting XGBoost teacher soft labels ...")
    teacher_probs_train = get_teacher_soft_labels(train_X.numpy())
    teacher_probs_val = get_teacher_soft_labels(val_X.numpy())
    teacher_probs_test = get_teacher_soft_labels(test_X.numpy())

    # Build KD loaders
    train_loader, val_loader, test_loader = create_kd_loaders(
        train_X, train_y, val_X, val_y, test_X, test_y,
        teacher_probs_train, teacher_probs_val, teacher_probs_test,
    )

    # Focal Loss + Distillation Loss (T=4.0, alpha=0.7 — same as FTT+KD+TreeReg)
    focal_loss = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    kd_loss = DistillationLoss(focal_loss, T=TEMPERATURE, alpha=KD_ALPHA)

    # ── FT-Transformer + KD (3A only, NO TreeReg) ──────────────────
    reset_seed(SEED)
    print(f"\n{'='*60}")
    print("Training: FT-Transformer + KD (3A only) — ABLATION")
    print(f"  No TreeReg (feature_weights=None, tree_reg=None)")
    print(f"{'='*60}")

    ftt_kd = FTTransformer(
        input_dim,
        d_model=FTT_KD_D_MODEL,
        n_heads=FTT_KD_N_HEADS,
        n_layers=FTT_KD_N_LAYERS,
        feature_weights=None,   # <-- NO tree prior
    )

    optimizer = torch.optim.AdamW(
        ftt_kd.parameters(), lr=FTT_KD_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8
    )

    trainer = KDTrainer(
        model=ftt_kd, loss_fn=focal_loss,
        optimizer=optimizer, scheduler=scheduler,
        model_name="ftt_kd_only",
        distillation_loss_fn=kd_loss,
        tree_reg=None,          # <-- NO TreeReg
    )
    trainer.fit(train_loader, val_loader, max_epochs=MAX_EPOCHS)
    trainer.load_best()

    # Evaluate
    test_scores, test_labels = trainer.predict(test_loader)
    val_scores, val_labels = trainer.predict(val_loader)
    best_threshold, _ = find_best_threshold(val_labels, val_scores)
    metrics = compute_all_metrics(test_labels, test_scores, threshold=best_threshold)

    print(f"\n{'='*60}")
    print("FTT+KD (3A only) — Ablation Results")
    print(f"{'='*60}")
    print(f"  Threshold: {best_threshold:.4f}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Print comparison context
    print(f"\n{'='*60}")
    print("Context: FTT baseline vs. FTT+KD vs. FTT+KD+TreeReg")
    print(f"{'='*60}")
    print(f"  FTT (baseline, Sparsemax):      AUPRC=0.9179  ECE=0.0120")
    print(f"  FTT+KD (3A only, this run):     AUPRC=???     ECE=???")
    print(f"  FTT+KD+TreeReg (3A+3B, prev):   AUPRC=0.8974  ECE=0.0010")
    print(f"\n  If FTT+KD AUPRC ≈ 0.92+ → KD itself helps FTT, TreeReg is harmful")
    print(f"  If FTT+KD AUPRC ≈ 0.90  → KD+TreeReg both harmful to FTT")
    print(f"  If FTT+KD AUPRC ≈ 0.92  → KD helps, TreeReg negates")

    # Save metrics
    results = pd.DataFrame([metrics])
    results.to_csv(os.path.join(RESULT_DIR, "ftt_kd_ablation_metrics.csv"), index=False)
    print(f"\nMetrics saved to {RESULT_DIR}/ftt_kd_ablation_metrics.csv")


if __name__ == "__main__":
    main()
