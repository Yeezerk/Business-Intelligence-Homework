"""
统一训练器：所有 NN 模型（FTT、DeepFM、MLP）共用同一套训练/验证/早停逻辑。
关键设计：
  - 模型无关：任何 nn.Module 传入即可用，不需要为每个模型写训练循环
  - 早停以 AUPRC 为监控指标（而非 loss）——因为 loss 下降不一定代表欺诈检测更好
  - 梯度裁剪 clip=1.0：防止 Focal Loss 的 γ 指数放大难样本梯度导致梯度爆炸
  - KDTrainer 继承此类，重写 train_epoch/validate/predict 以处理蒸馏的 3-tuple loader
"""
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score

from config import DEVICE, MODEL_DIR, EARLY_STOP_PATIENCE


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        patience: int = EARLY_STOP_PATIENCE,
        model_name: str = "model",
        gradient_clip: float = 1.0,
    ):
        self.model = model.to(DEVICE)                      # 模型移到 GPU/CPU
        self.loss_fn = loss_fn                             # 损失函数（FocalLoss 或 BCEWithLogitsLoss）
        self.optimizer = optimizer                         # 优化器（AdamW）
        self.scheduler = scheduler                         # 学习率调度器（ReduceLROnPlateau）
        self.patience = patience                           # 早停容忍轮数
        self.model_name = model_name                       # 模型名称（用于保存文件名）
        self.gradient_clip = gradient_clip                 # 梯度裁剪阈值

        self.best_state = None                             # 最佳验证 AUPRC 时的模型参数副本
        self.best_val_auprc = 0.0                          # 最佳验证 AUPRC
        self.epochs_without_improvement = 0                # 连续未改善轮数计数器
        self.train_losses = []                             # 训练损失记录
        self.val_losses = []                               # 验证损失记录
        self.val_auprcs = []                               # 验证 AUPRC 记录

    def _to_device(self, X, y):
        """将数据批次移到训练设备（GPU/CPU）。"""
        return X.to(DEVICE), y.to(DEVICE)

    def train_epoch(self, loader):
        """
        一个训练 epoch：遍历所有 batch，前向→反向→更新参数。
        """
        self.model.train()                                 # 切换到训练模式（启用 Dropout/BatchNorm）
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = self._to_device(X_batch, y_batch)
            self.optimizer.zero_grad()                     # 清空上一批的梯度
            logits = self.model(X_batch).squeeze(-1)       # 前向：模型输出 logits，squeeze 去掉多余维度
            loss = self.loss_fn(logits, y_batch)            # 计算损失
            loss.backward()                                 # 反向传播：计算梯度
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)  # 梯度裁剪
            self.optimizer.step()                           # 更新参数
            total_loss += loss.item() * len(y_batch)        # 累加损失（加权平均用）
        return total_loss / len(loader.dataset)             # 返回 epoch 平均损失

    @torch.no_grad()
    def validate(self, loader):
        """
        验证一个 epoch：不计算梯度，只做前向和指标统计。
        @torch.no_grad() 装饰器禁用梯度计算——节省显存加速推理。
        """
        self.model.eval()                                  # 切换到评估模式（关闭 Dropout/BatchNorm 均值和方差固定）
        total_loss = 0.0
        all_scores, all_labels = [], []
        for X_batch, y_batch in loader:
            X_batch, y_batch = self._to_device(X_batch, y_batch)
            logits = self.model(X_batch).squeeze(-1)
            loss = self.loss_fn(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            all_scores.append(torch.sigmoid(logits).cpu())  # logits → 概率，移到 CPU
            all_labels.append(y_batch.cpu())                # 标签移到 CPU

        scores = torch.cat(all_scores).numpy()              # 拼接所有 batch 的概率
        labels = torch.cat(all_labels).numpy()

        auprc = average_precision_score(labels, scores)     # 主指标 AUPRC（用于早停判断）
        avg_loss = total_loss / len(loader.dataset)
        return avg_loss, auprc, scores, labels

    def fit(self, train_loader, val_loader, max_epochs: int):
        """
        完整训练循环：迭代 max_epochs 轮，每轮训练+验证+早停检查。
        """
        for epoch in range(1, max_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_auprc, _, _ = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_auprcs.append(val_auprc)

            # 学习率调度：如果验证 AUPRC 连续 patience 轮不提高，lr×0.5
            if self.scheduler is not None:
                self.scheduler.step(val_auprc)

            # 早停检查：验证 AUPRC 是否新高
            if val_auprc > self.best_val_auprc:
                self.best_val_auprc = val_auprc
                self.best_state = copy.deepcopy(self.model.state_dict())  # 深拷贝最佳参数
                self.epochs_without_improvement = 0
                self._save_checkpoint()
            else:
                self.epochs_without_improvement += 1

            # 每 5 轮（或第 1 轮）打印训练状态
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{max_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val AUPRC: {val_auprc:.4f}")

            # 达到早停条件 → 提前结束
            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping at epoch {epoch} (best AUPRC={self.best_val_auprc:.4f})")
                break

        # 训练结束后恢复到最佳参数
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self.model

    @torch.no_grad()
    def predict(self, loader):
        """
        用训练好的模型预测新数据：输出概率和真实标签。
        """
        self.model.eval()
        all_scores, all_labels = [], []
        for X_batch, y_batch in loader:
            X_batch, y_batch = self._to_device(X_batch, y_batch)
            logits = self.model(X_batch).squeeze(-1)
            all_scores.append(torch.sigmoid(logits).cpu())  # logits → 概率
            all_labels.append(y_batch.cpu())
        return torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy()

    def _save_checkpoint(self):
        """保存最佳模型权重到 outputs/models/ 目录。"""
        path = os.path.join(MODEL_DIR, f"{self.model_name}_best.pt")
        torch.save(self.best_state, path)

    def load_best(self):
        """从文件加载最佳模型权重。"""
        path = os.path.join(MODEL_DIR, f"{self.model_name}_best.pt")
        self.model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        self.model.to(DEVICE)
        return self.model
