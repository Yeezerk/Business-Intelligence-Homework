"""
Focal Loss：处理 0.5% 极度不平衡的核心工具。
公式：FL(p_t) = -α_t · (1-p_t)^γ · log(p_t)
  - α：类别权重（α_t = α 当 y=1, 1-α 当 y=0）——控制梯度预算分配
  - γ：聚焦参数——降低易分类样本的损失贡献
  - γ=0 时退化为带 α 加权的标准交叉熵

为什么不用 ClassWeight + CE：
  - ClassWeight 对所有样本固定加权，不管它是否已被分对
  - Focal Loss 的 (1-p_t)^γ 会动态降低已分对样本的权重，避免过拟合到简单负样本
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        alpha: 正类权重系数（FTT用0.25，DeepFM用0.995——详见05_train_neural.py）
        gamma: 聚焦参数（2.0是原论文推荐值，本项目固定不改）
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: 模型输出的 logits（sigmoid 之前的值），形状 (N,)
        targets: 真实标签 {0, 1}，形状 (N,)
        """
        # 第 1 步：计算标准 BCE loss（每个样本独立，不求和/平均）
        # BCEWithLogitsLoss 内部做了 sigmoid + BCE，数值稳定性更好
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # 第 2 步：计算 p_t——模型对该样本的"确信度"
        # p_t = P(y=1) 如果真实标签是 1，否则 P(y=0)
        # p_t 越接近 1 表示模型分得越好
        probs = torch.sigmoid(inputs)          # 将 logits 转为概率
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # 第 3 步：聚焦权重 (1-p_t)^γ
        # 分得好的样本 p_t→1 → (1-p_t)^γ→0 → 损失被压低
        # 分得差的样本 p_t→0 → (1-p_t)^γ→1 → 损失保留
        focal_weight = (1 - p_t) ** self.gamma

        # 第 4 步：α_t 加权
        # α_t = α 对于正类（y=1），(1-α) 对于负类（y=0）
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # 第 5 步：三项相乘 → 取平均
        loss = alpha_t * focal_weight * bce_loss
        return loss.mean()
