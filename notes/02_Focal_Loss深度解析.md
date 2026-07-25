# 02 — Focal Loss 深度解析

> **一句话总结**：Focal Loss 让模型自动"忽略"那些已经学得很好的样本，专注于那些还分不清的难样本。

---

## 1. 从交叉熵到 Focal Loss 的演进

### 1.1 标准交叉熵 (Cross-Entropy, CE)

$$CE(p, y) = \begin{cases} -\log(p) & \text{if } y = 1 \\ -\log(1-p) & \text{if } y = 0 \end{cases}$$

简化写法（定义 $p_t$）：

$$p_t = \begin{cases} p & \text{if } y = 1 \\ 1-p & \text{if } y = 0 \end{cases}$$

$$CE(p_t) = -\log(p_t)$$

**问题**：所有样本的损失一视同仁。对于类别不平衡数据：
- 99% 是正常样本（易分类，$p_t$ 接近 1）
- 1% 是欺诈样本（难分类，$p_t$ 可能很低）

CE 对两者等权相加 → 正常样本的损失总和完全淹没了欺诈样本。

### 1.2 平衡交叉熵 (Balanced CE)

加入类别权重 $\alpha$：

$$CE(p_t) = -\alpha_t \log(p_t)$$

其中 $\alpha_t = \alpha$ if $y=1$ else $1-\alpha$。

**改进**：少数类得到更大权重。

**仍未解决**：所有少数类样本被同样对待。一个"已被模型正确分类"的欺诈样本和"模型完全搞错"的欺诈样本受到相同的惩罚。

### 1.3 Focal Loss 的关键创新

添加 **modulating factor** $(1-p_t)^\gamma$：

$$FL(p_t) = -\alpha_t \cdot \underbrace{(1-p_t)^\gamma}_{\text{modulating factor}} \cdot \log(p_t)$$

**这为什么有效？**

| $p_t$ | $(1-p_t)^\gamma$ ($\gamma=2$) | 含义 |
|-------|-------------------------------|------|
| 0.9（模型很自信且正确） | $(0.1)^2 = 0.01$ | 损失降低 100 倍！ |
| 0.5（模型不确定） | $(0.5)^2 = 0.25$ | 损失降低 4 倍 |
| 0.1（模型错得很离谱） | $(0.9)^2 = 0.81$ | 损失几乎不变 |

**直觉**：已经分对的样本 → 损失大幅降低 → 梯度很小 → 模型不再花精力优化它们。模型自动聚焦于"还分不对"的难样本。

---

## 2. 数学推导与代码实现

### 2.1 前向传播（手撕代码）

对照 `code/training/losses.py` 中的实现：

```python
def forward(self, inputs, targets):
    # Step 1: BCE Loss (per-sample, no reduction)
    # BCE = -[y * log(σ(x)) + (1-y) * log(1-σ(x))]
    bce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )

    # Step 2: p_t = σ(x) if y=1 else 1-σ(x)
    # 这表示"模型对正确答案的预测概率"
    probs = torch.sigmoid(inputs)
    p_t = probs * targets + (1 - probs) * (1 - targets)

    # Step 3: modulating factor = (1 - p_t)^γ
    focal_weight = (1 - p_t) ** self.gamma

    # Step 4: α_t weighting
    # α_t = α if y=1 else 1-α
    alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

    # Step 5: Final loss
    loss = alpha_t * focal_weight * bce_loss
    return loss.mean()
```

### 2.2 梯度分析

Focal Loss 对 logits $x$ 的梯度：

$$\frac{\partial FL}{\partial x} = \alpha_t \cdot y(1-p_t)^\gamma \cdot (\gamma \cdot p_t \log(p_t) + p_t - 1)$$

关键观察：当 $p_t \to 1$（模型自信且正确），$\frac{\partial FL}{\partial x} \to 0$。模型对已掌握的样本不再更新参数。

---

## 3. 超参数调参哲学

### 3.1 $\alpha$（平衡参数）

- **作用**：调整正负类的相对重要性
- **默认值**：$\alpha = 0.25$（原始论文用于目标检测）
- **本项目的设定**：$\alpha = \frac{N_{fraud}}{N_{total}}$（训练集中欺诈比例）

**直觉**：如果欺诈占 1%，设 $\alpha=0.01$，则欺诈样本的 $\alpha_t=0.01$，正常样本的 $\alpha_t=0.99$。等等，这不是让正常样本更重要了吗？

不对。$\alpha$ 控制的是"正类权重"，$\alpha_t = \alpha$ 当 $y=1$。所以 $\alpha=0.01$ → 欺诈权重 0.01，正常权重 0.99。这看起来反了？

**关键洞察**：$\alpha$ 和 $\gamma$ 一起工作。$\gamma$ 的 modulating factor 已经大幅降低了易分类正常样本的损失。在这种情况下，$\alpha$ 需要设得**小**来平衡。原始论文推荐 $\alpha=0.25$，实践中一般直接设为正类比例。

**调参建议**：
- 如果 Recall 太低（漏掉太多欺诈）：增大 $\alpha$ 或减小 $\gamma$
- 如果 Precision 太低（太多误报）：减小 $\alpha$ 或增大 $\gamma$

### 3.2 $\gamma$（聚焦参数）

- $\gamma = 0$：退化为标准交叉熵
- $\gamma = 1$：适度聚焦
- $\gamma = 2$：原始论文推荐，大多数情况适用
- $\gamma = 5$：极度聚焦——只有最难分的样本才被有效训练

**大 $\gamma$ 的风险**：
- 模型可能过度忽略"中等难度"样本
- 对噪声/异常样本极度敏感（因为它们的 $p_t$ 很低，会被重点学习）
- 训练初期不稳定

**调参建议**：从 $\gamma=2.0$ 开始，在 $[0.5, 5.0]$ 范围内搜索。

---

## 4. 与其他方法的对比实验

| 方法 | 数学形式 | 核心思想 | 何时用 |
|------|---------|---------|--------|
| Plain CE | $-\log(p_t)$ | 无特殊处理 | 平衡数据 |
| Balanced CE | $-\alpha_t \log(p_t)$ | 少数类放大 | 简单不平衡 |
| Focal Loss | $-\alpha_t (1-p_t)^\gamma \log(p_t)$ | 聚焦难样本 | 极度不平衡+噪声 |
| GHM Loss | 梯度密度加权 | 关注"梯度适中"的样本 | 替代 Focal Loss |

---

## 5. 常见问题

**Q: Focal Loss 为什么能处理不平衡？它不是为"易/难"设计的吗？**

A: 这两者是相关的。在极度不平衡数据上，多数类样本大量出现 → 模型很快学会它们 → 它们成了"易分类样本" → Focal Loss 降低它们的损失权重。少数类样本稀少 → 模型更难学会 → 它们保持高损失权重。效果等价于"少数类获得更多关注"。

**Q: Focal Loss 和 Class Weight 能一起用吗？**

A: 可以，但通常不需要。α 参数已经起到了 class weight 的作用。叠加使用可能导致过拟合少数类。

**Q: 多分类能用 Focal Loss 吗？**

A: 可以。将二分类的 p_t 定义推广到多分类：p_t = softmax 输出中对应正确类别的概率。其余公式完全相同。

**Q: 为什么高维数据下 SMOTE 效果下降？**

A: 高维空间中所有点间距离趋向相等（距离集中现象）。例如100维空间中，最近邻距离~0.057，第100近邻距离~0.063，比值~0.90——邻居选择实际接近随机。SMOTE 依赖 k 近邻找"临近的少数类样本"来做插值，但当距离无区分度时，插值方向可能指向多数类密集区域，而非少数类内部。

**Q: p_t 是如何计算的？**

A: p_t = probs * targets + (1 - probs) * (1 - targets)，其中 probs = sigmoid(logits)。当 y=1（欺诈）时 p_t = p（模型认为欺诈的概率）；当 y=0（正常）时 p_t = 1-p（模型认为正常的概率）。本质是"模型对正确答案的置信度"。

**Q: Sigmoid 的作用是什么？**

A: 将神经网络输出的任意实数 logits 映射到 [0,1] 概率。例如 logits=3.7 → sigmoid=0.976，表示"模型认为有 97.6% 概率是欺诈"。只有转成概率后，才能计算交叉熵损失和设置分类阈值。

**Q: 调参建议中的 Recall 和 Precision 是什么？**

A: Recall = TP/(TP+FN)，在所有实际欺诈中抓到了多少，Recall 低=漏掉太多欺诈。Precision = TP/(TP+FP)，模型说"欺诈"时有多少是真的，Precision 低=误报太多。Focal Loss 调参：增大 α 或减小 γ → Recall 上升；减小 α 或增大 γ → Precision 上升。

---

## 本项目实验发现（2026-05-30 更新）

### α 选值陷阱：class-balance α 并非最优

Focal Loss 原始论文中 $\alpha$=0.25 是通用默认值。本项目最初按直觉使用 class-balance 策略计算 $\alpha = 1 - \text{fraud\_ratio} \approx 0.995$（让少数类获得极高权重）。

**实验结果**：

| 模型 | α=0.995 AUPRC | α=0.25 AUPRC | 差异 |
|------|-------------|------------|------|
| MLP | 0.8828 | — | 正常 |
| FTT | 0.9034 | — | 正常 |
| DeepFM | 0.8818 | — | 正常 |
| **TabNet** | **0.8214** ❌ | **0.8737** ✓ | +5.2pp |

**原因**：α=0.995 意味着：
- 欺诈样本 loss 权重 = 0.995（极大）
- 正常样本 loss 权重 = 0.005（趋近于零）

对于 MLP/FTT/DeepFM，梯度仍能从欺诈样本中提取足够信号。但 TabNet 的**稀疏注意力机制**需要从两类样本中都获得梯度信号来学习有意义的特征选择掩码。当多数类的梯度趋零时，注意力掩码退化，无法学习有效特征选择。

**教训**：Focal Loss 的 α 并非越大越好。不同架构对 α 敏感度不同：
- 全连接/Transformer 架构：对 α 较鲁棒
- 稀疏注意力架构（TabNet）：需要更均衡的 α（接近 0.25-0.5）
- 经验法则：先用 α=0.25 验证训练稳定，再根据验证集 AUPRC 微调

> **下一步**：阅读 `03_TabNet架构剖析.md` 理解为什么表格数据也需要注意力机制
