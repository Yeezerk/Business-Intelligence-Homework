# 04 — FT-Transformer 原理

> **核心思想**：把表格数据的每个特征当作 NLP 中的一个"词"，用 Transformer 学习特征之间的全局依赖关系。

---

## 1. 从 NLP Transformer 到表格 Transformer

### 1.1 NLP Transformer 回顾

在文本中，Transformer 处理的是**离散 token**：
```
"I love this movie" → [Embedding(I), Embedding(love), Embedding(this), Embedding(movie)]
                    → + Positional Encoding
                    → Transformer Encoder
                    → Contextualized representations
```

每个词有一个**查表得到的嵌入向量**。

### 1.2 表格数据的挑战

表格数据是**连续的、异质的**：
- 特征V1 = 2.3（连续值，无词汇表可言）
- 特征V2 = -1.7
- Amount = 149.99

不能直接套用 Word Embedding，因为特征是标量而非离散 token。

### 1.3 FT-Transformer 的解决方案：Feature Tokenizer

每个特征通过一个**可学习的线性变换**映射到嵌入空间：

$$T_j = f_j(x_j) \cdot W_j + b_j \in \mathbb{R}^{d_{model}}$$

其中 $W_j \in \mathbb{R}^{d_{model}}$ 是第 $j$ 个特征专属的投影权重。

**直觉**：给每个特征一个"专属的 embedding 层"。$f_j(x_j)$ 是特征值的某种变换（原论文使用恒等变换 $f_j(x)=x$，但也可以使用分箱或样条变换）。

---

## 2. 架构全景

```
输入: x = [V1=2.3, V2=-1.7, ..., Amount=149.99]   (B, D)
         │
         ▼
┌─────────────────────────────────────────┐
│  Feature Tokenizer                      │
│  每个特征: T_j = x_j * W_j + b_j        │
│  输出: (B, D, d_model)                  │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Prepend [CLS] Token                    │
│  [CLS, T_1, T_2, ..., T_D]             │  (B, 1+D, d_model)
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Transformer Encoder × L layers         │
│  ┌───────────────────────────────────┐  │
│  │ LayerNorm → MHA → + Residual      │  │
│  │ LayerNorm → FFN → + Residual      │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
         │
         ▼
  取 [CLS] token 输出  →  LayerNorm  →  Linear(1)
```

---

## 3. 核心组件详解

### 3.1 Feature Tokenizer

```python
class FeatureTokenizer(nn.Module):
    def __init__(self, input_dim, d_model):
        # 每个特征有自己的投影权重
        self.W = nn.Parameter(torch.empty(input_dim, d_model))
        self.bias = nn.Parameter(torch.zeros(input_dim, d_model))

    def forward(self, x):  # x: (B, D)
        return x.unsqueeze(-1) * self.W.unsqueeze(0) + self.bias
        # → (B, D, d_model)
```

**关键设计**：每个特征有独立的 $W_j$，而不是所有特征共享。这是因为不同特征有不同的语义（一个是金额、一个是PCA分量、一个是时间相关），需要不同的嵌入表示。

等价于：`output[b, j, :] = x[b, j] * W[j, :] + bias[j, :]`

### 3.2 [CLS] Token

- 借鉴 BERT：在所有特征 token 前添加一个可学习的 [CLS] token
- [CLS] token 本身不携带任何特征信息
- 经过 Transformer 后，[CLS] 的输出汇聚了所有特征的信息（通过自注意力）
- 用 [CLS] 的输出做最终分类

```python
self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
# 在 forward 中：
cls = self.cls_token.expand(x.size(0), -1, -1)
tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+D, d_model)
```

### 3.3 为什么不需要位置编码？

NLP 中位置编码是必需的，因为词序包含信息。

在表格数据中：
- 特征没有顺序（V1,V2,...,V28 的顺序是任意的）
- 如果加了位置编码，模型可能学到"靠近 V1 的特征更重要"这种虚假模式
- 不加位置编码 = 告诉模型"特征顺序不重要" = **置换等变性 (Permutation Equivariance)**

### 3.4 Pre-Norm vs Post-Norm

FT-Transformer 使用 Pre-Norm 架构：

```
x = x + MHA(LayerNorm(x))    # 先归一化，后注意力
x = x + FFN(LayerNorm(x))    # 先归一化，后前馈
```

Pre-Norm 比原始 Transformer 的 Post-Norm 训练更稳定，尤其是在小数据/浅层网络上。

---

## 4. Sparsemax：本文的关键改进

### 4.1 Softmax 的问题

原始 FT-Transformer 用 Softmax 归一化注意力权重：

$$\mathrm{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

在 PCA 匿名化特征空间中，所有特征嵌入向量高度同质化 → Query-Key 内积趋近常数值 → Softmax 输出接近均匀分布 1/29 ≈ 0.0345。

**本文实验证实**：FTT 的注意力权重全在 [0.0327, 0.0342]，标准差仅 0.0004 → **注意力塌缩**。

### 4.2 Sparsemax 原理

Sparsemax 将 logits 向量投影到概率单纯形（probability simplex）上：

$$\mathrm{Sparsemax}(\mathbf{z}) = \arg\min_{\mathbf{p} \in \Delta^{d-1}} \|\mathbf{p} - \mathbf{z}\|_2^2$$

闭式解：Sparsemax(z)_i = max(z_i - τ(z), 0)，τ 是使 Σ max(z_i - τ, 0) = 1 的唯一阈值。

**核心差异**：Softmax 永远输出全非零概率（e^x > 0），Sparsemax 可以输出**真正的零值**——强制模型在各特征间做出选择。

### 4.3 实证效果

| 指标 | Softmax FTT | Sparsemax FTT | 改善 |
|------|------------|---------------|------|
| AUPRC | 0.8401 | 0.9179 | +4.4pp |
| ECE | 0.2328 | 0.0120 | ×20 倍 |
| Cost | 269 | 99 | -63% |

### 4.4 为什么 Sparsemax 帮助如此之大？

1. **强制稀疏**：即使 Query-Key 内积差异微小，Sparsemax 也会通过阈值 τ 筛掉低分特征
2. **梯度改善**：Softmax 均匀输出使所有特征获得均等梯度 → 无法区分；Sparsemax 让选中的特征获得主导梯度
3. **校准效果**：稀疏注意力防止概率输出过度分散 → ECE 从"不可用"提升到"良好"

> 注意：Sparsemax 解决了"完全塌缩"问题，但注意力分布仍近乎均匀——说明 PCA 空间本身缺乏注意力可利用的语义结构。这是引入树蒸馏的动机。

---

## 5. 超参数指南

| 超参数       | 推荐范围 | 作用                                                                |
| ------------ | -------- | ------------------------------------------------------------------- |
| $d_{model}$  | 32-256   | 特征嵌入维度。越大模型能力越强，但过拟合风险增加                    |
| $n_{heads}$  | 4-8      | 注意力头数。需整除 $d_{model}$                                      |
| $n_{layers}$ | 2-6      | Transformer 层数。对于大多数表格数据 3 层足够                       |
| $ffn_{dim}$  | 128-512  | FFN 隐藏维度。通常设为 $2 \times d_{model}$ 到 $4 \times d_{model}$ |
| $dropout$    | 0.1-0.3  | 正则化。表格数据比文本更容易过拟合，可用较高 dropout                |

---

## 6. 常见问题

**Q: 特征数量很大时（1000+），FT-Transformer 还适用吗？**

A: 自注意力的复杂度是 $O(D^2)$。当 $D > 500$ 时计算成本显著。可以先用特征选择降维，或改用 Linformer/Performer 等线性注意力变体。

**Q: 如果数据量小（< 10K 样本），能用 FT-Transformer 吗？**

A: 不建议。Transformer 本质上是"数据饥渴"的。小数据上用 XGBoost 或 MLP 往往更好。本项目 55 万+ 样本足够。

**Q: Sparsemax 已经大幅改善了 FTT，为什么不直接在 FTT 上用更大模型？**

A: Sparsemax 解决了归一化函数的低效问题，但没有解决根本原因——PCA 空间缺乏可供注意力区分的语义结构。注意力权重仍近乎均匀。本文因此转向树监督蒸馏：用 XGBoost 的外部特征重要性信号弥补注意力自身的盲区。但实验发现直接注入树先验反而不利——说明两种归纳偏置（轴对齐 vs 全局注意）是**结构性不兼容**的，而非简单的容量不足。

> **下一步**：阅读 `05_DeepFM与特征交叉.md` 了解如何显式建模特征之间的交互

---

## 附录：对话补充要点（快速回顾）

### A. Feature Tokenizer 的广播机制

```python
x.unsqueeze(-1) * self.W.unsqueeze(0)
# (B,D,1) * (1,D,d_model) → (B,D,d_model)
```

`unsqueeze` 的作用是**约束广播规则**：确保 $x[b,j]$ 只和 $W[j,:]$ 相乘，而不是乱乘。这是一个实现技巧，没有物理意义。

---

### B. [CLS] Token 的真正含义

**不是"留空"，而是可学习的注意力加权聚合器**。

```python
# [CLS] 输出 ≈ w₁×T₁ + w₂×T₂ + ... + w₅₀×T₅₀
# 其中 w 是注意力权重，模型学习得到
```

为什么不直接平均？
- **平均**：固定权重，无法学习
- **[CLS]**：通过注意力机制自动学习哪些特征更重要，训练端到端优化

---

### C. GELU 激活函数

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

- $\Phi(x)$ 是标准正态 CDF（不是假设输入服从正态分布，只是一个固定数学函数）
- 负值概率性通过（软性衰减），正值接近线性通过
- 比 ReLU 更平滑，比 Swish 有理论支撑
- 大模型标配：BERT、GPT、LLaMA 均使用

---

### D. 训练复杂度推导

**FT-Transformer** $O(D^2 \cdot d)$ 的来源：

```
Q @ K.T:  (D,d) @ (d,D) → (D,D)  → O(D²·d)
Softmax:  O(D²)
Softmax @ V: (D,D) @ (D,d) → O(D²·d)
```

**TabNet** $O(N_{steps} \cdot D \cdot d)$ 的来源：

```
每步 Q @ K.T:  (1,d) @ (d,D) → (1,D)  → O(D·d)
重复 N_steps 次
```

|        | TabNet            | FT-Transformer      |
| ------ | ----------------- | ------------------- |
| Q 形状 | (1, d) — 一个向量 | (D, d) — D 个 token |
| 本质   | 分步稀疏选择      | 全局密集交互        |

---

### E. MHA 和 FFN 速查

| 组件    | 全称                 | 作用                                    |
| ------- | -------------------- | --------------------------------------- |
| **MHA** | Multi-Head Attention | 让所有特征 token 互相交流，学习全局依赖 |
| **FFN** | Feed Forward Network | 逐位置非线性变换，两层全连接+GELU       |

```
MHA:  token 之间交互（信息交换）
FFN:  每个 token 独立变换（表达能力增强）
```
