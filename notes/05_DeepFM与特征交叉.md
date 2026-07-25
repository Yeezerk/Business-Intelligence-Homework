# 05 — DeepFM 与特征交叉

> **核心思想**：好的预测不仅取决于单个特征，更取决于特征之间的组合。FM 高效捕获二阶交互，Deep 组件捕获高阶交互，两者联合训练。

---

## 1. 什么是特征交叉？

### 1.1 为什么需要特征交叉？

在欺诈检测中，单个特征可能不足以判断：

- "交易金额 5000 元" → 可能是正常大额消费，也可能不是
- "交易时间凌晨 3 点" → 可能是国际网购，也可能不是
- "交易金额 5000 元 **且** 交易时间凌晨 3 点" → 这个组合更可疑

这就是**特征交互**：多个特征组合在一起产生的信号强于各自单独的信号之和。

### 1.2 MLP 能做特征交叉吗？

理论上能。全连接层的矩阵乘法 $Wx$ 中，所有权重都在线性组合中隐式地"交互"。但：

- 二阶交互（如 V1 × V2）需要网络自己学到，不做显式建模
- 在数据有限时，MLP 可能学不好低阶交互
- 没有直接的"特征对重要性"输出

### 1.2 MLP / DNN 基础

| 缩写 | 全称                   | 含义                           |
| ---- | ---------------------- | ------------------------------ |
| NN   | Neural Network         | 神经网络（1-2层）              |
| DNN  | Deep Neural Network    | 深度神经网络（3+层）           |
| MLP  | Multi-Layer Perceptron | 多层感知机，DNN 的一种经典结构 |

**MLP 结构**：
```
输入 → Linear → 激活函数 → Linear → 激活函数 → ... → 输出
MLP(x) = W₂ · σ(W₁ · x + b₁) + b₂
```

---

## 2. Factorisation Machine (FM)

### 2.1 朴素二阶交互

直接给每对特征一个权重：

$$\hat{y}_{2nd} = \sum_{i=1}^D \sum_{j=i+1}^D w_{ij} x_i x_j$$

问题：需要学习 $O(D^2)$ 个参数，且大多数 $w_{ij}$ 在稀疏数据上无法有效训练。

### 2.2 FM 的巧妙分解

FM 将每个交互权重分解为两个向量的内积：$w_{ij} \approx \langle v_i, v_j \rangle$

$$\hat{y}_{FM} = \sum_{i=1}^D \sum_{j=i+1}^D \langle v_i, v_j \rangle x_i x_j$$

其中 $v_i \in \mathbb{R}^k$ 是第 $i$ 个特征的嵌入向量。

**参数从 $O(D^2)$ 降到 $O(D \cdot k)$！** 对于 $D=29$ 的特征，$k=16$ 的嵌入维度，参数仅 $29 \times 16 = 464$。

### 2.2.1 内积（Inner Product）与 Embedding

**内积（点积）**：两个向量 $\vec{a}, \vec{b}$ 相乘得到**标量**
$$\vec{a} \cdot \vec{b} = \vec{a}^T \vec{b} = \sum_i a_i b_i$$

**Embedding Layer**：将离散特征映射到连续向量空间

| 方式      | 表示                    | 问题                   |
| --------- | ----------------------- | ---------------------- |
| One-Hot   | "猫" = `[1,0,0,...]`    | 稀疏高维，无法表示语义 |
| Embedding | "猫" = `[0.2,-0.5,0.8]` | 稠密低维，可学习语义   |

Embedding 向量**不是人为指定的**，而是随机初始化后通过任务训练反向学习得到的。语义相似的词会获得相似的向量。

### 2.3 计算优化：$O(nk)$ 而非 $O(n^2k)$

关键恒等式（高中数学）：

$$\sum_{i=1}^D \sum_{j=i+1}^D \langle v_i, v_j \rangle x_i x_j = \frac{1}{2} \left( \left\|\sum_{i=1}^D v_i x_i\right\|^2 - \sum_{i=1}^D \|v_i x_i\|^2 \right)$$

```python
def fm_interaction(embeddings):  # (B, D, k)
    sum_emb = embeddings.sum(dim=1)          # (B, k)
    sum_square = sum_emb.pow(2)              # (sum v_i)^2

    square_sum = embeddings.pow(2).sum(dim=1)  # sum(v_i^2)

    interaction = 0.5 * (sum_square - square_sum)  # (B, k)
    return interaction.sum(dim=1, keepdim=True)     # (B, 1)
```

### 2.4 FM 的梯度更新

设 FM 输出为 $y_{FM}$，损失函数为 $L$，则 $v_i$ 的梯度：

$$\frac{\partial L}{\partial v_i} = \frac{\partial L}{\partial y_{FM}} \cdot x_i \cdot \left( \sum_{j=1}^D v_j x_j - v_i x_i \right)$$

推导过程：
```
y_FM = Σ_{i<j} ⟨v_i, v_j⟩ x_i x_j
     = ½(‖Σ v_i x_i‖² - Σ‖v_i x_i‖²)

设 S = Σ v_i x_i

∂y_FM/∂v_i = ½(2S·x_i - 2v_i·x_i²)
           = (S - v_i·x_i)·x_i
           = (Σ v_j x_j - v_i x_i)·x_i
```

**直观理解**：特征 $v_i$ 的梯度 = 全局加权和（其他特征对它的"贡献"）× 自身特征值

---

## 3. DeepFM 完整架构

```
输入 x (B, D)
    │
    ├──→ Linear Layer ──────────────────────→ linear_out (B, 1)
    │    (1st-order: w_0 + Σ w_i x_i)
    │
    └──→ Embedding Layer ──────────────────→ emb (B, D, k)
              │                                    │
              ├──→ FM Interaction ─────────→ fm_out (B, 1)
              │    (2nd-order: pairwise inner products)
              │
              └──→ Flatten ──→ MLP ────────→ deep_out (B, 1)
                   (higher-order: DNN)
                        │
                        ▼
          ŷ = σ(bias + linear_out + fm_out + deep_out)
```

三个组件各司其职：

| 组件   | 阶数  | 捕获什么                 |
| ------ | ----- | ------------------------ |
| Linear | 1 阶  | 单个特征对结果的线性贡献 |
| FM     | 2 阶  | 特征两两之间的交互影响   |
| Deep   | 3+ 阶 | 多个特征的复杂非线性组合 |

### 3.1 MLP 的特征交互原理

FM 只捕获二阶交互，MLP 负责更高阶的交互。设 MLP 为两层：

```
MLP(x) = W₂ · σ(W₁ · x + b₁) + b₂
```

**为什么 MLP 能捕获高阶交互？**

| 层级     | 操作                                                                                 | 交互阶数       |
| -------- | ------------------------------------------------------------------------------------ | -------------- |
| 输入层   | $h_1 = \sigma(W_1 \cdot x)$                                                          | 各特征线性组合 |
| 第一层后 | $h_1$ 的每个元素是所有特征的**线性组合**                                             | 1 阶           |
| 第二层后 | $h_2 = W_2 \cdot h_1$，每个输出是 $h_1$ 的线性组合，而 $h_1$ 本身是 $x$ 的非线性函数 | **2 阶**       |
| 第三层后 | $h_3 = W_3 \cdot h_2$，继续组合                                                      | **3 阶**       |

**非线性激活函数 $\sigma$ 的关键作用**：

- 没有 $\sigma$：多层线性变换等价于一层 $W_{total} = W_n \cdots W_2 W_1$
- 有 $\sigma$：打破线性叠加，可以表达**乘积形式的交互**

例如，ReLU 可以近似"与"操作：
```
ReLU(w₁x₁ + w₂x₂ - b) 
≈ 1 当且仅当 x₁ 和 x₂ 都大于阈值
```

**MLP 捕获的特征交互阶数 ≈ 网络层数 + 1**（取决于宽度和激活函数）

---

## 4. 与 TabNet / FT-Transformer 的对比

| 维度         | DeepFM                | TabNet     | FT-Transformer  |
| ------------ | --------------------- | ---------- | --------------- |
| 特征交互方式 | 显式内积 + MLP        | 序列注意力 | 全局自注意力    |
| 可解释性     | 嵌入向量可分析        | 强（mask） | 中（attention） |
| 稀疏特征支持 | 优秀（FM 天然适合）   | 一般       | 一般            |
| 连续特征支持 | 好                    | 好         | 好              |
| 工业部署     | 非常广泛（广告/推荐） | 新兴       | 新兴            |

**为什么 TabNet / FT-Transformer 效果一般？**

| 原因           | 说明                                                    |
| -------------- | ------------------------------------------------------- |
| 数据量不够     | Transformer 需要大量数据（10万+），中小数据容易过拟合   |
| 交互可能不复杂 | 如果主要是二阶交互，FM 足够，注意力机制反而是杀鸡用牛刀 |
| 归纳偏置不匹配 | 表格数据不一定符合全局注意力、平权交互的假设            |
| 参数量大       | 数据有限时，大模型不如简单模型                          |

**FM 稀疏支持好的原因**：把 $w_{ij}$ 分解为 $\langle v_i, v_j \rangle$，即使 (用户A, 商品X) 从未同时出现，$v_{用户A}$ 和 $v_{商品X}$ 也能学到有意义的表示。

---

## 5. 核心代码解读

[`code/models/deepfm.py`](../code/models/deepfm.py) 中的关键代码：

```python
class DeepFM(nn.Module):
    def forward(self, x):
        # 1st-order: simple weighted sum
        linear_out = self.linear(x)  # (B, 1)

        # Shared embeddings for FM and Deep
        emb = self.embedding(x)      # (B, D, embed_dim)

        # 2nd-order: FM pairwise interaction
        fm_out = self.fm(emb)        # (B, 1)

        # Higher-order: Deep MLP on concatenated embeddings
        deep_out = self.deep(emb)    # (B, 1)

        # Combined output
        return self.bias + linear_out + fm_out + deep_out
```

注意：FM 和 Deep 共享同一个嵌入层。这意味着嵌入向量需要同时满足两个目标：
1. 在内积空间中能捕获特征交互（FM 需求）
2. 在拼接后能被 MLP 有效利用（Deep 需求）

---

## 6. 何时用 DeepFM vs 其他模型

- **特征高维稀疏**（如用户ID、商户ID）：DeepFM 的 FM 组件天然支持
- **主要是连续特征**（如本项目）：FM 的二阶交互仍有价值，但优势可能不如 TabNet/FTT 明显
- **需要显式特征交互分析**：可以通过 FM 的嵌入向量和交互贡献来分析"哪些特征对之间的交互最重要"

> **下一步**：阅读 `06_评估指标的商业视角.md`，从技术指标走向商业决策
