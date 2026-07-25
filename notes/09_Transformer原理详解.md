# 09 — Transformer 原理详解

> **前置知识**：本假设读者有基本的深度学习基础（MLP、反向传播、梯度下降），但不熟悉序列建模和注意力机制。
> **学习目标**：理解 Transformer 的完整原理，为学习 FT-Transformer 打下基础。

---

## 1. 序列建模的挑战：从 RNN 说起

### 1.1 为什么需要处理序列？

现实世界中大量数据具有序列性质：

```
文本: "我 爱 你" → 词有顺序，"我爱你" ≠ "你爱我"
语音: [声谱图] → 帧有顺序，乱序后毫无意义
视频: [帧1, 帧2, ..., 帧N] → 时间顺序
时间序列: [价格1, 价格2, ..., 价格N] → 先后关系
```

处理序列数据需要模型能够：
1. **记住**之前看到的内容
2. **理解**前后文的依赖关系
3. **捕捉**长距离的关联（如段落开头和结尾的关联）

---

### 1.2 RNN 的设计思路

**循环神经网络 (Recurrent Neural Network)** 是早期处理序列的主流方法：

```
RNN Cell:
┌─────────────────────────────────┐
│  h_t = tanh(W · [x_t, h_{t-1}] + b)  │
│         ↑                    ↑         │
│      当前输入           上一时刻状态    │
└─────────────────────────────────┘

序列处理:
x₁ → RNN → h₁ → RNN → h₂ → RNN → h₃ → ...
       ↑
     h₁ 包含 x₁ 的信息
       ↑
     h₂ 包含 x₁,x₂ 的信息
       ↑
     h₃ 包含 x₁,x₂,x₃ 的信息
```

**核心思想**：用隐状态 $h_t$ 编码到时刻 $t$ 为止的所有历史信息。

---

### 1.3 RNN 的三大致命问题

#### 问题 1：梯度消失 / 梯度爆炸

反向传播时，梯度要穿过每一个时间步：

```
∂L/∂h₀ = ∂L/∂hₙ · ∂hₙ/∂h_{n-1} · ... · ∂h₁/∂h₀

如果 ∂h_t/∂h_{t-1} 的连乘 > 1 → 梯度爆炸
如果 ∂h_t/∂h_{t-1} 的连乘 < 1 → 梯度消失
```

**后果**：RNN 难以学习长距离依赖。"我今天心情很好，所以中午吃了一顿大餐"——这句话开头和结尾关系密切，但 RNN 很难捕捉到。

#### 问题 2：顺序计算，无法并行

```
x₁ → RNN → h₁
  ↓
x₂ → RNN → h₂  (必须等 h₁ 算完)
  ↓
x₃ → RNN → h₃  (必须等 h₂ 算完)
```

训练时，100 个词的句子需要等 100 步才能输出，GPU 并行优势无法发挥。

#### 问题 3：长距离依赖衰减

即使解决了梯度问题，RNN 的信息传递路径太长，有效信息会逐渐衰减。

---

### 1.4 LSTM 和 GRU：RNN 的改良

LSTM（长短期记忆网络）通过引入**门控机制**缓解梯度消失：

```
LSTM:
- 遗忘门：决定丢弃什么信息
- 输入门：决定写入什么新信息
- 输出门：决定输出什么信息

h_t = o_t * tanh(c_t)
c_t = f_t * c_{t-1} + i_t * c'_t  ← 记忆细胞直接传递，减缓衰减
```

**效果**：LSTM 能捕捉更长的依赖，但：
- 计算仍然是**顺序的**，无法并行
- 对超长序列（如 1000+ tokens）仍然困难

---

## 2. 注意力机制：让模型自己决定关注什么

### 2.1 注意力机制的诞生背景

2014 年，Bahdanau 等人在机器翻译论文《Neural Machine Translation by Jointly Learning to Align and Translate》中首次提出注意力机制。

**核心问题**：翻译 "The cat sat on the mat" → "猫坐在垫子上"时，
- 翻译"猫"应该主要关注 "cat"
- 翻译"坐"应该主要关注 "sat"
- 翻译"在...上"应该主要关注 "on"

**传统 Seq2Seq**："编码器"把整个句子压缩成一个向量，解码器从这个向量中抽取信息。这个向量成了信息瓶颈。

**带注意力的 Seq2Seq**：解码器每一步都可以"回头看"编码器的所有隐状态，并根据当前输出选择关注哪些部分。

---

### 2.2 注意力机制的直观理解

**类比**：你在翻译一段话时会反复回头看原文中相关的词。

```
翻译第 3 个词时：
原文: [The] [cat] [sat] [on] [the] [mat]
权重: [0.05] [0.40] [0.25] [0.10] [0.05] [0.15]
                    ↑                      ↑
               最高关注                  次高关注
```

---

### 2.3 注意力分数的计算

**Query-Key-Value (QKV) 框架**：

```
Query (Q): 我当前要翻译的词 → "坐"
Key (K):    原文中每个词的"索引" → [The, cat, sat, on, the, mat]
Value (V):  原文中每个词的"内容" → [The, cat, sat, on, the, mat]

注意力分数 = Query 和所有 Key 的相似度
注意力输出 = 分数加权 × Value
```

---

### 2.4 点积注意力的数学形式

```python
# 缩放点积注意力 (Scaled Dot-Product Attention)
def attention(Q, K, V):
    d_k = K.size(-1)  # Key 的维度

    # Q @ K.T: (seq_len, seq_len) 相似度矩阵
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    # softmax: 归一化成概率分布（权重和为1）
    weights = F.softmax(scores, dim=-1)

    # 加权求和
    output = weights @ V
    return output
```

**为什么要除以 $\sqrt{d_k}$？**
- 当 $d_k$ 很大时，点积的值会很大
- 大值经过 softmax 会趋近于 one-hot，梯度很小
- 除以 $\sqrt{d_k}$ 可以让点积的方差保持稳定

**图示**：

```
Q: (1, d_k)  — 当前要查询的向量
K: (seq_len, d_k)  — 所有 key 向量
K^T: (d_k, seq_len)

scores = Q @ K^T = (1, seq_len)  — 当前词对每个 key 的相似度
```

---

### 2.5 Multi-Head Attention：多角度理解

单一注意力头只能捕捉一种关联模式。Multi-Head 让模型同时从多个角度关注信息：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # 4 个投影矩阵（W_Q, W_K, W_V, W_O）
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        B = Q.size(0)

        # 1. 线性投影，分成多个头
        Q = self.W_Q(Q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. 缩放点积注意力
        attn_output = attention(Q, K, V)

        # 3. concat 多个头
        concat = attn_output.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        # 4. 最终线性投影
        return self.W_O(concat)
```

**为什么分成多个头？**

```
头 1 (关注语法):  "The cat" → 主语-动词关系
头 2 (关注语义):  "cat" → 动物相关
头 3 (关注位置):  "cat" → 句首区域
...
每个头独立计算 attention，最后 concat
```

---

## 3. Self-Attention：自己关注自己

### 3.1 什么是 Self-Attention？

**Self-Attention（自注意力）**：Query、Key、Value 都来自同一个输入。

```python
# NLP 中的 Self-Attention
输入: [我, 爱, 中, 国]  → 先嵌入成向量 → X = (seq_len, d_model)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

output = attention(Q, K, V)  # 每个词都attend到序列中所有词（包括自己）
```

**核心能力**：捕捉序列内部的依赖关系——"我爱中国"中，"中"和"国"应该attend到"我"和"爱"。

---

### 3.2 Self-Attention vs 交叉注意力

| 类型 | Q 来源 | K,V 来源 | 用途 |
|------|--------|---------|------|
| **Self-Attention** | X | X | 捕捉序列内部依赖 |
| **交叉注意力 (Cross Attention)** | 解码器 | 编码器 | 翻译/生成时看原文 |

```
Self-Attention in Encoder:
[我, 爱, 中, 国] → attend to each other → [我', 爱', 中', 国']

Cross-Attention in Decoder:
[我, 爱, 中] → attend to [The, cat, sat] → [我'', 爱'', 中'']
```

---

### 3.3 Self-Attention 的三大优势

1. **并行计算**：所有位置的 attention 同时计算，不依赖上一时刻输出
2. **长距离依赖**：任意两个位置之间的路径长度为 O(1)，信息直接交互
3. **可解释性**：注意力权重直观展示"谁关注谁"

```
RNN: 位置1 ↔ 位置100 需要经过 99 步传递
Self-Attention: 位置1 ↔ 位置100 一步到位
```

---

## 4. 位置编码：让模型知道词的顺序

### 4.1 问题：Self-Attention 是置换不变的

```python
# 打乱输入顺序，Attention 输出不变！
Attention([T₁, T₂, T₃]) = Attention([T₃, T₁, T₂])
```

但 NLP 中词序至关重要：
- "我爱你" ≠ "你爱我"
- "狗咬人" ≠ "人咬狗"

---

### 4.2 绝对位置编码 (Absolute Positional Encoding)

原始 Transformer (Vaswani et al., 2017) 使用正弦/余弦编码：

```python
def positional_encoding(seq_len, d_model):
    PE = torch.zeros(seq_len, d_model)

    # 位置 pos，维度 i
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            # 偶数维度用 sin
            PE[pos, i] = math.sin(pos / 10000 ** (2 * i / d_model))
            # 奇数维度用 cos
            PE[pos, i+1] = math.cos(pos / 10000 ** (2 * i / d_model))

    return PE  # shape: (seq_len, d_model)
```

**特点**：
- 每个位置有唯一编码
- 周期性的 sin/cos 组合让模型能学习不同频率的位置模式
- 编码是固定的，不需要学习

**直觉**：不同频率的波叠加，可以表示任意位置。

---

### 4.3 可学习的位置编码

BERT、GPT 采用可学习的 Embedding：

```python
self.position_embedding = nn.Embedding(max_seq_len, d_model)

# 输入: 位置索引 0, 1, 2, ..., seq_len-1
positions = torch.arange(seq_len).unsqueeze(0)  # (1, seq_len)
pos_enc = self.position_embedding(positions)   # (1, seq_len, d_model)

# 最终输入 = Token Embedding + Position Embedding
input_embedding = token_embedding + pos_enc
```

---

### 4.4 相对位置编码 (Relative Positional Encoding)

不编码"绝对位置"，而是编码"相对距离"：

```python
# Shaw et al., 2018
# bias(j - i) = 可学习的相对距离 j-i 的偏置

attention_score = (x_i W_Q)(x_j W_K)^T + bias(j - i)
```

**优势**：对不同距离用不同参数建模，更符合语言习惯（如"紧邻的词关系更密切"）。

---

### 4.5 旋转位置编码 RoPE (2022年后主流)

RoPE 已成为大模型的事实标准，核心思想：**通过旋转 Query/Key 向量来编码位置**。

```python
# 二维情况下
R(θ) = [cos(θ), -sin(θ)]
       [sin(θ),  cos(θ)]

# 位置 m 的 Query 旋转
q'_m = R(mω) · q_m

# 位置 n 的 Key 旋转
k'_n = R(nω) · k_n

# 注意力分数只依赖相对位置 m-n
q'_m · k'_n = q_m · R(m-n)·ω · k_n
```

**三大优势**：
1. 推理时可处理任意长度（理论无限上下文）
2. 无需额外 bias 参数
3. 相对位置信息自然融入向量方向

代表模型：LLaMA、PaLM、Falcon、Mistral。

---

## 5. Transformer Encoder 完整架构

### 5.1 整体结构

```
输入: [我, 爱, 中, 国]  (seq_len=4)
        ↓
┌─────────────────────────┐
│  Token Embedding        │  每个词 → d_model 维向量
│  + Positional Encoding   │  加上位置信息
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│  Encoder Layer × N       │  N 层 Encoder
│  ┌───────────────────┐  │
│  │ LayerNorm → MHA   │  │  Self-Attention
│  │     + Residual    │  │  残差连接
│  ├───────────────────┤  │
│  │ LayerNorm → FFN   │  │  前馈网络
│  │     + Residual    │  │  残差连接
│  └───────────────────┘  │
└─────────────────────────┘
        ↓
输出: [h₁, h₂, h₃, h₄]  (每个位置都有一个 d_model 维向量)
```

---

### 5.2 残差连接 (Residual Connection)

```python
output = x + SubLayer(x)
```

**作用**：
- 梯度可以直接回传到输入，缓解梯度消失
- 让网络更容易学习恒等映射（如果子层输出没什么用，可以学成接近0）
- 层再深也不怕训练困难

---

### 5.3 LayerNorm vs BatchNorm

```python
# LayerNorm：对单个样本的所有特征归一化
# 输入 x: (B, seq_len, d_model)
ln = LayerNorm(x)  # 对最后一维 d_model 归一化

# BatchNorm：对 batch 内所有样本的单个特征归一化（不适合序列！）
```

**为什么 Transformer 用 LayerNorm？**
- 序列任务中每个样本长度可能不同
- BatchNorm 依赖 batch 内统计量，不稳定
- LayerNorm 对每个样本独立，更适合自回归生成

---

### 5.4 前馈网络 (Feed Forward Network)

```python
class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.l2(F.gelu(self.l1(x)))
```

**特点**：
- 两层全连接 + 非线性激活（GELU 或 ReLU）
- 中间维度通常是 d_model 的 2-4 倍（如 512→2048→512）
- 逐位置独立计算（不涉及 token 间交互）

---

### 5.5 Pre-Norm vs Post-Norm

```python
# Post-Norm（原始 Transformer）
x = x + MHA(LayerNorm(x))    # 先子层，后残差
x = x + FFN(LayerNorm(x))

# Pre-Norm（FT-Transformer 等现代模型）
x = x + MHA(LayerNorm(x))    # 先归一化，后子层
x = x + FFN(LayerNorm(x))
```

**区别**：
- Post-Norm：输出层归一化，训练初期梯度不稳定
- Pre-Norm：每层输入归一化，训练更稳定，适合深层网络

---

## 6. Transformer Decoder

### 6.1 Decoder 的特殊之处

Decoder 用于**自回归生成**（如翻译、写作），需要：

1. **Masked Self-Attention**：不能看到未来的词
2. **Cross Attention**：看编码器的输出

---

### 6.2 Masked Self-Attention（因果注意力）

训练时，Decoder 输入是**整个目标序列**，但计算注意力时必须"遮挡"未来位置：

```python
def mask_future_attention(scores, seq_len):
    """
    scores: (seq_len, seq_len) 相似度矩阵
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores.masked_fill_(mask, -1e9)  # 未来位置填负无穷
    return scores

# seq_len=4 时的 mask:
# [0, 1, 1, 1]    ← 第 1 个词只能看自己
# [0, 0, 1, 1]    ← 第 2 个词能看前两个
# [0, 0, 0, 1]    ← 第 3 个词能看前三个
# [0, 0, 0, 0]    ← 第 4 个词能看到所有
```

**为什么不能看未来？**
- 训练时：已知完整目标序列（"我爱中国"的正确翻译是" I love China"）
- 但解码器必须模拟真实生成过程：先生成"I"，再生成"love"，依此类推
- 否则模型会"作弊"——看到答案再输出

---

### 6.3 Cross Attention

Decoder 的每个位置 attend 到 Encoder 的所有输出：

```
Encoder 输出: [h₁, h₂, h₃, h₄, h₅]
                     ↑
Decoder 第 1 层 Q   ← K,V 来自 Encoder
```

---

### 6.4 完整 Decoder 结构

```
输入: [START, I, love, China, END]  (目标语言，shifted right)
        ↓
┌─────────────────────────────────────┐
│  Masked Self-Attention              │  不能看未来
│  (Query=Decoder输入, K=V=Decoder输入) │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│  Cross Attention                    │  看 Encoder 输出
│  (Query=Decoder, K=V=Encoder输出)    │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│  FFN                                │
└─────────────────────────────────────┘
        ↓ (重复 N 层)
        ↓
    Linear → Softmax → 预测下一个词
```

---

## 7. 训练与推理

### 7.1 训练：Teacher Forcing

```python
# 训练时：给模型正确的上一步输出
target = ["I", "love", "China", "END"]
input_shifted = ["START", "I", "love", "China"]  # 右移一位

loss = model(input_shifted)  # 计算每个位置的交叉熵损失
loss.backward()
```

**为什么右移？**
- 预测 "love" 时，输入应该是 "START I"
- 而不是 "START I love"（那已经是答案了）

---

### 7.2 推理：自回归生成

```python
def generate(model, start_token, max_len=100):
    output = [start_token]

    for _ in range(max_len):
        input_ids = tensor(output).unsqueeze(0)
        logits = model(input_ids)

        # 取最后一个词的 logits（刚生成的词）
        next_token = logits[:, -1, :].argmax(-1)
        output.append(next_token.item())

        if next_token == EOS:
            break

    return output
```

**问题**：每次只能生成一个词，然后把它加到输入里继续——**串行生成，速度慢**。

---

### 7.3 推理加速：Beam Search

朴素贪婪解码（每次取概率最高的词）可能陷入局部最优。

**Beam Search**：保留 top-k 条最可能的路径，同时搜索：

```
t=1: [I:0.8] [A:0.1] [The:0.1]  → 保留 top 3
t=2: [I love:0.7] [I am:0.2] [A is:0.05]  → 继续扩展
...
```

**权衡**：更好的生成质量 vs 更大的计算量（是贪婪解码的 k 倍）。

---

## 8. 与 FT-Transformer 的关系

### 8.1 FT-Transformer 沿用了什么

| Transformer 组件 | FT-Transformer 是否使用 |
|-----------------|----------------------|
| Self-Attention | ✅ 完全使用 |
| Multi-Head Attention | ✅ 完全使用 |
| FFN (GELU, 残差) | ✅ 完全使用 |
| Pre-Norm | ✅ 完全使用 |
| 位置编码 | ❌ 不使用（表格特征无顺序） |

### 8.2 FT-Transformer 改了什么

| 改动 | 原因 |
|------|------|
| 用 **Feature Tokenizer** 替代 Token Embedding | 表格数据是连续值，无法直接查表 |
| 去掉位置编码 | 表格特征 V1,V2,...,V28 顺序任意 |
| 加 **[CLS] token** | 用于最终分类，替代平均池化 |
| 输入不是词，是特征值 | 直接处理数值型特征 |

---

### 8.3 FT-Transformer 的独特价值

**原始 Transformer** 适用于：
- NLP（文本 token 有离散词表）
- CV（图像 patch 有空间位置关系）
- 语音（帧有顺序）

**FT-Transformer** 证明了：
- Transformer 思想可以迁移到纯数值型表格数据
- 全局注意力对捕捉特征间依赖有效
- 为表格数据提供了一个新范式

---

## 9. 总结

### Transformer 的核心组件

```
Input Embedding + Positional Encoding
         ↓
┌──────────────────────────────────┐
│     Encoder Layer (×N)           │
│  ┌────────────────────────────┐ │
│  │ Self-Attention (Multi-Head) │ │
│  │   → Q, K, V 全部来自输入    │ │
│  │   → O(n²) 但可并行          │ │
│  └────────────────────────────┘ │
│              + Residual         │
│  ┌────────────────────────────┐ │
│  │ FFN (GELU, 两层全连接)     │ │
│  └────────────────────────────┘ │
│              + Residual         │
└──────────────────────────────────┘
         ↓
Output (分类 / 继续生成)
```

### 核心优势

| 能力 | Transformer | RNN/LSTM |
|------|-------------|----------|
| 并行计算 | ✅ 完全并行 | ❌ 顺序依赖 |
| 长距离依赖 | ✅ O(1) 路径 | ❌ O(n) 路径 |
| 可解释性 | ✅ 注意力权重可视化 | ❌ 黑盒 |
| 训练稳定性 | ✅ Pre-Norm + 残差 | ⚠️ 梯度问题 |

### 关键设计决策

1. **QKV 框架**：通用性强，Self-Attention 和 Cross Attention 统一表示
2. **Multi-Head**：从多角度捕捉依赖
3. **位置编码**：弥补置换不变性，编码序列信息
4. **残差 + LayerNorm**：保障深层网络的稳定训练

---

> **延伸阅读**：
> - [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) — Jay Alammar，图文并茂
> - [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始论文
> - [RoFormer](https://arxiv.org/abs/2104.09864) — RoPE 论文
