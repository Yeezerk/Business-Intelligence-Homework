# 08 — XGBoost vs 深度学习：表格数据的博弈

> **核心问题**：XGBoost 在表格数据上称霸多年，深度神经网络真的能赢吗？在什么条件下能赢？

---

## 1. "无免费午餐"在表格数据上的体现

### 1.1 XGBoost 为什么在表格数据上强？

**结构优势**：

- **天然处理异构特征**：树模型对特征的尺度和分布不敏感。标准化？不需要。异常值？树模型天然鲁棒。
- **天然特征选择**：每次分裂只选一个特征，无关特征被自然忽略。
- **天然处理缺失值**：XGBoost 自动学习缺失值的最优分裂方向。
- **小样本友好**：几百棵树、深度=6 的 XGBoost 在 5K 样本上就能很好工作。

**对比神经网络**：
- 需要精心设计的数据预处理（标准化、异常值处理）
- 所有特征被一视同仁地输入（除非有注意力机制）
- 需要处理缺失值
- 通常需要 50K+ 样本才能稳定训练

### 1.2 什么时候神经网络能赢？

根据 2023-2024 年的研究共识（Grinsztajn et al., 2022; Gorishniy et al., 2021）：

| 条件             | 树模型优势 | 神经网络优势              |
| ---------------- | ---------- | ------------------------- |
| 数据量 < 10K     | ★★★        | ★                         |
| 数据量 10K-100K  | ★★         | ★★                        |
| 数据量 > 100K    | ★          | ★★★                       |
| 特征数 < 20      | ★★★        | ★                         |
| 特征数 20-100    | ★★         | ★★                        |
| 特征数 > 100     | ★          | ★★★                       |
| 特征主要是数值型 | ★★         | ★★                        |
| 有大量类别特征   | ★          | ★★★（Embedding）          |
| 需要特征交互     | ★★（隐式） | ★★★（注意力/Transformer） |

---

## 2. 关键论文发现

### 2.1 "Why do tree-based models still outperform deep learning on tabular data?" (Grinsztajn et al., 2022)

核心发现：
- 在 45 个中等规模表格数据集上，XGBoost 和随机森林在**没有超参调优**的情况下就超越了深度模型
- 神经网络对**无信息特征**（随机噪声特征）更敏感——添加随机特征后，NN 性能下降比树模型大得多
- **结论**：树模型对表格数据的归纳偏置（轴对齐分裂、对无信息特征的鲁棒性）极其有效

### 2.2 "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., 2021)

核心发现：
- FT-Transformer 在多个大规模表格数据基准上**首次严重挑战了 XGBoost**
- ResNet-like 架构（DCN-V2）也表现优异
- **关键**：需要更大的数据量和更多的超参调优，但深度模型的上限可能更高

### 2.3 "TabNet: Attentive Interpretable Tabular Learning" (Arik & Pfister, 2021)

核心发现：
- TabNet 在多个数据集上达到或超越了 XGBoost
- 关键场景：高维数据、特征选择重要的场景
- 优势：可解释性（逐步特征选择掩码）

---

## 3. 本项目中的实证检验

### 3.1 实验设计

本项目设计了一个"公平对决"的实验（见 [`code/04_train_baselines.py`](../code/04_train_baselines.py) 和 [`code/05_train_neural.py`](../code/05_train_neural.py)）：

**XGBoost**：300 棵树，max_depth=8，自动 scale_pos_weight
**TabNet**：Focal Loss，AdamW + CosineAnnealing
**FT-Transformer**：Focal Loss，AdamW + CosineAnnealing

都在**同一训练集、同一测试集**上评估。

### 3.2 预期结果

基于论文发现和数据集特性（55 万+样本，29 特征）：

- 如果数据集的信号主要由少数关键特征驱动 → TabNet 的稀疏特征选择可能胜出
- 如果特征之间有大量复杂交互 → FT-Transformer 的全局自注意力可能胜出
- 如果数据集的信号相对简单（主要依赖线性+少量交互）→ XGBoost 仍然可能胜出

### 3.3 结果的学术意义

无论结果如何，都有价值：
- **深度模型 > XGBoost**：验证了大规模数据下深度方法的可行性，为工业界升级提供证据
- **XGBoost > 深度模型**：验证了 Grinsztajn et al. 的发现，说明"简单有效"的原则，提醒不要盲目追求复杂模型

---

## 4. 工业实践建议

### 4.1 从 XGBoost 开始

不管最终用什么，XGBoost 是你的第一个基线：
- 秒级训练
- 强基线性能
- 特征重要性免费获取
- 帮助你理解"这个问题有多难"

### 4.2 深度模型的增量价值

只有当以下条件同时满足时，深度模型才是合理的：
1. XGBoost 的性能确实不够（业务要求更高的 AUPRC）
2. 数据量足够（> 10 万样本）
3. 有足够的工程资源维护神经网络管线
4. 可解释性要求可以通过 SHAP/注意力满足

### 4.3 实际部署中的考量

| 因素         | XGBoost              | 深度模型                     |
| ------------ | -------------------- | ---------------------------- |
| 推理延迟     | < 1ms                | 1-10ms（GPU）/ 5-50ms（CPU） |
| 模型大小     | MB 级                | 10-100 MB                    |
| 在线更新     | 困难（树结构调整）   | 支持（梯度更新）             |
| GPU 加速     | 不支持               | 原生支持                     |
| 可解释性工具 | TreeSHAP（快速精确） | KernelSHAP（慢速近似）       |

---

## 5. 核心结论

**没有放之四海皆准的赢家。** 选择取决于：

1. **数据特性**：规模、特征类型、噪声水平
2. **业务目标**：AUPRC 提升的边际价值 vs 工程复杂度的成本
3. **运维能力**：团队是否具备维护深度学习管线的能力
4. **监管环境**：是否需要细粒度的单笔交易解释

本项目通过在同一数据上公平比较，为你提供做出这个判断的**第一手经验**——这比阅读任何论文都更有价值。

---

## 6. 2024-2025 前沿：表格数据的新格局

### 6.1 TabPFN：表格数据的 Foundation Model

**TabPFN**（Tabular Prior-Data Fitted Network，Hollmann et al., 2024, ICLR）代表了表格学习的一个范式转变——从"为每个数据集训练一个模型"到"预训练一个通用模型，推理时直接使用"。

**核心思想**：
- 在数百万个合成表格数据集上预训练一个 Transformer
- 推理时将训练集作为上下文（in-context learning），无需梯度更新
- 在小到中等规模数据集（< 10K 样本）上表现惊人

**性能特点**：

| 数据规模  | TabPFN   | XGBoost | FT-Transformer |
| --------- | -------- | ------- | -------------- |
| < 1K 样本 | ★★★ 最优 | ★★      | ★              |
| 1K-10K    | ★★★      | ★★      | ★★             |
| 10K-100K  | ★        | ★★★     | ★★             |
| > 100K    | 不适用   | ★★★     | ★★★            |

**对本项目的启示**：
- 本项目数据集（55万+样本）超出了 TabPFN 的适用范围
- 但在真实场景中，很多风控冷启动问题（新客户、新市场）数据量很小，TabPFN 可能是更好的选择
- TabPFN 的不确定性量化（通过多次前向传播）是金融风控的天然需求

### 6.2 LightGBM：XGBoost 的现代替代

LightGBM 在 2024-2025 年的 Kaggle 竞赛和工业实践中已超越 XGBoost 成为最常用的 GBDT 框架：

| 维度               | XGBoost  | LightGBM               |
| ------------------ | -------- | ---------------------- |
| 训练速度           | 基准     | 2-5x 更快              |
| 内存使用           | 基准     | 更低（直方图算法）     |
| 类别特征支持       | 需要编码 | 原生支持               |
| 大数据集性能       | 好       | 更好                   |
| 小数据集过拟合风险 | 中       | 较高（leaf-wise 生长） |

**为什么本项目仍用 XGBoost**：XGBoost 在学术文献中更常作为基线，便于与其他论文横向对比。但实际部署建议优先考虑 LightGBM。

### 6.3 校准分析：概率预测的可靠性

在金融风控中，模型输出的概率值需要**可靠**——如果模型说"80%概率是欺诈"，那么实际欺诈率应该接近 80%。

**Expected Calibration Error (ECE)**：

$$\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} |acc(b) - conf(b)|$$

- 将预测概率分为 B 个桶（如 [0,0.1], [0.1,0.2], ...）
- 每个桶计算实际准确率与平均置信度的差距
- ECE 越低，模型越"诚实"

**校准方法**：
- **Platt Scaling**：用逻辑回归校准输出
- **Isotonic Regression**：非参数校准
- **Temperature Scaling**：深度模型专用，除以温度参数 T

**对本项目的启示**：当前评估只关注 AUPRC/F1，未检查概率校准。如果模型用于动态阈值调整（如根据风险等级分配审核资源），校准质量至关重要。

### 6.4 Conformal Prediction：有保证的不确定性量化

Conformal Prediction（保形预测）是 2024 年金融 AI 领域的热门话题，它提供**分布无关的覆盖率保证**：

- 不假设数据分布
- 给出预测集（而非点估计），保证真实标签以 $(1-\alpha)$ 的概率落入预测集
- 适用于任何预训练模型（作为后处理步骤）

```python
from sklearn.model_selection import train_test_split
from nonconformist.icp import IcpClassifier
from nonconformist.nc import NcFactory

# 将 XGBoost 包装为保形预测器
icp = IcpClassifier(NcFactory.create_nc(xgb_model))
icp.fit(X_cal, y_cal)
prediction = icp.predict(X_test, significance=0.1)
# prediction 包含每个样本的预测集，保证 90% 覆盖率
```

**业务价值**：
- "我们有 90% 的把握认为这笔交易属于{欺诈}集合" → 比单纯概率更有监管说服力
- 可以根据覆盖率要求动态调整审核策略

### 6.5 技术趋势总结

| 趋势                            | 成熟度     | 对本项目的适用性       |
| ------------------------------- | ---------- | ---------------------- |
| TabPFN（表格 Foundation Model） | 🟡 研究阶段 | 数据量过大，不适用     |
| LightGBM 替代 XGBoost           | 🟢 工业成熟 | 可作为额外基线         |
| 校准分析（ECE）                 | 🟢 工业成熟 | 推荐添加               |
| Conformal Prediction            | 🟡 快速成熟 | 推荐添加               |
| SHAP Interaction Index          | 🟢 工具成熟 | 推荐添加（见笔记 07）  |
| 在线学习/增量更新               | 🟡 研究阶段 | 概念漂移场景适用       |
| 公平性审计                      | 🟢 法规驱动 | 如果有受保护属性则必须 |

---

## SAINT 论文学习笔记（2026-05-30 补充）

### SAINT 架构要点

Somepalli et al. 2022 提出 **SAINT** (Self-Attention and Intersample Attention Transformer)：

1. **双重注意力**：每个 stage 包含 self-attention（列注意力，特征间交互）+ intersample attention（行注意力，样本间交互）
2. **连续特征嵌入**：用独立单层 ReLU MLP 将每个连续特征投影到 d 维——TabTransformer 仅对分类特征做此操作，SAINT 证明连续特征嵌入可显著提升性能（AUROC 从 89.38→91.72）
3. **对比预训练**：首个在表格数据上使用 contrastive learning（CutMix + mixup 双重增强）+ denoising loss
4. **在 13/16 数据集上超越 XGBoost/CatBoost/LightGBM**

### 与本项目的关系

- SAINT 是 FT-Transformer 的升级版（增加了 intersample attention）
- 本项目的 FT-Transformer 结果与 SAINT 的发现一致：全局自注意力在表格数据上有效
- SAINT 论文的 caption 极简风格被本项目采纳为排版规范

### 论文排版规范（从 SAINT 论文学习）

- **Caption 精简**：1-2句纯描述，不包含分析/解读
- **Figure 浮动**：使用 `[htbp]` 而非 `[H]`
- **正文承载分析**：所有数据解读、业务启示放在正文段落中
- **表格紧凑**：用小字号（`\small`）压缩表格空间
- **页数控制**：通过缩小子图宽度（0.72-0.82\textwidth）和减小页边距（2.0cm）实现 19 页

### MinerU PDF 转换工作流

```bash
# flash-extract（无需认证，<20页/<10MB）
mineru-open-api flash-extract paper.pdf -o output.md --language en
```

> **返回**：阅读 `00_项目全景与流程.md` 回顾整个项目的全局架构
