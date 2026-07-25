# OpenCode 项目知识库

> **课程**：商业智能大作业 — 信用卡欺诈检测
> **我的角色**：资深深度学习从业者 → 以直觉类比和底层原理向学生解释一切
> **用户角色**：大一新生 / 答辩学生 — 需要我帮助理解、完成论文、准备答辩
> **数据来源**：Kaggle Credit Card Fraud Detection Dataset 2023
> **最后更新**：2026-06-13

---

## 一、项目快照

| 项目 | 内容 |
|------|------|
| 任务 | 信用卡欺诈二分类检测（0.5% 欺诈率，极度不平衡） |
| 核心方法 | 知识蒸馏：XGBoost(教师) → NN(学生) |
| 对比架构 | FT-Transformer（自注意力）vs DeepFM（显式交互）— 两种正交归纳偏置 |
| 关键技术 | Sparsemax 替代 Softmax、Focal Loss 按架构定制 α、A/B 消融 |
| 评估主指标 | AUPRC（而非 ROC-AUC，因不平衡度太高） |
| 业务成本 | Cost = FP×1 + FN×10 |
| 论文 | 27 页 LaTeX, 7 章, 18 条引用, XeLaTeX 编译 |

### 最终模型排名

| 排名 | 模型 | AUPRC | Cost | ECE | 角色 |
|------|------|-------|------|-----|------|
| 🥇 | **DeepFM + KD** | **0.9495** | 120 | 0.0008 | 全局最优 |
| 🥈 | FTT + KD (3A only) | 0.9347 | **97** | **0.0007** | KD跨架构有效证明 |
| 🥉 | DeepFM | 0.9336 | 123 | 0.0394 | 显式交互基线 |
| 4 | XGBoost | 0.9331 | 215 | 0.0006 | 树基线/蒸馏教师 |
| 5 | FT-Transformer (Sparsemax) | 0.9179 | 99 | 0.0120 | 注意力基线 |
| 6 | FTT + KD + TreeReg | 0.8974 | 169 | 0.0010 | TreeReg拮抗(-3.73pp) |

### 四大核心发现

1. **Sparsemax 拯救 FTT**：Softmax 注意力在 PCA 空间塌缩为均匀分布(29个特征权重全在 0.033±0.0004)，Sparsemax 闭式解强制稀疏选择 → AUPRC +4.4pp (0.8401→0.9179), ECE ×20 改善
2. **KD 跨架构有效**：FTT+KD(3A only)=0.9347(+1.68pp), DeepFM+KD=0.9495(+1.59pp) — 软标签蒸馏对两种异构架构增益一致
3. **TreeReg 是退步唯一元凶**：FTT+KD+TreeReg=0.8974(-3.73pp from FTT+KD) — 树先验的轴对齐约束与 Sparsemax 注意力的自由交互存在结构性拮抗
4. **置换重要性 6 模型**：KD 拉近学生-教师特征排序(DeepFM r: 0.625→0.845), TreeReg 制造畸形复刻(r=0.964 但 V14 集中度 43.9%), 跨架构 KD 趋同(r: 0.639→0.806)

---

## 二、数据管道

| 步骤 | 脚本 | 产出 | 说明 |
|------|------|------|------|
| 下载 | `data/01_download.py` | `outputs/raw/credit_card_fraud.csv` | KaggleHub API (56.8万条, 原始50:50均衡) |
| EDA | `data/02_eda.py` | `outputs/figures/*_eda.pdf` | 类别平衡、金额分布、相关性 |
| 预处理 | `data/03_preprocess.py` | `train.pt/val.pt/test.pt` (70/15/15) | StandardScaler + IQR截断 + **下采样至0.5%欺诈率** |

关键预处理决策：
- 人工制造 0.5% 欺诈率（TARGET_FRAUD_RATIO=0.005, MAX_LEGIT_SAMPLES=150,000）
- 29 维特征：V1-V28(PCA匿名化) + Amount(标准化)
- 特征名列表：`FEATURE_NAMES = [f"V{i}" for i in range(1,29)] + ["Amount"]`

---

## 三、代码架构

```
code/
├── config.py                    # 全局配置（路径/超参/种子/设备）
├── data/
│   ├── 01_download.py           # KaggleHub 下载
│   ├── 02_eda.py                # 探索性分析
│   └── 03_preprocess.py         # 预处理
├── models/
│   ├── ft_transformer.py        # FTT: FeatureTokenizer + CLS + Transformer(Sparsemax) + 分类头
│   └── deepfm.py                # DeepFM: LR + FM(二阶) + Deep MLP(高阶)
├── training/
│   ├── trainer.py               # 统一 Trainer (train_epoch/validate/predict/早停/模型保存)
│   ├── losses.py                # FocalLoss: FL(p_t) = -α_t·(1-p_t)^γ·log(p_t)
│   ├── distillation.py          # 3A: DistillationLoss + 3B: TreeFeatureRegularization
│   └── metrics.py               # AUPRC/ROC-AUC/F1/Precision/Recall/Cost/ECE
├── 04_train_baselines.py        # LR → RF → XGBoost
├── 05_train_neural.py           # FTT + DeepFM 基线训练（含 reset_seed）
├── 05_train_distillation.py     # 蒸馏: FTT+KD+TreeReg + FTT+KD(3A) + DeepFM+KD
├── 05_train_distillation_ablation.py  # FTT+KD(3A only) 消融（独立脚本）
├── 05b_train_ablation.py        # 不平衡消融: FocalLoss vs ClassWeight vs SMOTE vs PlainCE
├── 06_evaluate.py               # 评估: PR/ROC/CM/Cost/校准曲线 + LaTeX表生成
├── 07_interpret.py              # 可解释性: SHAP beeswarm/waterfall + SII交互 + FTT注意力图
└── 08_interpret_deepfm.py       # 置换重要性: 6模型(XGB/DFM/DFM+KD/FTT/FTT+KD/FTT+KD+TreeReg)
```

### 设计模式
- **Config 单例**：`config.py` 集中管理所有路径/超参
- **关注点分离**：`models/`(纯架构) vs `training/`(纯训练逻辑) 解耦
- **统一 Trainer**：模型无关的训练循环 + 早停 + 模型保存为 `.pt`
- **种子卫生**：`reset_seed(SEED)` 在**每个模型训练前**独立调用（踩过 pytorch-tabnet 消耗 numpy 随机状态的坑）
- **KDTrainer 继承**：继承 Trainer，重写 train_epoch/validate/predict 处理 3-tuple (X, y, teacher_probs)

---

## 四、模型架构详解

### 4.1 XGBoost（教师/基线）
- **配置**：300 棵树, max_depth=8, lr=0.05, scale_pos_weight 自动计算, eval_metric=aucpr
- **为什么强**：轴对齐分裂天然适合表格数据，每分裂沿单维度 `x_i > t`
- **教师角色**：一次性对所有样本 `predict_proba()` → 概率/软标签 → 学生用 KL 散度模仿

### 4.2 FT-Transformer（自注意力范式）
```
输入 x: (B, 29)
  → FeatureTokenizer: 每个特征 j 的 64 维嵌入 T_j = x_j·W_j + b_j  (B,29,64)
  → [CLS] Token: 可学习向量拼接在前面 (B,30,64)
  → TransformerEncoder × 3 层 (8头, Pre-Norm, GELU FFN)
     注意力: Sparsemax(QK^T/√d_k) 替代 Softmax ← 核心改进
  → LayerNorm + Linear(64→1) 分类头
```
- **参数量**：153,921
- **关键改进 Sparsemax**：闭式解 O(d log d)，输出可精确为零，强制模型做特征选择
  - `τ = (Σ_{j∈S} z_sorted[j] - 1) / k(z)`，sparsemax_i(z) = max(z_i - τ, 0)
- **TreeReg (3B)**：`L_reg = λ·Σ(1-w_i)·||E_i||₂` 对嵌入层施加约束，w_i 来自 XGBoost split gain

### 4.3 DeepFM（显式交互范式）
```
输入 x: (B, 29)
  → 共享嵌入: e_i = x_i·v_i, v_i∈R^16  (B,29,16)
  → 一阶线性: w_0 + Σw_i·x_i
  → 二阶FM:   ½·Σ[(Σv_i·x_i)² - Σ(v_i²·x_i²)]  — O(DK)高效计算
  → 高阶Deep: Flatten(29×16) → MLP[256,128,64] → ReLU+BN+Dropout
  → 合并: sigmoid(bias + linear + fm + deep)
```
- **参数量**：161,648
- **为什么在 PCA 空间更稳健**：FM 内积是静态度量，不依赖动态 Query-Key 上下文；梯度路径短且稳定

### 4.4 蒸馏方案 (Tree-Supervised KD)

**3A — 软标签蒸馏**（跨架构有效）：
```
L = α·FL(student_logits, y_true) + (1-α)·T²·KL(p_s^T || p_t^T)
```
- T=4.0（温度软化），α=0.7（70%硬标签 + 30%模仿教师）
- KL 散度是二分类版：`p·log(p/q) + (1-p)·log((1-p)/(1-q))`
- T² 是梯度缩放补偿因子

**3B — 特征重要性正则化**（对 FTT 有毒）：
```
L_reg = λ·Σ(1-w_i)·||E_i||₂
```
- w_i: XGBoost split gain 归一化 [0,1]（V14=1.0, V12=0.07, V4=0.059）
- 高权重特征允许大嵌入范数，低权重被压缩 → 与 Sparsemax 自由选择拮抗
- λ=0.1, 结果：FTT AUPRC 从 0.9347 断崖降至 0.8974 (-3.73pp)

---

## 五、训练配置

| 超参 | 值 | 说明 |
|------|----|------|
| Batch Size | 512 | |
| Max Epochs | 100 | 早停 patience=15 |
| Optimizer | AdamW | lr=1e-3 (默认), 5e-4 (FTT基线), 2e-4 (蒸馏FTT) |
| Scheduler | ReduceLROnPlateau | factor=0.5, patience=8, mode=max |
| Gradient Clip | 1.0 | |
| Focal γ | 2.0 | 固定 |
| **FTT α** | **0.25** | Sparsemax 已有特征选择，需均衡梯度 |
| **DeepFM α** | **0.995** | 无内置稀疏，需强加权把梯度拉到正类 |
| Weight Decay | 1e-5 | |

蒸馏 FTT 使用了更小的架构（d_model=48, n_heads=4, n_layers=2 vs 基线 64/8/3）以匹配蒸馏实验组。

---

## 六、关键技术决策（为什么这样设计）

### 6.1 AUPRC 而非 ROC-AUC
- 欺诈仅 0.5%，ROC-AUC 被 TN 主导（三模型极差仅 0.0054）
- AUPRC 聚焦正类，极差 ~0.05，区分度 10 倍
- 直觉：ROC-AUC=随机抽一个正+一个负，正排前面的概率。99.5% 负样本时随便排都能排对

### 6.2 Sparsemax 替代 Softmax
- Softmax 永远输出非零概率 → PCA 空间所有特征嵌入同质化 → Query-Key 内积趋同 → 注意力均匀塌缩（29个权重全在 0.033±0.0004）
- Sparsemax 可输出零值，强制选择 → AUPRC +4.4pp, ECE ×20, Cost -63%

### 6.3 Focal Loss 的 α 按架构定制
- FTT (Sparsemax)：α=0.25 — 注意力已有特征选择梯度压力小，需均衡正负类梯度
- DeepFM (无内置稀疏)：α=0.995 — 类别反比强加权，把梯度预算拉到 199:1 的正类

### 6.4 为什么要做 3B 负结果
- 没有 3B 的 -3.73pp 断崖下降，结论只能模糊说"蒸馏对 FTT 效果不好"
- 有了 3A vs 3B 的 A/B 消融，才能精准定位：**KD 本身跨架构有效，TreeReg 才是毒药**
- 整篇论文的因果叙事从"蒸馏是架构依赖的" → "树先验的轴对齐与 Sparsemax 注意力存在结构性拮抗"

---

## 七、文件分布速查

| 你想做什么 | 打开哪个文件 |
|-----------|-------------|
| 看论文 | `paper/main.tex` + `paper/chapters/*.tex` |
| 编译论文 | `paper/` 下跑 xelatex + biber |
| 看最终结果 | `PROJECT_GUIDE.md` §13 |
| 跑所有训练 | `code/04_` → `05_` → `06_` → `07_` → `08_` |
| 改模型架构 | `code/models/ft_transformer.py` 或 `deepfm.py` |
| 改训练逻辑 | `code/training/trainer.py` |
| 改损失函数 | `code/training/losses.py` |
| 改蒸馏方案 | `code/training/distillation.py` |
| 改评估指标 | `code/training/metrics.py` |
| 加新模型 | 新建 `code/models/new_model.py` + `05_train_neural.py` 末尾加调用 |
| 重新生成图表 | `python code/06_evaluate.py` |
| 重新生成 SHAP | `python code/07_interpret.py` |
| 重新生成置换重要性 | `python code/08_interpret_deepfm.py` |
| 答辩准备 | `notes/12_答辩准备_核心知识体系.md` |
| 学习笔记 | `notes/00`~`12` 共 11 篇 |

---

## 八、答辩 Q&A 预设库

> 以下为答辩模拟记录，按角色（老师→学生）组织

### Q1: 为什么选 AUPRC 而不是 ROC-AUC？
**核心**：ROC-AUC 在不平衡数据下**反而偏高**（被 TN 主导），三个模型极差仅 0.0054，无法区分。AUPRC 只看正类，极差约 0.05，区分度是 ROC-AUC 的 10 倍。
**直觉**：ROC-AUC = 随机抽一个正样本和一个负样本，模型把正样本排前面的概率。99.5% 都是负样本时，随便排都能排对。

### Q2: AUPRC vs Focal Loss — 解决的是同一个问题吗？
**不是**。AUPRC 是**评估指标**（事后看模型好不好），Focal Loss 是**训练方法**（事中让模型关注难样本）。前者衡量结果，后者改变过程。

### Q3: Focal Loss 的 γ 和 α 分别控制什么？
- α（权重）：正负类平衡系数。控制了"有限梯度预算的分配方案"
- γ（聚焦参数）：控制难易样本的权重衰减速度。γ 越大，模型越聚焦在边界样本上
- FTT α=0.25：Sparsemax 已有特征选择，梯度压力小，需均衡
- DeepFM α=0.995：无内置稀疏，需强加权把梯度拉到正类上

### Q4: 为什么选 FT-Transformer 和 DeepFM？
核心选择标准是**两种正交的归纳偏置**：
- **FTT（自注意力范式）**：全局、动态、二阶以上交互，能学条件性交互
- **DeepFM（显式交互范式）**：FM 对所有特征对做固定内积 + Deep MLP 隐式高阶
- **排除其他**：MLP(无显式交互太弱)、TabNet(种子污染bug+论文已撤稿)、NODE(可微树与蒸馏目标重叠)、SAINT(复杂度高收益不显著)

### Q5: 3B 为什么做差？"白费功夫"怎么回应？
负结果本身是学术贡献：
1. 没有 3B 的 -3.73pp，结论只能模糊说"蒸馏对 FTT 效果不好"
2. 有了 3A vs 3B 的 A/B 消融，才能精准定位：**KD 跨架构有效，TreeReg 才是毒药**
3. 结论从"蒸馏是架构依赖的" → "树先验的轴对齐与 Sparsemax 注意力存在结构性拮抗"
4. 如果只做 3A 不做 3B，结论会停留在"DeepFM+KD 比 FTT+KD 好"，而不知道为什么

### Q6: 置换重要性 6 模型对比的核心发现
1. **KD 拉近学生-教师特征排序**：DeepFM r=0.625→0.845
2. **TreeReg 制造畸形复刻**：r=0.964 看似完美拟合，但 V14 集中度 43.9%，丢失所有交互
3. **跨架构趋同**：FTT vs DeepFM r=0.639→0.806（各自加 KD 后）
4. 排序相关性用 **Spearman** 而非 Pearson（因为置换重要性数值尺度不同，只有排序顺序有意义）

### Q7: Sparsemax vs Softmax 的根本区别
- **Softmax**：`e^z_i / Σ e^z_j`，输入值接近时数学上必然趋近均匀分布 1/d，永远不能输出零
- **Sparsemax**：`max(z_i - τ, 0)`，欧氏投影到概率单纯形，低于阈值 τ 的直接砍成 0，强制选择
- **为什么有效**：PCA 空间中 29 个特征的 Query-Key 内积高度同质化，Softmax 后注意力在 [0.0327, 0.0342] 标准差仅 0.0004（接近均匀 1/29=0.0345），Sparsemax 通过排序 + 阈值截断强制模型聚焦于最相关的特征子集
- 效果：AUPRC +4.4pp（0.8401→0.9179），ECE ×20 改善，Cost -63%

### Q8: 经典蒸馏公式
`L = α·T²·KL(学生^T || 教师^T) + (1-α)·CE(学生, y_true)`
- T>1 软化概率分布，暴露类别间相对关系（暗知识）
- KL 散度衡量分布差距：`KL(P||Q) = ΣP(i)·log(P(i)/Q(i))`，非对称，≥0
- T² 是梯度缩放补偿
- 本项目用 FocalLoss 替换 CE，因数据不平衡

### Q9: 本项目蒸馏 vs 经典蒸馏的区别
- **公式相同**：骨架完全沿用 Hinton 2015
- **目的不同**：经典=大NN→小NN 压缩；本项目=XGBoost→NN 归纳偏置迁移
- **教师来源不同**：经典=Softmax 输出；本项目=300 棵树叶子值汇总
- **训练方式**：离线蒸馏——教师一次性预生成软标签
- **替换项**：CE→FocalLoss

### Q10: 为什么二分类存标量 p 而非 logits 向量？
因为二分类中一个标量 p 等价于二维向量 [p, 1-p]。多分类（如10类）才需要存10维向量。

### Q11: 此项目跟上 2026 年学术前沿了吗？
- FTT(2021)/DeepFM(2017) 作为课程项目足够
- 2026 前沿：TabPFN-3(基座模型)、TabICLv2(ICML 2026 上下文学习)、TabM(ICLR 2025)
- 项目亮点：TreeReg 负结果的结构性分析、A/B 消融设计、置换重要性 6 模型全量对比

### Q12: DeepFM+KD 精度最高(0.9495)但 Cost 不是最低(120 vs FTT+KD 97)，怎么解释？
- 精度-成本非单调：FTT 成本最低(99)但 AUPRC 排第 4(0.9179)
- DeepFM+KD：AUPRC 最高(0.9495)但 Cost=120，因 KD 提高了召回但引入少量 FP
- FTT+KD(3A only)：AUPRC=0.9347 但 Cost=97（最低），因 Sparsemax 天然稀疏选择减少 FP
- 业务含义：如果 FN 成本极高（漏一笔欺诈=巨大损失），选 DeepFM+KD；如果 FP 成本高（审核资源有限），选 FTT+KD

### Q13: XGBoost 教师 vs NN 学生，谁参数量更大？
**XGBoost 更大（~23万 vs NN ~15万）**，属于经典的大老师→小学生压缩场景。
- XGBoost 参数量估算：300棵树 × 深度8 → 每棵树约255个内部节点 × 3个值(分裂特征+分裂阈值+叶子值) ≈ 229,500
- FTT：153,921，DeepFM：161,648

### Q14: XGBoost 如何"选特征"？与 NN 的特征权重是一回事吗？
**不是一回事。** 这是本项目的核心概念辨析：
- **XGBoost**：用**轴对齐分裂 (axis-aligned split)**，每次问 `x_i > t?`，选的是**分裂维度和阈值**，由信息增益驱动。它没有"特征权重"这个概念。
- **NN 嵌入层**：每个特征 j 有一个可学习向量 W_j，通过梯度下降调整其方向和范数。这是真正的"特征权重"。
- **为什么 3B(TreeReg) 失败**：用 split gain 约束嵌入范数 = **用椅子的结构改造汽车的方向盘**——两种机制根本不兼容。

### Q15: 为什么不直接调 XGBoost，偏要折腾蒸馏？
**因为目标不是"工程精度"，而是"学术理解"。**
- 如果只看工程：XGBoost 0.9331 已经很好，调树数/深度/学习率更简单
- 我们要回答的研究问题是：**树模型的归纳偏置能否通过蒸馏迁移到异构深度架构中？**
- 结果确实有学术价值：(1) Sparsemax 修复 FTT 注意力塌缩的机制发现 (2) 3A vs 3B 消融揭示 KD 跨架构有效、TreeReg 结构性拮抗
- **答辩金句**：*"如果只为了精调 XGBoost，我们不需要建 6 个模型、做 200 多次实验。我们想做的是理解'为什么有些架构适合蒸馏，有些不适合'——3B 的失败和 3A 的成功加在一起，才给了我们完整的答案。"*

---

## 九、运行流程

```bash
# 完整管道
python code/data/01_download.py
python code/data/02_eda.py
python code/data/03_preprocess.py
python code/04_train_baselines.py
python code/05_train_neural.py
python code/05_train_distillation.py
python code/05_train_distillation_ablation.py
python code/06_evaluate.py
python code/07_interpret.py
python code/08_interpret_deepfm.py

# 编译论文
cd paper
xelatex -interaction=nonstopmode main.tex
biber main
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

**MiKTeX Portable 注意**：须用 `fontset=fandol`（已在 main.tex 中配置），因为 Portable 版无 Windows 系统字体。

---

## 十、环境

| 项目 | 版本 |
|------|------|
| OS | Windows 11 23H2 |
| Python | 3.13 |
| PyTorch | 2.11.0+cu128 |
| GPU | NVIDIA RTX 4060 (8GB), CUDA 13.1 |
| LaTeX | MiKTeX Portable + XeLaTeX |
| 依赖 | `requirements.txt` (torch/numpy/pandas/sklearn/xgboost/matplotlib/seaborn/shap/optuna) |

---

### Q16: Sparsemax 被砍成 0 的特征梯度怎么回传？会像 ReLU 一样死亡吗？
**不会，Sparsemax 的零 ≠ ReLU 的死亡。**
- ReLU 的零：梯度 = 0，从此与该神经元无关，永久死亡
- Sparsemax 的零：虽然输出为 0，但该位置的 logit z_i 依然通过**排序 + 阈值 τ 的计算**影响其他非零位置的权重
- 梯度路径：`z_i → 排序 → τ → 保留集合 → 非零位置的输出值`，PyTorch 自动微分会沿此链把梯度传回 z_i
- **直观类比**：ReLU 死亡 = 被开除，彻底与公司无关；Sparsemax 归零 = 降为观察员，仍坐在会议室里决定"留多少人开会"
- 训练现象：初始注意力分布分散，前几轮后快速收敛到 V14/V12/V4 等少数关键特征。被置零的特征梯度非零，因为它们参与 τ 计算，只是贡献从"直接输出"降级为"决定阈值"

### Q17: Sparsemax 和 Softmax 的根本区别 — 深层追问
**不仅仅是"Softmax 输出非零，Sparsemax 输出可零"——本质是投影空间不同**：
- Softmax：`e^z_i / Σ e^z_j` 是**信息论投影**（最大化熵），数学上永远不能输出精确零。等价于在 logits 空间做 soft 归一化
- Sparsemax：`max(z_i - τ, 0)` 是**欧氏投影**（最小化 L2 距离到概率单纯形），允许解落在单纯形边界上。τ 通过排序 + 截断自动确定
- **为什么在 PCA 空间有效**：29 维 PCA 特征的 Query-Key 内积高度同质化（标准差仅 0.0004），Softmax 后趋近均匀分布 1/29≈0.0345。Sparsemax 的欧氏投影强制截断低信息量特征，迫使注意力聚焦
- **代价**：Sparsemax 的排序操作 O(d log d) 比 Softmax O(d) 慢，但 d=29 时无感

### Q18: 置换重要性为什么用 Spearman 而非 Pearson？
**因为只有排序顺序有意义，数值尺度无意义：**
- 置换重要性本质：随机打乱某特征后模型性能下降多少。不同特征的下降幅度量级可能差 10 倍以上（V14 降 0.05，V28 降 0.002），但数值本身受模型随机性、打乱顺序影响
- **Pearson 相关系数**：衡量线性相关，对数值尺度敏感——A=[1,2,3], B=[10,20,30] 的 Pearson=1 但绝对数值差 10 倍
- **Spearman 秩相关系数**：只比较排名顺序——A=[1,2,3], B=[10,20,30] 的 Spearman=1（排名完全一致）
- **本项目场景**：KD 拉近学生-教师特征排序 0.625→0.845，说的是"师生模型认为最重要的特征排名趋于一致"，而非"数值接近"。如果数值接近，TreeReg 的 r=0.964 反而该是最好的
- **一句话**：Spearman 回答"是不是都觉得 V14 最重要"，Pearson 回答"觉得 V14 重要到多少程度"

### Q19: Focal Loss 的 α 为什么 FTT 和 DeepFM 不同？
**因为两个架构的"梯度预算分配机制"不同，α 是互补调参：**
- **FTT（Sparsemax）**：注意力机制已经在做特征选择（哪些特征值得关注），梯度压力天然偏正类。α=0.25 是为了**均衡正负梯度**，防止 Sparsemax 的稀疏选择被过高的正类梯度扭曲
- **DeepFM（无内置稀疏机制）**：FM 对所有特征对做内积，无任何选择偏好，梯度均匀分布在所有特征上。α=0.995 ≈ 类别反比加权（正样本权重 = 负样本数 / 正样本数 ≈ 199），**强行把梯度拉到正类上**
- **直觉类比**：
  - FTT = 已经知道哪些是嫌疑人的警探（注意力选择），只需告诉他"别只盯着一个人"（α=0.25 均衡）
  - DeepFM = 菜鸟警探对所有线索无差别调查，需要吼"给我盯紧那个嫌疑人！"（α=0.995 强加权）

### Q20: KD 蒸馏温度 T=4.0 怎么来的？大了/小了会怎样？
**温度 T 控制软标签的"光滑程度"——T 越大，软标签分布越均匀：**
- **T→1**：几乎等于硬标签，蒸馏退化为普通训练，教师信息几乎丢失
- **T=4.0**：合适范围，软标签概率分布被展宽，暴露类别间相对关系（"这个样本虽然是正类，但形状有点像负类"），给学生提供更多暗知识
- **T→∞**：概率分布趋近均匀，信息量趋零，学生什么都学不到
- **本项目验证**：在蒸馏实验中 T=4.0 是常用起点（Hinton 2015 原论文在 MNIST 上推荐 2~8），实际对比过 T=2/4/6 三组，T=4 AUPRC 最高
- **T² 补偿因子的数学原因**：蒸馏损失 L = α·T²·KL(p_s^T || p_t^T) + ...，T² 保证软标签梯度大小不受 T 选择影响——否则调 T 会同时改变信号强度和分布形态，难以分离

### Q21: 为什么二分类的 KL 散度和多分类公式不同？
**本质相同，只是二分类做了化简：**
- **多分类 KL**：`KL(P||Q) = Σᵢ P(i)·log(P(i)/Q(i))`，对每个类别 i 求和
- **二分类 KL**：`KL(P||Q) = P·log(P/Q) + (1-P)·log((1-P)/(1-Q))`
- **化简原理**：二分类中 P(正类)+P(负类)=1, Q(正类)+Q(负类)=1，所以 Σ²ᵢ₌₁ = 完整求和
- **为什么存标量 p 而非 2 维向量**：因为给定 p，(p, 1-p) 完全确定，存 2 维是冗余。模型中最后是 sigmoid 输出 1 个标量
- **扩展**：如果你做的是多分类 CIFAR-10，就需要存 10 维 softmax 输出

### Q22: 为什么本项目用离线蒸馏而非在线蒸馏？
**因为教师是 XGBoost，训练方式和 NN 完全不同：**
- **离线蒸馏**：教师先跑完所有数据 `predict_proba()`，一次性生成软标签存成 `.npy` 文件，学生训练时直接读取。**解耦训练流程**
- **在线蒸馏**：教师和学生同时训练，教师每步输出 logits 给学生。这就要求老师和学生能在同一个训练循环里前向、反向
- **为什么不行**：XGBoost 训练是逐树添加（boosting），每一步在 CPU 上做贪心分裂搜索，没有"可微 logits"的概念。PyTorch Student 在 GPU 上跑 mini-batch SGD。两者的训练循环无法对齐
- **本质**：离线蒸馏 = 老师和学生**不同时间上课**（老师先录好课，学生回放）；在线蒸馏 = 老师和学生**同堂辩论**（NN 当老师才能这么干）

### Q23: 为什么说 KD 是"归纳偏置迁移"而非"知识迁移"？
**因为迁移的不是具体知识，而是决策边界的形状/倾向：**
- **知识迁移**（狭义）：教师告诉学生"这个样本是欺诈的概率是 92%"。这只是**信息压缩**——学生学到的信息量 <= 教师含有的信息量
- **归纳偏置迁移**：树模型的归纳偏置是**轴对齐的分段常数决策边界**（沿 x_i 轴分裂），NN 的归纳偏置是**光滑的非线性流形**。蒸馏让学生看到教师的软标签分布——相当于让 NN 的决策边界去拟合树模型的 "形状倾向"
- **为什么有价值**：树模型擅长表格数据（轴对齐分裂天然适合），NN 擅长泛化和特征交互。蒸馏让 NN 吸收树模型的"轴对齐倾向"到自己的特征表示中，把两种归纳偏置融合
- **本项目验证**：置换重要性发现 KD 后 DeepFM 的特征排序从 r=0.625 拉近到 0.845——学生的"特征偏好"向老师的"分裂偏好"靠拢，这就是归纳偏置迁移的定量证据

### Q24: DeepFM+KD 精度最高(0.9495)为什么 Cost 不是最低？
**因为 AUPRC 和 Cost 优化的目标不同，不存在单调关系：**
- **AUPRC**：关注 Precision-Recall 曲线下的面积，衡量排序质量——只要模型把正类排在负类前面就行，**不管阈值在哪里**
- **Cost**：`FP×1 + FN×10`，是**固定阈值决策**的代价，高度依赖阈值选择
- **为什么 DeepFM+KD 的 Cost 偏高**：KD 提高了召回（更多真欺诈被检出），但也引入了少量额外 FP（正常交易被误判为欺诈）。FN 的权重(10)远大于 FP(1)，所以多召回 1 个欺诈（减 10 成本）如果引入 2~3 个 FP（加 2~3 成本），总成本确实降低了，但如果 FP 更多就会反弹
- **FTT+KD(3A only) Cost 最低**：Sparsemax 的稀疏注意力天然减少 FP——它只盯着少数强特征做判断，泛化时更保守，宁可漏也不轻易判欺诈。这种保守主义在 AUPRC 上吃亏（排第 5），但在成本上最优（排第 1）
- **业务解读**：
  - 如果你**不能漏任何欺诈**（银行风控），选 DeepFM+KD，AUPRC 最高，召回最强
  - 如果你**审核人力有限**（FP 成本高），选 FTT+KD，Cost 最低，FP 最少
  - 没有正确答案，只有业务选择

### Q25: 为什么不直接用 XGBoost 的 predict_proba 还要蒸馏？

### Q26: XGBoost scale_pos_weight 怎么算的？和 Focal Loss α 是一回事吗？
**不是一回事，作用阶段不同：**
- **scale_pos_weight = 负样本数 / 正样本数**（本项目约 199）
  - 作用位置：XGBoost 的叶子值计算中，正样本的**一阶梯度 g_i 和二阶梯度 h_i 均乘以 w**
  - 叶子值公式：`w_j = - (Σg_i) / (Σh_i + λ)` → 加权后 `w_j = - (Σ_{负}g_i + w·Σ_{正}g_i) / (Σ_{负}h_i + w·Σ_{正}h_i + λ)`
  - 本质：**在二阶优化的树分裂阶段**给正样本放大梯度预算
- **Focal Loss α**
  - 作用位置：损失函数系数 `FL(p_t) = -α_t·(1-p_t)^γ·log(p_t)`
  - 本质：**在前向/反向传播阶段**控制正负类损失的权重分配
- **关键区别**：
  - scale_pos_weight 是**固定权重 × 梯度**，无 γ 配合，作用于树结构学习
  - α 是**损失系数 + γ 聚焦控制**，作用于 NN 的梯度下降
- **直觉类比**：
  - scale_pos_weight = 给正样本配了 199 倍砝码，称重时天平直接倾斜
  - Focal Loss α = 先称完，再根据结果乘系数决定"这次称的重要程度"
  - 一个是**加料时就加重**，一个是**出锅后调味**
**因为目标不是工程精度，是学术因果理解——这是整篇论文的灵魂问题：**
- 如果只求精度：XGBoost 0.9331 已经很好，调几棵树就够了
- 我们问的是：**树模型的归纳偏置能不能**跨架构**迁移到 NN 里？哪些架构适合？哪些机制会阻断迁移？**
- 要回答这个问题必须做 3 件事：
  1. **选两种正交架构**（FTT 自注意力 vs DeepFM 显式交互）——这样才能说"跨架构"
  2. **做 A/B 消融**（3A vs 3B）——没有 3B 的 -3.73pp 断崖，结论只能是模糊的"蒸馏对 FTT 没多大用"
  3. **置换重要性 6 模型全量对比**——定量看到 KD 把特征排序从 0.625 拉到 0.845
- **答辩金句**：*"如果只为了精调 XGBoost，我们不需要建 6 个模型、做 200 多次实验。我们想做的是理解'为什么有些架构适合蒸馏，有些不适合'——3B 的失败和 3A 的成功加在一起，才给了我们完整的答案。"*

---

> ⚠ 此文件为 OpenCode 项目知识库，覆盖项目全景。新对话时阅读此文件即可快速恢复上下文。
> 对应代码位置：`D:\商业智能\商业智能大作业\`
