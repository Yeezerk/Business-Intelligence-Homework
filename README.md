# 基于深度神经网络的信用卡欺诈检测

> 商业智能（Business Intelligence）课程大作业 · 深度学习方向

**作者**：Yeezerk  
**导师评估入口**：https://github.com/Yeezerk/Business-Intelligence-Homework

---

## 研究概述

本研究系统比较了两种异构深度表格架构在信用卡欺诈检测任务上的表现，并提出 **Tree-Supervised 知识蒸馏方案**，将 XGBoost 的树归纳偏置迁移至深度模型。

### 核心发现

| 发现 | AUPRC 影响 |
|------|-----------|
| **Sparsemax 替代 Softmax** 解决注意力均匀塌缩 | +4.4pp |
| **软标签蒸馏 (KD)** 对两种架构均有正向增益 | FTT +1.68pp, DeepFM +1.59pp |
| **特征重要性正则化 (TreeReg)** 对抗自注意力自由学习 | -3.73pp（结构性拮抗） |

### 最终模型排行

| 模型 | AUPRC | 角色 |
|------|-------|------|
| 🥇 DeepFM + KD | 0.9495 | 显式特征交互 + 蒸馏 |
| 🥈 FTT (Sparsemax) + KD | 0.9347 | 自注意力 + 蒸馏 |
| 🥉 DeepFM | 0.9336 | 显式特征交互基线 |
| XGBoost | 0.9331 | 树基线 / 蒸馏教师 |

---

## 项目结构

```
├── README.md                         ← 本文档（项目说明）
├── PROJECT_GUIDE.md                  ← AI 协作开发指南（可忽略）
├── requirements.txt                  ← Python 依赖
│
├── 📄 paper/                         ← 原始论文（LaTeX）
│   ├── main.tex                      # 主文档
│   ├── chapters/                     # 各章节 .tex
│   ├── figures/                      # 论文插图
│   └── references.bib                # 参考文献
│
├── 📄 revision/                      ← 论文修订版 + 扩展实验
│   ├── paper/                        # 修订版论文
│   ├── code/                         # LOR-MIM / Sparsemax / Entmax 实验
│   ├── notes/                        # 答辩准备笔记
│   └── outputs/                      # 修订版结果图表
│
├── 📊 code/                          ← 原始实验代码
│   ├── data/                         # 数据下载、EDA、预处理
│   ├── models/                       # FT-Transformer、DeepFM
│   ├── training/                     # Trainer、FocalLoss、蒸馏、评估指标
│   ├── 04_train_baselines.py         # XGBoost 基线
│   ├── 05_train_neural.py            # 神经网络基线
│   ├── 05_train_distillation.py      # Tree-Supervised 蒸馏
│   ├── 06_evaluate.py                # 评估可视化
│   └── 07_interpret.py / 08_*.py     # SHAP/注意力/置换重要性
│
├── 📝 notes/                         ← 技术笔记（11篇）
│   ├── 00_项目全景与流程.md
│   ├── 01_类别不平衡处理全景.md
│   ├── 02_Focal_Loss深度解析.md
│   ├── 03_SAINT论文解读.md
│   ├── 04_FT_Transformer原理.md
│   ├── 05_DeepFM与特征交叉.md
│   ├── 06_评估指标的商业视角.md
│   ├── 07_SHAP模型可解释性.md
│   ├── 08_XGBoost_vs_深度学习_表格数据.md
│   ├── 09_Transformer原理详解.md
│   ├── 09_树监督学习与归纳偏置迁移.md
│   └── 12_答辩准备_核心知识体系.md ⭐
│
└── 📈 outputs/                       ← 原始实验输出
    ├── figures/                      # 评估图（ROC/PR/SHAP）
    └── results/                      # 指标 CSV
```

---

## 技术创新点

1. **Sparsemax 注意力**：替代传统 Softmax，解决 PCA 空间中注意力均匀塌缩问题
2. **Tree-Supervised Knowledge Distillation**：方案 3A（软标签蒸馏）跨架构有效，方案 3B（TreeReg）对注意力机制有毒
3. **AUPRC 为主要评估指标**：在不平衡数据（欺诈率 0.5%）上比 ROC-AUC 更有区分度
4. **成本矩阵评估**：`Cost = FP × 1 + FN × 10`，反映商业场景中漏检欺诈的实际代价
5. **三重可解释性**：SHAP + 注意力权重 + 置换重要性，统一框架下对比 6 个模型

---

## 环境与运行

### 依赖

```bash
pip install -r requirements.txt
```

主要框架：PyTorch, XGBoost, Scikit-learn, SHAP, Matplotlib

### 运行流程

```bash
# 1. 数据准备（需 Kaggle 凭证或直接放入 outputs/raw/）
python code/data/01_download.py   # 下载
python code/data/02_eda.py         # 探索性分析
python code/data/03_preprocess.py  # 预处理

# 2. 训练
python code/04_train_baselines.py  # XGBoost/LogisticRegression/RF
python code/05_train_neural.py     # FT-Transformer / DeepFM

# 3. 蒸馏
python code/05_train_distillation.py  # Tree-Supervised KD

# 4. 评估与分析
python code/06_evaluate.py         # PR/ROC/Cost/Calibration 图表
python code/07_interpret.py        # SHAP + 注意力可视化
python code/08_interpret_deepfm.py # 置换重要性对比
```

### 论文编译

```bash
cd paper/
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

---

## 数据来源

Kaggle — [Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)  
原始 568,630 条 × 31 列（V1-V28 PCA 特征 + Amount + Class），预处理时人工制造 0.5% 欺诈率的不平衡分布。
