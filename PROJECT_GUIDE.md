# 商业智能大作业 — 项目全景指引

> 当开启新对话时，将此文件内容提供给 Claude，即可快速恢复上下文，继续项目开发。

---

## 一、项目身份

**课程**：商业智能（Business Intelligence）— 机器学习/深度学习方向  
**选题**：基于深度神经网络的信用卡欺诈检测  
**目标**：用 LaTeX 写论文（10-15 页），导出 PDF  
**数据**：Kaggle — [Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)（原始 568,630 条 × 31 列：id + V1-V28 + Amount + Class。注意：原始数据为 50:50 均衡版本，预处理时人工制造 0.5% 欺诈率的不平衡）  
**技术水平**：进阶 — 熟练 PyTorch，能设计和调优神经网络架构  
**周期**：约 1 个月  
**语言**：中文（论文 + 笔记），代码注释为英文  

---

## 二、核心设计思路

**论文核心论证**：比较两种异构深度表格架构（FT-Transformer 的自注意力 vs. DeepFM 的显式特征交互）与 XGBoost 基线的性能，提出 Tree-Supervised 知识蒸馏方案将树归纳偏置迁移至深度模型。核心发现：(1) 软标签蒸馏 (3A) 对两种架构具有同等正向增益——FTT+KD=0.9347 (+1.68pp), DeepFM+KD=0.9495 (+1.59pp)；(2) 特征重要性正则化 (TreeReg, 3B) 是 FTT 蒸馏退步的唯一元凶——从 0.9347 断崖降至 0.8974 (-3.73pp)，揭示了树先验对嵌入层的刚性约束与 Sparsemax 注意力的自由交互学习之间的结构性拮抗；(3) Sparsemax 替代 Softmax 使 FTT 的 AUPRC 提升 4.4pp、ECE 改善 20 倍。

**最终模型线**：

| 模型 | 参数量 | 角色 | AUPRC | 状态 |
|------|--------|------|-------|------|
| XGBoost | — | 树基线 / 蒸馏教师 | 0.9331 | ✅ |
| FT-Transformer (Sparsemax) | 153,921 | 自注意力范式 | 0.9179 | ✅ |
| DeepFM | 161,648 | 显式交互范式 | 0.9336 | ✅ |
| **FTT + KD (3A only)** | 153,921 | 软标签蒸馏消融 | **0.9347** | ✅ 🥈 |
| **DeepFM + KD** | 161,648 | 软标签蒸馏 | **0.9495** | ✅ 🥇 |
| FTT + KD + TreeReg | 153,921 | 完整 Tree-Supervised | 0.8974 | ⚠️ TreeReg 拮抗 |

**核心技术创新点**：
- **Sparsemax** 替代 Softmax：解决 PCA 空间中注意力均匀塌缩（AUPRC +4.4pp, ECE ×20）
- **Tree-Supervised KD**：方案 3A（软标签蒸馏）跨架构有效（FTT +1.68pp, DeepFM +1.59pp），方案 3B（TreeReg）对注意力有毒（-3.73pp）
- **置换重要性全模型对比**：XGBoost/DeepFM/FTT 及蒸馏变体（DFM+KD, FTT+KD, FTT+KD+TreeReg）共 6 模型统一框架对比——KD 拉近学生-教师排序（DeepFM +0.220）、TreeReg 制造 XGBoost 复刻但 V14 集中度 43.9% 致性能崩塌
- AUPRC 为主要评估指标（ROC-AUC 在不平衡数据上区分度低）
- 成本矩阵评估：`Cost = FP×1 + FN×10`
- SHAP + 注意力 + 置换重要性三重可解释性

---

## 三、目录结构与文件清单

```
商业智能大作业/
│
├── PROJECT_GUIDE.md              ← 【本文档】新对话时提供此文件
├── requirements.txt              # Python 依赖
│
├── transfer/                     # 参考论文转换产物
│   └── SAINT.md                  # MinerU flash-extract 转换的 SAINT 论文
│
├── notes/                        # 学习笔记 (11篇 Markdown)
│   ├── 00_项目全景与流程.md        # ML 项目生命周期、运行顺序（已更新至最终状态）
│   ├── 01_类别不平衡处理全景.md     # 三大流派对比（SMOTE/Weight/Focal）
│   ├── 02_Focal_Loss深度解析.md    # 数学推导 + 调参哲学
│   ├── 04_FT_Transformer原理.md    # Feature Tokenizer/CLS/Sparsemax（含最终效果）
│   ├── 05_DeepFM与特征交叉.md      # FM O(DK)推导 + Deep联合训练
│   ├── 06_评估指标的商业视角.md     # AUPRC vs ROC-AUC/成本矩阵
│   ├── 07_SHAP模型可解释性.md      # Shapley值/TreeSHAP/瀑布图
│   ├── 08_XGBoost_vs_深度学习_表格数据.md  # 何时用树、何时用NN
│   ├── 09_Transformer原理详解.md   # 通用Transformer参考
│   ├── 09_树监督学习与归纳偏置迁移.md # 蒸馏方案3AB + 最终实测结果
│   └── 12_答辩准备_核心知识体系.md ⭐ # 故事线→模型→创新→Q&A→记忆清单
│
├── code/                         # Python 代码
│   ├── config.py                 # 全局路径/超参/随机种子/DEVICE
│   ├── data/
│   │   ├── 01_download.py        # KaggleHub 下载数据集
│   │   ├── 02_eda.py             # EDA → outputs/figures/*_eda.pdf
│   │   └── 03_preprocess.py      # 预处理 → outputs/processed/{train,val,test}.pt
│   ├── models/
│   │   ├── ft_transformer.py     # FT-Transformer + Sparsemax + TreeReg
│   │   └── deepfm.py             # DeepFM (FM + Deep MLP)
│   ├── training/
│   │   ├── trainer.py            # 统一训练循环 + 早停 + 模型保存
│   │   ├── losses.py             # FocalLoss
│   │   ├── distillation.py       # KD Loss + TreeFeatureRegularization（方案3AB）
│   │   └── metrics.py            # AUPRC/ROC-AUC/F1/Cost/ECE
│   ├── 04_train_baselines.py     # XGBoost → outputs/models/
│   ├── 05_train_neural.py        # FTT + DeepFM 基线训练（含 reset_seed）
│   ├── 05_train_distillation.py          # Tree-Supervised 蒸馏训练（方案3AB）
│   ├── 05_train_distillation_ablation.py  # FTT+KD (3A only) 消融实验
│   ├── 06_evaluate.py                     # PR/ROC/CM/Cost/Calibration + LaTeX表（已去TabNet）
│   ├── 07_interpret.py                    # SHAP + FTT attention
│   └── 08_interpret_deepfm.py             # 置换重要性：XGBoost/DeepFM/FTT + 蒸馏变体共 6 模型全量对比
│
├── paper/                        # LaTeX 论文
│   ├── main.tex                  # 主文档（ctexart, XeLaTeX）
│   ├── chapters/
│   │   ├── abstract.tex          # 中英双语摘要
│   │   ├── introduction.tex      # 引言（背景、研究问题、贡献）
│   │   ├── related_work.tex      # 相关工作
│   │   ├── methodology.tex       # 方法论（预处理、模型、训练、评估）
│   │   ├── experiments.tex       # 实验（性能对比、消融）
│   │   ├── analysis.tex          # 分析（可解释性、成本、模型选择建议）
│   │   ├── discussion.tex        # 讨论（商业启示、局限性、未来工作）
│   │   └── conclusion.tex        # 结论
│   ├── figures/                  # 论文插图（矢量 PDF）
│   ├── tables/                   # LaTeX 表格文件
│   └── references.bib            # IEEE 格式参考文献（15 篇，含 2022-2024 进展）
│
└── outputs/                      # 生成产物（.gitignore）
    ├── raw/                      # 原始 CSV
    ├── processed/                # train.pt, val.pt, test.pt, preprocessor.pkl
    ├── models/                   # *.pt (NN), *.pkl (sklearn), *.json (XGBoost)
    ├── figures/                  # 生成的评估图 + SHAP 图
    └── results/                  # metrics.csv, model_comparison.tex
```

---

## 四、运行流程

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载数据（需要 Kaggle API key 或直接下载 CSV 放入 outputs/raw/）
python code/data/01_download.py

# 3. 探索性分析（查看数据分布、类别平衡等）
python code/data/02_eda.py

# 4. 预处理（清洗、标准化、划分，产出 .pt 文件）
python code/data/03_preprocess.py

# 5. 训练传统基线（LR, RF, XGBoost）
python code/04_train_baselines.py

# 6. 训练神经网络（MLP, TabNet, FT-Transformer, DeepFM）
python code/05_train_neural.py

# 7. 评估可视化（生成论文所需图表）
python code/06_evaluate.py

# 8. 可解释性分析（SHAP + 注意力可视化）
python code/07_interpret.py

# 9. 编译论文
cd paper
# 注意：MiKTeX Portable 需使用 fontset=fandol，已在 main.tex 中配置
D:/MiKTeX/texmfs/install/miktex/bin/x64/xelatex.exe -interaction=nonstopmode main.tex
D:/MiKTeX/texmfs/install/miktex/bin/x64/biber.exe main
D:/MiKTeX/texmfs/install/miktex/bin/x64/xelatex.exe -interaction=nonstopmode main.tex
D:/MiKTeX/texmfs/install/miktex/bin/x64/xelatex.exe -interaction=nonstopmode main.tex
```

---

## 五、关键技术决策（为什么这样设计）

1. **AUPRC 而非 ROC-AUC 为主指标**：欺诈仅占 0.5%，ROC-AUC 被 TN 主导（三模型间极差仅 0.0054，无法区分），AUPRC 区分度约 3 倍。详见 `notes/06`

2. **Sparsemax 替代 Softmax**：Softmax 永远输出非零概率，PCA 空间中所有特征嵌入同质化 → Query-Key 内积趋同 → 注意力塌缩为均匀分布。Sparsemax 可输出真正零值，强制模型做选择。效果：AUPRC +4.4pp, ECE ×20 改善, Cost -63%。

3. **异构架构对比而非枚举架构**：FT-Transformer（全局自注意力）和 DeepFM（显式特征交互）代表了两种根本不同的归纳偏置。论文的核心叙事不是"哪种架构更好"，而是"归纳偏置如何影响蒸馏策略"。

4. **Focal Loss 的 α 按架构定制**：FTT（Sparsemax 注意力）用 α=0.25（均衡梯度），DeepFM（无内置稀疏）用 α=0.995（类别反比强加权）。

5. **成本矩阵 FP=1, FN=10**：漏报欺诈的直接资金损失远大于误报的审核成本。FTT 成本最低（99），DeepFM+KD 精度最高但成本 120——揭示精度-成本的非单调关系。

---

## 六、维护约定

> **每次交互后，此文档的"当前进度"章节都会被更新**，确保始终反映最新状态。
> 当对话轮次超过 15-20 轮或上下文接近上限时，Claude 会主动提醒开启新对话。

---

## 七、当前进度

| 模块 | 状态 | 说明 |
|------|------|------|
| 数据+预处理+EDA | ✅ | 完整管道，产出 train/val/test.pt |
| XGBoost 基线 | ✅ | AUPRC=0.9331 |
| FT-Transformer (Sparsemax) | ✅ | AUPRC=0.9179, ECE=0.0120 |
| DeepFM | ✅ | AUPRC=0.9336 ≈ XGBoost |
| Tree-Supervised 蒸馏 | ✅ | DeepFM+KD=0.9495🥇, FTT+KD=0.9347🥈, FTT+KD+TreeReg=0.8974 |
| 蒸馏消融 (FTT+KD) | ✅ | **新增**：证实 KD 跨架构有效，TreeReg 是退步唯一元凶 |
| 置换重要性全模型对比 | ✅ | **更新**：6模型含蒸馏变体——KD拉近排序、TreeReg制造复刻、跨架构趋同 |
| 评估+图表 | ✅ | 全部图表已去 TabNet 重新生成，含置换重要性图 |
| 论文 | ✅ | 27页, 0错误, 审稿修改已执行, 消融分析完成 |
| 引用合规 | ✅ | 18条引用, 0缺失, bib类型已修正 |
| 笔记 | ✅ | 11篇, 含答辩准备笔记 |
| 种子可复现性 | ✅ | reset_seed 确保独立可复现 |

---

## 八、学习笔记导航

```
00_项目全景与流程         → 先建立全局认知（已更新至最终状态）
01_类别不平衡处理全景      → 理解核心挑战
02_Focal_Loss深度解析      → 核心解决方案
04_FT_Transformer原理      → 模型原理（含 Sparsemax + 实证效果）
05_DeepFM与特征交叉        → 模型原理
06_评估指标的商业视角      → 如何评判好坏
07_SHAP模型可解释性        → 解释模型决策
08_XGBoost_vs_深度学习     → 退一步看全局
09_树监督学习与归纳偏置     → 蒸馏方案 3AB + 最终实测结果
09_Transformer原理详解     → 通用 Transformer 参考
12_答辩准备_核心知识体系 ⭐ → 故事线→模型→创新→Q&A→记忆清单
```

每篇笔记遵循 **"为什么 → 是什么 → 怎么实现 → 与项目代码的对应关系"** 的四层结构。

---

## 九、如何自己编辑论文

所有论文源文件在 `paper/` 下，都是纯文本 `.tex` 文件，VS Code 直接打开编辑。

### 文件与内容对照

| 文件                              | 内容                           |
| --------------------------------- | ------------------------------ |
| `paper/main.tex`                  | 文档类、包、标题、作者         |
| `paper/chapters/abstract.tex`     | 中英双语摘要                   |
| `paper/chapters/introduction.tex` | 引言                           |
| `paper/chapters/related_work.tex` | 相关工作                       |
| `paper/chapters/methodology.tex`  | 方法论                         |
| `paper/chapters/experiments.tex`  | 实验（结果表格、消融实验分析） |
| `paper/chapters/discussion.tex`   | 讨论（商业启示、局限性）       |
| `paper/chapters/conclusion.tex`   | 结论                           |
| `paper/references.bib`            | 参考文献库（15 篇）            |

### 常用 LaTeX 语法

| 效果         | 写法                                                     |
| ------------ | -------------------------------------------------------- |
| 一级标题     | `\section{标题}`                                         |
| 二级标题     | `\subsection{标题}`                                      |
| 引用文献     | `\cite{key}`（key 见 references.bib 每行第一个 `{key,`） |
| 加粗         | `\textbf{文字}`                                          |
| 引用图       | `图\ref{fig:pr_curves}`                                  |
| 引用表       | `表\ref{tab:model_comparison}`                           |
| 插入图片     | `\includegraphics[width=0.8\textwidth]{xxx.pdf}`         |
| 强制图片位置 | 用 `\begin{figure}[H]`                                   |

### 编辑后编译

```bash
cd paper
XE="D:/MiKTeX/texmfs/install/miktex/bin/x64/xelatex.exe"
BI="D:/MiKTeX/texmfs/install/miktex/bin/x64/biber.exe"
"$XE" -interaction=nonstopmode main.tex
"$BI" main
"$XE" -interaction=nonstopmode main.tex
"$XE" -interaction=nonstopmode main.tex
```

> 编译后 `main.pdf` 在 `paper/` 下。
> 如果改了参考文献（引用条数变化）才需要跑 biber，平时只改正文跑一次 xelatex 就够了。
> `fontset=fandol` 确保 MiKTeX Portable 无系统字体也能编译中文。

---

## 十、常见问题

**Q: 编译时出现 "CTeX fontset `windows' is unavailable" 错误？**  
A: 因为 MiKTeX Portable 没有 Windows 系统字体。已在 `main.tex` 中将 `\documentclass{ctexart}` 改为 `fontset=fandol`（MiKTeX 自带的开源中文字体）。如果你用系统安装的 MiKTeX 或 TeX Live，可改回默认（删掉 `fontset=fandol`）。

**Q: 为什么数据集有 PCA 匿名化特征但论文还提 SHAP 的业务解读？**  
A: 这是一个已知局限。笔记 `07` 和论文 `discussion.tex` 中都讨论了这一点。SHAP 在方法论层面仍然有效（可以按重要性排序特征），但无法将 V4 映射为"交易时间"或"商户类别"这样的业务概念。

**Q: config.py 中的超参需要调整吗？**
A: 当前超参是基于论文推荐和经验设置的合理默认值。如果需要更好的性能，可以用 Optuna（`pip install optuna`，已在 requirements 中）对 learning_rate、dropout、hidden_dims 做网格/贝叶斯搜索。

**Q: 没有 GPU 怎么办？**
A: `config.py` 中的 `DEVICE` 会自动检测 CUDA 可用性。CPU 上训练 55 万样本会比较慢。建议用 Google Colab 免费 GPU。

**Q: 如何新增一个模型？**
A: 1) 在 `code/models/` 添加 `new_model.py`；2) 在 `05_train_neural.py` 末尾加一个 `train_and_eval("new_model", ...)` 调用；3) 在 `06_evaluate.py` 的 `load_neural_model()` 和 `get_all_predictions()` 中注册新模型名。由于 trainer 是模型无关的，不需要修改训练逻辑。

**Q: 怎么重新生成图表？**
A: 重新跑对应脚本即可：
```bash
python code/06_evaluate.py   # 评估图 + LaTeX 表
python code/07_interpret.py   # SHAP + 注意力图
cp outputs/figures/*.pdf paper/figures/
```

---

## 十点五、论文写作 Skills 自动调用映射表

> 以下规则供 Claude 在每次对话时自动匹配。当用户提到相应任务时，Claude 必须优先调用对应的 Skill。

| 用户说（关键词触发） | 自动调用 Skill | 说明 |
|---------------------|---------------|------|
| 写论文/写大作业/写报告/写摘要/写引言/写讨论/写结论/搭框架/起草论文 | `academic-paper` | 三步直出法：跳过中间产物，直接产出 .tex |
| 润色/改写/学术表达/中文学术写作/论文语言 | `nature-polishing` | 加载 paper_type=research, language=zh |
| 审稿/审查论文/找论文问题/投稿前自审 | `nature-reviewer` | 3份审稿报告+交叉综合 |
| 论文配图/科研绘图/出图/画图/图表优化 | `nature-figure` | Python matplotlib 优先 |
| 加引用/补文献/找支撑/配文献/学术引用 | `nature-citation` | Nature/CNS系列引用 |
| 查文献/文献检索/找论文/文献综述检索 | `nature-academic-search` | PubMed/CrossRef/arXiv |
| 编译论文/编译LaTeX/重新编译/PDF导出 | `latex-workshop-config` | 自动用 XeLaTeX + fontset=fandol |
| 论文排版/LaTeX布局/页面松散/图位置/压缩页数 | `nature-polishing` `references/latex-layout.md` | Float优化、页边距、行距 |
| 论文做PPT/组会汇报/文献汇报/学术汇报 | `nature-paper2ppt` | 自动生成PPTX |

**优先级规则**：
1. 如果任务同时匹配 `academic-paper` 和其他 skill → 先调 `academic-paper` 建立框架，再调其他
2. 如果任务是纯编译/配置 → 调 `latex-workshop-config`
3. 如果任务是纯内容生成 → 调 `nature-writing`
4. 每次修改 .tex 后必须编译验证（`xelatex -interaction=nonstopmode main.tex`）

---

## 十一、本项目论文的学术规范标准（2026-05-30 更新）

### 参考论文

本项目论文参考了以下高水平学术论文的结构和规范：

| 参考论文 | 关键特点 | 如何参考 |
|----------|----------|----------|
| **Gorishniy et al. 2021** ("Revisiting Deep Learning Models for Tabular Data") | 15随机种子+Wilcoxon检验；Experiments/Analysis分离；Takeaway总结；符号定义表 | 论文结构（分章、符号表、多维度评估） |
| **Somepalli et al. 2022** ("SAINT", SAINT原始论文) | 图表caption极简（1-2句纯描述）；分析全部放在正文；表格紧凑；figure浮动放置 | caption精简至100字内、[H]→[htbp]浮动、缩减figure宽度 |

### 已实施的学术规范优化

| 优化项 | 位置 | 说明 |
|--------|------|------|
| **编号贡献列表** | `introduction.tex` 末尾 | 4条编号贡献（系统性模型对比、消融研究、多维可解释性、成本敏感评估），每条带具体数据 |
| **Experiments/Analysis分离** | `main.tex` | 实验(experiments.tex)与分析(analysis.tex)分开，符合参考论文结构 |
| **Takeaway总结** | 各subsection末 | 每个实验/分析subsection末尾添加 `\paragraph{主要发现}` 总结 |
| **形式化符号表** | `methodology.tex` 新增 | `tab:notation` 包含13个核心符号（数据集/特征/模型/阈值/指标/成本/Shapley值） |
| **局限性增强** | `discussion.tex` | 6条局限性（新增"缺乏统计显著性检验"），每条含详细说明 |

### LaTeX图表规范标准（参考 SAINT 2022 格式）

**Caption 规则**：纯描述性，1-2句话，30-100中文字。不放置分析解读（放在正文）。
**正文规则**：`如图\ref{fig:xxx}所示` 引导后，在正文段落中展开分析。
**Figure 放置**：使用 `[htbp]` 浮动（非 `[H]` 强制），宽度 0.72-0.82\textwidth。

**示例**：
```latex
% Caption（精简到纯描述）
\caption{各模型的Precision-Recall曲线。图例标注AUPRC值，XGBoost在多数召回区间保持最高精确率。}

% 正文分析（正文中展开）
如图\ref{fig:pr_curves}所示，XGBoost曲线在低-中召回区域(0.6-0.9)优势明显，
而TabNet曲线整体位置偏低，所有曲线在高召回端(>0.95)均急剧下降。
```

### 论文当前结构

```
paper/chapters/
├── abstract.tex          # 中英双语摘要
├── introduction.tex      # 引言（含编号贡献列表）
├── related_work.tex       # 相关工作
├── methodology.tex        # 方法论（含符号定义表 tab:notation）
├── experiments.tex        # 实验（设置、性能对比、消融、Takeaway）
├── analysis.tex           # 分析（可解释性、成本分析、模型选择建议）
├── discussion.tex         # 讨论（商业启示、局限性、未来工作）
└── conclusion.tex         # 结论
```

---

## 十二、给新对话的指引

如果你在**新对话**中，请告诉 Claude：

> 我正在做一个商业智能课程的大作业项目。项目全景在 `PROJECT_GUIDE.md` 中。请先阅读这个文件（或者我已经把内容粘贴在下面了），然后帮助我 [在这里描述你的具体需求]。

这样 Claude 就能快速理解项目，继续回答问题和修改代码。

---

## 十三、最终状态（2026-06-10 第四次对话完成）

### 最终性能排名

| 排名 | 模型 | AUPRC | ROC-AUC | F1 | Cost | ECE |
|------|------|-------|---------|-----|------|-----|
| 🥇 | DeepFM + KD | 0.9495 | 0.9988 | 0.9058 | 120 | 0.0008 |
| 🥈 | FTT + KD (3A only) | 0.9347 | 0.9953 | 0.9279 | 97 | 0.0007 |
| 🥉 | DeepFM | 0.9336 | 0.9979 | 0.8938 | 123 | 0.0394 |
| 4 | XGBoost | 0.9331 | 0.9979 | 0.8750 | 215 | 0.0006 |
| 5 | FT-Transformer | 0.9179 | 0.9925 | 0.9196 | 99 | 0.0120 |
| 6 | FTT + KD + TreeReg | 0.8974 | 0.9945 | 0.8848 | 169 | 0.0010 |

### 环境
- PyTorch 2.11.0+cu128, CUDA 13.1, GPU: RTX 4060 (8GB)
- FT-Transformer/DeepFM 手写实现

### 核心发现（v2 — 消融实验后修正）
1. **DeepFM ≈ XGBoost**：差距仅 0.0005，统计不可区分
2. **Sparsemax 拯救了 FTT**：AUPRC 0.8401→0.9179 (+4.4pp), ECE 0.2328→0.0120 (×20)
3. **KD 跨架构有效，TreeReg 是唯一元凶**：FTT+KD=0.9347 (+1.68pp) 证明软标签蒸馏对注意力型架构同样高效；TreeReg 将 FTT 从 0.9347 拖降至 0.8974 (-3.73pp)
4. **KD 提升校准质量**：所有 KD 变体均达 ECE < 0.002
5. **置换重要性全模型对比**：6 模型（含 DFM+KD, FTT+KD, FTT+KD+TreeReg）统一框架——KD 拉近学生-教师特征排序（DeepFM r 0.625→0.845, +0.220），TreeReg 制造 XGBoost 完美复刻（r=0.964）但 V14 集中度 43.9% 致性能崩塌，KD 提升跨架构行为相似性（r 0.639→0.806）

### 第四次对话新增内容（2026-06-10）
- **FTT+KD (3A only) 消融**：`code/05_train_distillation_ablation.py` + 训练结果
- **置换重要性分析**：`code/08_interpret_deepfm.py` → 6 模型全量对比图（基线 + 蒸馏变体）
- **论文审稿修改**：4 项 Critical/High + 蒸馏因果叙事重写 + 可解释性补全
- **引用合规检查**：18 条引用 0 缺失，bib 类型修正
- 论文编译：**27 页**，0 错误

### 论文状态
- 27 页，0 错误，含 FTT+KD 消融 + 全模型（6模型）置换重要性分析
- TabNet 已从论文、图表、引用全部清除
- 审稿修改已全部执行（统计语言 / 因果限定 / Sparsemax归因 / 可解释性范围）

### 关键理解点（给新对话的 Claude）
1. **Sparsemax 已替换 Softmax** 且效果已验证——AUPRC +4.4pp
2. **种子卫生**：`reset_seed(SEED)` 在每个模型训练前独立调用
3. **蒸馏因果链**：KD 本身跨架构有效 (FTT +1.68pp, DeepFM +1.59pp)；TreeReg 是退步唯一元凶 (-3.73pp)
4. **FTT α=0.25**（Sparsemax 需均衡梯度），DeepFM α=0.995（类别反比强加权）
5. **论文不含 TabNet**，已添加 FTT+KD 消融行 + 全模型（6模型）置换重要性分析
6. **答辩笔记**：`notes/12_答辩准备_核心知识体系.md` 包含完整知识链和 Q&A
7. **可解释性三阶梯**：注意力(看了什么) → 置换重要性(用了什么) → SHAP(每个特征贡献多少)

---

## 十四、工具链与 Skills 总览

### 14.1 开发环境

| 类别 | 工具 | 版本/说明 |
|------|------|-----------|
| **OS** | Windows 11 Home China | 10.0.26200 |
| **Python** | Python 3.13 | `D:\Python project\Python3.13\` |
| **DL 框架** | PyTorch 2.11.0+cu128 | CUDA 13.1, RTX 4060 8GB |
| **GPU** | NVIDIA GeForce RTX 4060 Laptop | 8GB VRAM, CUDA 13.1 |
| **包管理** | pip | 阿里云镜像 (`mirrors.aliyun.com`) |
| **Shell** | Git Bash | Unix 语法（路径用 `/d/...`，`/dev/null`） |

### 14.2 Python 关键依赖

| 库 | 用途 |
|----|------|
| `torch` (CUDA) | 神经网络训练（FT-Transformer, DeepFM） |
| `pytorch-tabnet` | TabNet 官方库（早期实验用，论文已移除） |
| `xgboost` | 基线模型 + 蒸馏教师 |
| `scikit-learn` | 预处理、评估指标、SMOTE |
| `numpy`, `pandas` | 数据处理 |
| `matplotlib`, `seaborn` | 可视化（PR/ROC/CM/Cost/Calibration/SHAP） |
| `shap` | TreeExplainer + Beeswarm + Waterfall |

### 14.3 LaTeX 工具链

| 工具 | 路径 | 用途 |
|------|------|------|
| **XeLaTeX** | `D:\MiKTeX\texmfs\install\miktex\bin\x64\xelatex.exe` | 编译 .tex → PDF |
| **Biber** | `D:\MiKTeX\texmfs\install\miktex\bin\x64\biber.exe` | 参考文献排序 |
| **MiKTeX Portable** | `D:\MiKTeX\` | Portable 版，无系统字体 |
| **文档类** | `ctexart` | 中文 LaTeX 文档类 |
| **字体方案** | `fontset=fandol` | MiKTeX Portable 必须用 Fandol（无 Windows 系统字体） |

### 14.4 LaTeX 关键宏包

| 宏包 | 用途 |
|------|------|
| `geometry` | 页边距 2.0cm |
| `graphicx` | 插图（`\includegraphics`） |
| `booktabs` | 三线表（`\toprule/\midrule/\bottomrule`） |
| `tikz` + `positioning/arrows.meta/shapes` | 模型架构流程图 |
| `amsmath/amssymb/bm` | 数学公式、粗体向量 |
| `hyperref` | PDF 超链接 |
| `biblatex` (backend=biber, style=ieee) | IEEE 格式参考文献 |
| `algorithm2e` | 伪代码（预留） |
| `float/subcaption/caption` | 图表浮动与子图 |
| `setspace` (stretch=1.05) | 行距压缩 |
| `makecell/array` | 表格单元格 |

### 14.5 Claude Code Skills 调用记录

本项目中调用的 Skills 及其贡献：

| Skill | 调用阶段 | 贡献 |
|-------|----------|------|
| **`academic-paper`** | 论文全周期 | 三步直出法：跳过中间产物，直接产出 .tex；摘要/引言/方法论/实验/分析/讨论/结论全部章节 |
| **`nature-reviewer`** | 论文质量把关 | 3 份模拟审稿报告 → 发现 6 类问题（图表、文字量、校准分析等）→ 逐项修复 |
| **`nature-polishing`** | 论文润色 | 中文学术表达优化；LaTeX 排版修复（Float 位置、页边距、行距） |
| **`nature-figure`** | 图表生成 | PR 曲线、ROC 曲线、混淆矩阵、成本对比、校准曲线、SHAP 图 |
| **`nature-citation`** | 文献支撑 | 为正文关键声明查找 Nature/CNS 引用支撑 |
| **`nature-academic-search`** | 文献检索 | 树监督学习相关文献（DeepTLF, Tree-Regularized, KD）的系统检索 |
| **`code-reviewer`** | 代码审计 | Sparsemax 实现、蒸馏模块、训练脚本的 bug 检测和安全审查 |
| **`latex-workshop-config`** | 编译配置 | 确保 `fontset=fandol`、XeLaTeX recipe、VS Code 编译工具链 |
| **`chart-generator`** | 数据可视化 | EDA 图表（类别平衡、金额分布） |
| **`xlsx`** | 数据表 | 指标表导出和格式化 |

### 14.6 代码架构模式

| 模式 | 位置 | 说明 |
|------|------|------|
| **Config 单例** | `code/config.py` | 所有路径/超参/种子集中管理 |
| **关注点分离** | `code/models/` vs `code/training/` | 模型架构与训练逻辑解耦 |
| **统一 Trainer** | `code/training/trainer.py` | 模型无关的训练循环（fit/validate/predict/早停/模型保存） |
| **seed hygiene** | `05_train_neural.py::reset_seed()` | 每模型前独立重置 random/numpy/torch/cuda 状态 |
| **KDTrainer 继承** | `05_train_distillation.py` | 继承 Trainer，重写 train_epoch/validate/predict 处理 3-tuple |
| **Sparsemax 闭式解** | `ft_transformer.py::sparsemax()` | O(d log d) 欧氏投影到概率单纯形 |

---

## 十五、项目开展流程（按时间线）

### Phase 1：环境搭建与数据管道（2026-05-16 ~ 05-18）

```
数据下载 → EDA → 预处理 → XGBoost/LR/RF 基线
```

- 手动下载 Kaggle CSV（kagglehub API 不稳定）
- 人工制造 0.5% 欺诈率（TARGET_FRAUD_RATIO=0.005, MAX_LEGIT_SAMPLES=150,000）
- IQR 异常值截断 + StandardScaler → `train.pt/val.pt/test.pt`
- XGBoost baseline: AUPRC=0.9331
- 产出：notes 00-02, EDA 图表

### Phase 2：神经网络实现与训练（2026-05-18 ~ 05-25）

```
MLP → TabNet → FT-Transformer → DeepFM → 统一训练管线
```

- 手写 FT-Transformer、DeepFM（rtdl/deepctr-torch 不适配）
- TabNet 使用 pytorch-tabnet 官方库
- 统一 Trainer + FocalLoss + metrics 模块
- 消融实验：Focal Loss / Class Weight / SMOTE / Plain CE
- 产出：notes 03-05, 09-11

### Phase 3：论文写作与审稿（2026-05-25 ~ 05-30）

```
论文框架 → 图表 → 正文 → Nature 审稿 → 修改
```

- 7 章节论文框架（中英双语摘要）
- 评估 + SHAP 图表全部生成
- Nature-reviewer 3 份审稿报告 + 交叉综合
- 6 项一致性修复（参数量、排名、TabNet 描述、消融结果等）
- SAINT 格式学习：caption 精简、[H]→[htbp]、行距 1.05
- 编译通过（19 页，零 Warning）
- 产出：notes 06-08

### Phase 4：GPU 迁移与 Sparsemax（2026-06-06 ~ 06-07）

```
CUDA torch 安装 → Sparsemax 实现 → FTT 修复
```

- pip CUDA 安装踩坑：阿里云镜像无 CUDA build → 官方 `--index-url` 安装
- Sparsemax 原理学习（Martins & Astudillo 2016）→ 闭式解实现
- debug：tau 阈值过大 → 修正为 `(support·z_sorted - 1)/k_z`
- 初步实验：FTT Sparsemax AUPRC 仅 0.8401（种子未独立，过拟合）

### Phase 5：种子修复 + 蒸馏实现（2026-06-07）

```
种子可复现性诊断 → reset_seed → 蒸馏代码 → 学习笔记
```

- 诊断：pytorch-tabnet `.fit()` 消耗 numpy 随机状态 → 影响后续模型初始化
- 修复：每模型前 `reset_seed(SEED)` → random/numpy/torch/cuda 全重置
- `code/training/distillation.py`：DistillationLoss + TreeFeatureRegularization
- `code/05_train_distillation.py`：KDTrainer 继承 + 3-tuple DataLoader
- code-reviewer 审计：1 Critical + 2 Medium + 4 Low → 全部修复
- 产出：notes/09_树监督学习与归纳偏置迁移

### Phase 6：完成训练 + TabNet 移除（2026-06-08 第三次对话）

```
基线重训 → 蒸馏训练 → TabNet 清除 → 论文重构 → 笔记优化 → 答辩准备
```

1. **基线重训**（种子修复后）：FTT 0.9179 (+4.4pp), DeepFM 0.9336 (超越 XGBoost)
2. **蒸馏训练**：DeepFM+KD 0.9495🥇, FTT+KD+TreeReg 0.8974▼
3. **TabNet 移除**：8 个 .tex + references.bib + 全部图表重生成
4. **模型深度讲解**：model_ftt.tex (~3.5pp) + model_deepfm.tex (~3pp)，各含 tikz 流程图
5. **论文数字全更新**：摘要/实验/分析/讨论/结论全部同步最新 AUPRC/ECE/Cost
6. **笔记重构**：删 3 旧（TabNet/Attention/Illustrated），更 4，新建答辩笔记
7. **图表重生成**：06_evaluate.py 去 TabNet → PR/ROC/CM/Cost/Calibration 全部更新
8. **编译**：26 页，0 错误，0 未定义引用

### Phase 4.5：论文审稿与消融补全（2026-06-10 第四次对话）

```
academic-paper-reviewer → Critical/High修复 → FTT+KD消融 → 置换重要性 → 引用检查 → 笔记更新
```

1. **5 人审稿**：EIC + 方法论 + 领域 + 跨学科 + 魔鬼代言人 → 4 Critical + 4 High
2. **审稿修改执行**：统计语言修正 / 消融缺失标注 / Sparsemax归因限定 / SHAP→三模型可解释性
3. **FTT+KD (3A only) 消融训练**：AUPRC=0.9347 (+1.68pp), ECE=0.0007, Cost=97 — 颠覆性发现：KD 跨架构有效，TreeReg 是唯一元凶
4. **置换重要性全模型对比**：从 3 基线扩展至 6 模型（含 DFM+KD, FTT+KD, FTT+KD+TreeReg）→ KD 拉近排序 (DeepFM +0.220) / TreeReg 制造复刻 (r=0.964) 但 V14 集中度 43.9% 致性能崩塌 / KD 提升跨架构行为相似性 (r 0.639→0.806)
5. **论文因果叙事重写**：蒸馏结论从"KD 架构依赖性"→"TreeReg 与注意力结构性拮抗"
6. **引用合规检查**：ars-citation-check → 18 条 0 缺失 → 修正 2 条 arXiv preprint 类型标记

### 时间统计

| 对话 | 日期 | 核心产出 |
|------|------|----------|
| 第 1 次 | 05-16 ~ 05-30 | 环境、数据管道、模型实现、论文初稿、审稿修改 |
| 第 2 次 | 06-06 ~ 06-07 | GPU 迁移、Sparsemax、种子修复、蒸馏代码、笔记 |
| 第 3 次 | 06-08 | 完成训练、TabNet 移除、论文重构、答辩准备、收尾 |
| 第 4 次 | 06-10 | 审稿→消融→置换重要性(6模型)→因果重写→引用检查 |

### 经验教训

1. **pip CUDA 安装**：阿里云镜像只有 CPU build → 用 `--index-url https://download.pytorch.org/whl/cu128`
2. **种子可复现性**：第三方库（pytorch-tabnet）的 `.fit()` 消耗随机状态 → 必须在每个新模型前独立重置
3. **MiKTeX Portable 字体**：无系统字体必须 `fontset=fandol`，编译前需检查
4. **模型架构选择**：不要枚举架构 → 选有对比价值的两种归纳偏置进行深度对比
5. **消融实验是科学结论的基础**：FTT+KD 消融将论文核心发现从模糊的"蒸馏是架构依赖的"精确为"TreeReg 与注意力拮抗"——一个消融实验改变了整篇论文的因果叙事
6. **负结果是学术贡献**：FTT+KD+TreeReg 的失败比 DeepFM+KD 的成功更有诊断价值——诚实报告负结果并深挖原因是课程作业的加分项
7. **论文驱动开发**：先定论文叙事，再定实验设计，最后写代码——避免做了实验却用不上
8. **脚本输出≠论文输入**：`06_evaluate.py` 的 `FIGURE_DIR` 指向 `outputs/figures/`，而论文 `\graphicspath{{figures/}}` 指向 `paper/figures/`。两个目录同名不同位。每次重跑脚本后必须 `cp outputs/figures/*.pdf paper/figures/ && cp outputs/results/model_comparison.tex paper/figures/`，否则论文编译时看到的仍是旧图。验证方式：`ls -la paper/figures/*.pdf` 的时间戳应在脚本运行之后。
