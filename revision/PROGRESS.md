# 路径A 修订进度 — v5（2026-06-15 深夜·第二轮审稿修改完成）

## 当前状态

**PDF**: `revision/paper/main.pdf` — **19页，0错误，0未定义引用，0 overfull hbox**

## 第二轮审稿修改（v5）

### P0 — 数字一致性修复 ✅
| 位置 | 旧值 | 新值 |
|------|------|------|
| 英文摘要 | AUPRC 0.8738→0.9179 (+4.4pp) | **0.9126→0.9384 (+2.6pp)** |
| 引言 | 同旧数字 + ECE 0.2328→0.0120 | 一致化为新数字 + ×3.3放大+×7.5恢复 |
| 背景.tex | 参数量153,921; GELU | **203,841; ReGLU** |
| 设置.tex | "使用官方rtdl库"; ~154K | **"手写实现"; 203,841** |
| 诊断.tex | 嵌入SVD图=置换重要性图(label错配) | label改为`fig:permutation_importance` |
| references.bib | 5组重复条目 | 已去重 |

### P1 — 过度声称软化 ✅
| 位置 | 旧表述 | 新表述 |
|------|--------|--------|
| 诊断 L76 | "可操作化的三阶段诊断框架" | "描述性的三阶段诊断模板" |
| 讨论 L18 | "标准化也是语义抹平的来源" | 加限定词：仅一个特征、差异微小、提示假设 |
| 结论 L22 | "BCE取得了最高AUPRC" | 加"在本实验条件下"、"单种子单次实验"限定 |

### P2 — 格式修复 ✅
| 问题 | 处理 |
|------|------|
| results.tex L8-9: overfull 19.4pt | 删"FTT+Softmax与FTT+Sparsemax"缩短句 |
| results.tex L47-48: overfull 11.1pt | 简化JS散度公式 |
| conclusion.tex L8-9: overfull 49.5pt | 删括号内三个变体名称 |
| "No \author given" | 添加`\author{}` |

## 剩余cosmetic warnings（无需修复）
- Underfull hbox ×2: 讨论章表格caption换行（p{}列宽强制断行导致，可接受）
- hyperref Unicode: 中文PDF书签（预期行为）

## 训练结果（统一架构：手写ReGLU, 203,841参数）

| 模型 | AUPRC | AttnStd | Std/Uniform |
|------|-------|---------|-------------|
| FTT+Softmax+FocalLoss | 0.9126 | 0.00125 | 0.036 |
| FTT+Sparsemax+FocalLoss | 0.9384 | 0.00942 | 0.273 |
| FTT+Softmax+BCE | 0.9491 | 0.00408 | 0.118 |

## 三条因果链

1. **PCA根因**: BCE下塌缩仍存在(std仅11.8%基线) → PCA嵌入同构是根因
2. **Focal Loss放大器**: Focal Loss将塌缩放大了×3.3 (std 0.00408→0.00125)
3. **Sparsemax修复**: 注意力多样性×7.5, AUPRC +2.6pp, 但修复症状非病因
4. **不平衡-注意力权衡**: BCE AUPRC最高(0.9491), 揭示Focal Loss注意力压缩副作用
