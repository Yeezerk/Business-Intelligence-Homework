"""
评估指标集合——本项目核心方法论之一。

主指标选型逻辑：
  - AUPRC（主指标）：聚焦正类（欺诈），区分度是 ROC-AUC 的 10 倍
    → 欺诈仅 0.5%，ROC-AUC 被 TN（真负类，占 99.5%）主导
    → 三个模型 ROC-AUC 极差仅 0.0054，而 AUPRC 极差 ~0.05
  - 业务成本 Cost = FP×1 + FN×10：FN（漏报）权重 10 倍于 FP（误报）
    → 漏一笔欺诈的损失远大于误拦一笔正常交易
  - ECE：衡量概率校准质量——模型说 90% 时是否真的 90% 正确
"""
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
)


def compute_all_metrics(y_true, y_scores, threshold: float = 0.5):
    """
    计算全套评估指标。
    y_true: 真实标签 (N,)
    y_scores: 模型输出的概率 (N,)
    threshold: 硬分类的决策阈值（从验证集 F1 最大化得到）
    """
    y_pred = (y_scores >= threshold).astype(int)

    # ── 排序指标（不依赖阈值）──
    auprc = average_precision_score(y_true, y_scores)  # 主指标：Precision-Recall 曲线下面积
    roc_auc = roc_auc_score(y_true, y_scores)          # 辅助指标：ROC 曲线下面积

    # ── 阈值依赖指标 ──
    precision = precision_score(y_true, y_pred, zero_division=0)  # TP / (TP + FP)
    recall = recall_score(y_true, y_pred, zero_division=0)        # TP / (TP + FN)
    f1 = f1_score(y_true, y_pred, zero_division=0)                # 调和平均

    # ── 混淆矩阵 + 业务成本 ──
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # FP（误报）= 把正常交易判为欺诈 → 增加审核成本
    # FN（漏报）= 没检出欺诈 → 直接经济损失
    cost_fp = 1
    cost_fn = 10
    total_cost = fp * cost_fp + fn * cost_fn

    return {
        "AUPRC": round(auprc, 4),
        "ROC_AUC": round(roc_auc, 4),
        "F1": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Cost": int(total_cost),
    }


def find_best_threshold(y_true, y_scores):
    """
    在验证集上通过 F1 最大化找到最优决策阈值。
    注意：绝不在测试集上调阈值！否则会数据泄露。
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # F1 = 2 * P * R / (P + R)，对每个可能的阈值计算 F1
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    return best_threshold, f1_scores[best_idx]


def compute_ece(y_true, y_scores, n_bins: int = 10):
    """
    计算 Expected Calibration Error（期望校准误差）。
    原理：把 [0,1] 分成 10 个等宽箱子，每个箱子内比较
    - 平均预测概率（confidence）
    - 实际正类比例（accuracy）
    ECE = Σ (|箱子大小/总数| × |accuracy - confidence|)

    ECE 越小越好：0 表示完美校准（模型说 80% 时正好 80% 正确）
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_info = []

    for i in range(n_bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        # 最后一个箱子包含右端点，前面不包含
        if i == n_bins - 1:
            mask = (y_scores >= lower) & (y_scores <= upper)
        else:
            mask = (y_scores >= lower) & (y_scores < upper)

        n_b = mask.sum()
        if n_b == 0:
            continue

        acc_b = y_true[mask].mean()   # 实际正类比例 = accuracy
        conf_b = y_scores[mask].mean()  # 平均预测概率 = confidence
        ece += (n_b / len(y_true)) * abs(acc_b - conf_b)

        bin_info.append({
            "bin_lower": round(lower, 2),
            "bin_upper": round(upper, 2),
            "count": int(n_b),
            "accuracy": round(float(acc_b), 4),
            "confidence": round(float(conf_b), 4),
            "gap": round(float(acc_b - conf_b), 4),
        })

    return round(ece, 4), bin_info
