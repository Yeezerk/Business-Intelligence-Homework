"""
训练传统 ML 基线：Logistic Regression → Random Forest → XGBoost。

XGBoost 是本项目的教师模型（方案 3A 的蒸馏源）：
  - 300 棵树, max_depth=8, lr=0.05
  - scale_pos_weight = 负样本/正样本 ≈ 199（自动计算）
  - 训练完成后保存为 .json → 蒸馏脚本直接 load 做教师

LR 和 RF 作为传统基线对照，最终论文仅 XGBoost 进入同台对比。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from config import PROCESSED_DIR, MODEL_DIR, RESULT_DIR, SEED
from training.metrics import find_best_threshold

np.random.seed(SEED)


def load_processed():
    """加载 .pt 格式的预处理数据，转为 numpy。"""
    train_X, train_y = torch.load(os.path.join(PROCESSED_DIR, "train.pt"), weights_only=True)
    val_X, val_y = torch.load(os.path.join(PROCESSED_DIR, "val.pt"), weights_only=True)
    test_X, test_y = torch.load(os.path.join(PROCESSED_DIR, "test.pt"), weights_only=True)
    return (train_X.numpy(), train_y.numpy(), val_X.numpy(), val_y.numpy(), test_X.numpy(), test_y.numpy())


def evaluate(name, y_true, y_scores, y_val=None, val_scores=None):
    """
    评估并打印模型性能。
    注意：用验证集 F1 最优阈值来决定测试集上的硬分类结果（不直接在测试集上调参）。
    """
    auprc = average_precision_score(y_true, y_scores)
    roc = roc_auc_score(y_true, y_scores)
    # 从验证集找最优阈值（与 NN 模型一致的做法）
    if y_val is not None and val_scores is not None:
        best_thr, _ = find_best_threshold(y_val, val_scores)
    else:
        best_thr = 0.5
    y_pred = (y_scores >= best_thr).astype(int)
    f1 = f1_score(y_true, y_pred)
    print(f"  {name:>25s} | AUPRC={auprc:.4f} | ROC-AUC={roc:.4f} | F1={f1:.4f} | thr={best_thr:.3f}")
    return {"Model": name, "AUPRC": auprc, "ROC_AUC": roc, "F1": f1, "Threshold": best_thr}


def main():
    print("Loading processed data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_processed()

    # 传统 ML 模型不需要单独的验证集，合并 train+val 一起训练
    X_train_full = np.concatenate([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    print(f"Train: {X_train_full.shape[0]:,} samples, fraud={y_train_full.mean():.4%}")
    print(f"Test:  {X_test.shape[0]:,} samples, fraud={y_test.mean():.4%}\n")

    results = []

    # ── Logistic Regression（线性基线，class_weight="balanced" 自动加权）──
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    lr.fit(X_train_full, y_train_full)
    lr_scores = lr.predict_proba(X_test)[:, 1]            # 取正类概率
    lr_val_scores = lr.predict_proba(X_val)[:, 1]
    results.append(evaluate("Logistic Regression", y_test, lr_scores, y_val, lr_val_scores))
    with open(os.path.join(MODEL_DIR, "lr_baseline.pkl"), "wb") as f:
        pickle.dump(lr, f)

    # ── Random Forest（树集成基线，200棵树，max_depth=12，class_weight="balanced"）──
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        random_state=SEED, n_jobs=-1,
    )
    rf.fit(X_train_full, y_train_full)
    rf_scores = rf.predict_proba(X_test)[:, 1]
    rf_val_scores = rf.predict_proba(X_val)[:, 1]
    results.append(evaluate("Random Forest", y_test, rf_scores, y_val, rf_val_scores))
    with open(os.path.join(MODEL_DIR, "rf_baseline.pkl"), "wb") as f:
        pickle.dump(rf, f)

    # ── XGBoost（树基线 + 蒸馏教师，scale_pos_weight 自动处理不平衡）──
    print("Training XGBoost...")
    scale_pos_weight = (len(y_train_full) - y_train_full.sum()) / y_train_full.sum()  # ≈ 199
    xgb = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,  # 正类梯度放大 ~199 倍
        random_state=SEED, eval_metric="aucpr", verbosity=0,
    )
    xgb.fit(X_train_full, y_train_full)
    xgb_scores = xgb.predict_proba(X_test)[:, 1]
    xgb_val_scores = xgb.predict_proba(X_val)[:, 1]
    results.append(evaluate("XGBoost", y_test, xgb_scores, y_val, xgb_val_scores))
    xgb.save_model(os.path.join(MODEL_DIR, "xgb_baseline.json"))  # 保存为 .json → 蒸馏脚本加载

    # ── 保存结果表格 ──
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULT_DIR, "baseline_metrics.csv"), index=False)
    print(f"\nBaseline results saved to {RESULT_DIR}/baseline_metrics.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
