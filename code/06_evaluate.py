"""
评估全流程：加载所有训练好的模型 → 生成 PR/ROC/混淆矩阵/成本/校准 → 输出 LaTeX 表。

答辩要点：
  - PR 曲线是论文主图——在不平衡 0.5% 下区分度远好于 ROC
  - 校准曲线 + ECE 揭示 KD 改善概率校准（DeepFM 0.0394→0.0008）
  - 成本分析展示业务视角（FP×1 + FN×10）
  - LaTeX 表自动生成，可直接嵌入论文
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, roc_curve, confusion_matrix,
    average_precision_score, roc_auc_score,
)
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier

from config import PROCESSED_DIR, MODEL_DIR, FIGURE_DIR, RESULT_DIR, DEVICE, SEED
from training.metrics import find_best_threshold, compute_ece
from models.ft_transformer import FTTransformer
from models.deepfm import DeepFM

# matplotlib 全局设置
plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "legend.fontsize": 9, "figure.dpi": 150,
})


def load_test_data():
    test_X, test_y = torch.load(os.path.join(PROCESSED_DIR, "test.pt"), weights_only=True)
    return test_X.numpy(), test_y.numpy()


def load_val_data():
    val_X, val_y = torch.load(os.path.join(PROCESSED_DIR, "val.pt"), weights_only=True)
    return val_X.numpy(), val_y.numpy()


def load_neural_model(name, input_dim, weight_file=None, **kwargs):
    """加载训练好的手搓 PyTorch 模型。

    name: 架构名（"FT-Transformer" 或 "DeepFM"）
    weight_file: 权重文件名（不含 _best.pt），默认与架构名对应
    kwargs: 传给 FTTransformer 的构造参数（d_model, n_heads, n_layers 等）
    """
    if name == "FT-Transformer":
        model = FTTransformer(input_dim, **kwargs) if kwargs else FTTransformer(input_dim)
    else:
        model = DeepFM(input_dim)
    file_name = weight_file if weight_file else ("FT-Transformer" if name == "FT-Transformer" else "deepfm")
    path = os.path.join(MODEL_DIR, f"{file_name}_best.pt")
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    return model


def get_all_predictions(X_test, y_test, X_val, y_val, input_dim):
    """
    获取所有模型的预测结果，每个模型使用验证集优化的阈值。
    TabNet 已排除——保留代码仅为复现完整性。
    """
    preds = {"y_true": y_test}
    thresholds = {}

    # ── FT-Transformer ──
    ftt = load_neural_model("FT-Transformer", input_dim)
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    Xv_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        ftt_scores = torch.sigmoid(ftt(X_tensor)).squeeze(-1).cpu().numpy()
        ftt_val = torch.sigmoid(ftt(Xv_tensor)).squeeze(-1).cpu().numpy()
    preds["FT-Transformer"] = ftt_scores
    best_thr, _ = find_best_threshold(y_val, ftt_val)
    thresholds["FT-Transformer"] = best_thr

    # ── DeepFM ──
    deepfm = load_neural_model("DeepFM", input_dim)
    with torch.no_grad():
        dfm_scores = torch.sigmoid(deepfm(X_tensor)).squeeze(-1).cpu().numpy()
        dfm_val = torch.sigmoid(deepfm(Xv_tensor)).squeeze(-1).cpu().numpy()
    preds["DeepFM"] = dfm_scores
    best_thr, _ = find_best_threshold(y_val, dfm_val)
    thresholds["DeepFM"] = best_thr

    # ── XGBoost（教师基线）──
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(MODEL_DIR, "xgb_baseline.json"))
    xgb_scores = xgb_model.predict_proba(X_test)[:, 1]
    xgb_val = xgb_model.predict_proba(X_val)[:, 1]
    preds["XGBoost"] = xgb_scores
    best_thr, _ = find_best_threshold(y_val, xgb_val)
    thresholds["XGBoost"] = best_thr

    # ── DeepFM + KD ──
    deepfm_kd = load_neural_model("DeepFM", input_dim, weight_file="deepfm_kd")
    with torch.no_grad():
        dfm_kd_scores = torch.sigmoid(deepfm_kd(X_tensor)).squeeze(-1).cpu().numpy()
        dfm_kd_val = torch.sigmoid(deepfm_kd(Xv_tensor)).squeeze(-1).cpu().numpy()
    preds["DeepFM+KD"] = dfm_kd_scores
    best_thr, _ = find_best_threshold(y_val, dfm_kd_val)
    thresholds["DeepFM+KD"] = best_thr

    # ── FTT + KD ──
    ftt_kd = load_neural_model("FT-Transformer", input_dim, weight_file="ftt_kd_only",
                               d_model=48, n_heads=4, n_layers=2)
    with torch.no_grad():
        ftt_kd_scores = torch.sigmoid(ftt_kd(X_tensor)).squeeze(-1).cpu().numpy()
        ftt_kd_val = torch.sigmoid(ftt_kd(Xv_tensor)).squeeze(-1).cpu().numpy()
    preds["FTT+KD"] = ftt_kd_scores
    best_thr, _ = find_best_threshold(y_val, ftt_kd_val)
    thresholds["FTT+KD"] = best_thr

    # ── FTT + KD + TreeReg ──
    ftt_kd_tree = load_neural_model("FT-Transformer", input_dim, weight_file="ftt_kd_treereg",
                                    d_model=48, n_heads=4, n_layers=2)
    with torch.no_grad():
        ftt_kd_tree_scores = torch.sigmoid(ftt_kd_tree(X_tensor)).squeeze(-1).cpu().numpy()
        ftt_kd_tree_val = torch.sigmoid(ftt_kd_tree(Xv_tensor)).squeeze(-1).cpu().numpy()
    preds["FTT+KD+TreeReg"] = ftt_kd_tree_scores
    best_thr, _ = find_best_threshold(y_val, ftt_kd_tree_val)
    thresholds["FTT+KD+TreeReg"] = best_thr

    return preds, thresholds


def plot_pr_curves(preds):
    """
    PR 曲线——论文主图。
    在不平衡数据中，PR 曲线的区分度远好于 ROC 曲线。
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(preds) - 1))
    model_names = [k for k in preds if k != "y_true"]
    for i, name in enumerate(model_names):
        precision, recall, _ = precision_recall_curve(preds["y_true"], preds[name])
        auprc = average_precision_score(preds["y_true"], preds[name])
        ax.plot(recall, precision, color=colors[i], lw=1.8, label=f"{name} (AUPRC={auprc:.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "pr_curves.pdf"), dpi=200)
    plt.close(fig)


def plot_roc_curves(preds):
    """
    ROC 曲线——辅助参考。
    注意：ROC-AUC 在不平衡数据中被 TN（99.5%）主导，区分度差。
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(preds) - 1))
    model_names = [k for k in preds if k != "y_true"]
    for i, name in enumerate(model_names):
        fpr, tpr, _ = roc_curve(preds["y_true"], preds[name])
        auc = roc_auc_score(preds["y_true"], preds[name])
        ax.plot(fpr, tpr, color=colors[i], lw=1.8, label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)  # 随机基线
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "roc_curves.pdf"), dpi=200)
    plt.close(fig)


def plot_confusion_matrices(preds, thresholds):
    """混淆矩阵：展示 Top-4 模型的 TP/FP/FN/TN。"""
    model_names = [k for k in preds if k != "y_true"]
    scores = {n: average_precision_score(preds["y_true"], preds[n]) for n in model_names}
    top4 = sorted(scores, key=scores.get, reverse=True)[:4]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, name in zip(axes.flat, top4):
        thr = thresholds.get(name, 0.5)
        y_pred = (preds[name] >= thr).astype(int)
        cm = confusion_matrix(preds["y_true"], y_pred)
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_title(f"{name}  (thr={thr:.3f})", fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Legit", "Fraud"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Legit", "Fraud"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm[i, j] > cm.max()/2 else "black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.suptitle("Confusion Matrices (4 Models)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "confusion_matrices.pdf"), dpi=200)
    plt.close(fig)


def plot_cost_comparison(preds, thresholds):
    """
    业务成本柱状图：Cost = FP×1 + FN×10。
    展示最"省钱"的模型——不只看精度，还要看实际业务成本。
    """
    model_names = [k for k in preds if k != "y_true"]
    costs = {}
    for name in model_names:
        thr = thresholds.get(name, 0.5)
        y_pred = (preds[name] >= thr).astype(int)
        cm = confusion_matrix(preds["y_true"], y_pred)
        tn, fp, fn, tp = cm.ravel()
        costs[name] = fp * 1 + fn * 10
    fig, ax = plt.subplots(figsize=(7, 4))
    sorted_names = sorted(costs, key=costs.get)
    values = [costs[n] for n in sorted_names]
    bars = ax.barh(sorted_names, values,
                   color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(values))))
    ax.set_xlabel("Total Cost (FP×1 + FN×10)")
    ax.set_title("Business Cost Comparison")
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                str(v), va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "cost_comparison.pdf"), dpi=200)
    plt.close(fig)


def generate_latex_table(preds, thresholds):
    """
    自动生成 LaTeX 对比表，可直接复制到论文 .tex 文件中。
    按 AUPRC 降序排列，包含 ECE（校准误差）列。
    """
    from training.metrics import compute_all_metrics
    model_names = [k for k in preds if k != "y_true"]
    rows = []
    for name in model_names:
        thr = thresholds.get(name, 0.5)
        metrics = compute_all_metrics(preds["y_true"], preds[name], threshold=thr)
        ece, _ = compute_ece(preds["y_true"], preds[name])
        rows.append({
            "Model": name, "AUPRC": metrics["AUPRC"],
            "ROC-AUC": metrics["ROC_AUC"], "F1": metrics["F1"],
            "Precision": metrics["Precision"], "Recall": metrics["Recall"],
            "Cost": metrics["Cost"], "ECE": ece,
        })
    df = pd.DataFrame(rows).sort_values("AUPRC", ascending=False)
    df.to_csv(os.path.join(RESULT_DIR, "all_metrics.csv"), index=False)

    # 构建 LaTeX 表格字符串
    latex = r"""\begin{table}[htbp]
\centering
\caption{Model Performance Comparison (sorted by AUPRC)}
\label{tab:model_comparison}
\small
\begin{tabular}{lccccccc}
\toprule
\textbf{Model} & \textbf{AUPRC} & \textbf{ROC-AUC} & \textbf{F1} & \textbf{Precision} & \textbf{Recall} & \textbf{Cost} & \textbf{ECE} \\
\midrule
"""
    for _, row in df.iterrows():
        latex += (f"{row['Model']} & {row['AUPRC']:.4f} & {row['ROC-AUC']:.4f} & "
                  f"{row['F1']:.4f} & {row['Precision']:.4f} & {row['Recall']:.4f} & "
                  f"{int(row['Cost']):,} & {row['ECE']:.4f} \\\\\n")
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(os.path.join(RESULT_DIR, "model_comparison.tex"), "w") as f:
        f.write(latex)
    print(f"LaTeX table saved to {RESULT_DIR}/model_comparison.tex")


def plot_calibration(preds):
    """
    校准曲线（可靠性图）：每个模型一张子图。
    对角线 = 完美校准（模型说 90% 时正好 90% 正确）。
    条带偏离对角线越远 → ECE 越大 → 校准越差。
    KD 显著改善校准：对比 DeepFM（ECE=0.0394）vs DeepFM+KD（ECE=0.0008）。
    """
    model_names = [k for k in preds if k != "y_true"]
    n_models = len(model_names)
    n_cols = 2
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    ece_results = {}
    for idx, name in enumerate(model_names):
        ax = axes[idx // n_cols, idx % n_cols]
        ece, bin_info = compute_ece(preds["y_true"], preds[name])
        ece_results[name] = ece
        bin_centers = [(b["bin_lower"] + b["bin_upper"]) / 2 for b in bin_info]
        accuracies = [b["accuracy"] for b in bin_info]
        ax.bar(bin_centers, accuracies, width=0.08, alpha=0.6, color="#3498db",
               label="Actual fraction", edgecolor="white")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"{name}\nECE={ece:.4f}", fontsize=10)
        ax.legend(fontsize=7, loc="upper left"); ax.grid(True, alpha=0.2)
    # 隐藏多余的子图
    for idx in range(n_models, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    fig.suptitle("Calibration Curves (Reliability Diagrams)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "calibration_curves.pdf"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\nECE (Expected Calibration Error) per model:")
    print("-" * 40)
    for name, ece in sorted(ece_results.items(), key=lambda x: x[1]):
        print(f"  {name:>25s}: ECE={ece:.4f}")
    return ece_results


def main():
    print("Loading test data...")
    X_test, y_test = load_test_data()
    X_val, y_val = load_val_data()
    input_dim = X_test.shape[1]
    print(f"Test set: {X_test.shape[0]:,} samples, {input_dim} features\n")

    print("Collecting predictions from all models...")
    preds, thresholds = get_all_predictions(X_test, y_test, X_val, y_val, input_dim)
    print(f"Optimal thresholds (from val):")
    for k, v in thresholds.items():
        print(f"  {k}: {v:.3f}")

    print("Generating PR curves..."); plot_pr_curves(preds)
    print("Generating ROC curves..."); plot_roc_curves(preds)
    print("Generating confusion matrices..."); plot_confusion_matrices(preds, thresholds)
    print("Generating cost comparison..."); plot_cost_comparison(preds, thresholds)
    print("Generating calibration analysis..."); plot_calibration(preds)
    print("Generating LaTeX comparison table..."); generate_latex_table(preds, thresholds)
    print(f"\nAll evaluation outputs saved to {FIGURE_DIR}/ and {RESULT_DIR}/")


if __name__ == "__main__":
    main()
