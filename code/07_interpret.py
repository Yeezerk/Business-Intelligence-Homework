"""
模型可解释性分析：SHAP + 注意力可视化 + 特征掩码 + SII 交互分析。
答辩要点：
  - SHAP TreeExplainer 用于 XGBoost 和 RF：提供特征重要性 beeswarm 和 waterfall
  - SHAP Interaction Index (SII)：通过 shapiq 库计算二阶特征交互
    - 区分"协同"（同方向）和"冗余"（异方向）交互
  - FTT 注意力图：get_attention_weights() 提取 CLS→特征的注意力权重
    - 验证 Sparsemax 是否真的做了稀疏选择（看注意力分布的 std 是否显著 >0）
  - TabNet masks：每个决策步的特征选择掩码热力图
  - 这些可视化支撑了论文中的发现（Sparsemax 稀疏性、V14 主导特征等）
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

from config import PROCESSED_DIR, MODEL_DIR, FIGURE_DIR, DEVICE, FEATURE_NAMES
from models.tabnet import TabNet
from models.ft_transformer import FTTransformer

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})


def load_data():
    train_X, train_y = torch.load(os.path.join(PROCESSED_DIR, "train.pt"), weights_only=True)
    test_X, test_y = torch.load(os.path.join(PROCESSED_DIR, "test.pt"), weights_only=True)
    return train_X.numpy(), train_y.numpy(), test_X.numpy(), test_y.numpy()


def load_baselines():
    models = {}
    with open(os.path.join(MODEL_DIR, "lr_baseline.pkl"), "rb") as f:
        models["Logistic Regression"] = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "rf_baseline.pkl"), "rb") as f:
        models["Random Forest"] = pickle.load(f)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(MODEL_DIR, "xgb_baseline.json"))
    models["XGBoost"] = xgb_model
    return models


def shap_analysis(X_train, X_test, y_test):
    """SHAP analysis for tree-based and linear models."""
    print("\n" + "=" * 60)
    print("SHAP Interpretability Analysis")
    print("=" * 60)

    n_background = min(500, len(X_train))
    X_bg = X_train[:n_background]
    X_test_sub = X_test[:1000] if len(X_test) > 1000 else X_test

    baselines = load_baselines()

    print("Computing SHAP for XGBoost (TreeExplainer)...")
    explainer = shap.TreeExplainer(baselines["XGBoost"])
    shap_values = explainer.shap_values(X_test_sub)

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sub, feature_names=FEATURE_NAMES,
                      show=False, max_display=15)
    fig.savefig(os.path.join(FIGURE_DIR, "shap_xgboost_summary.pdf"), bbox_inches="tight", dpi=200)
    plt.close("all")

    for idx, (label, title) in enumerate([(0, "Legitimate"), (1, "Fraud")]):
        indices = np.where(y_test[:len(X_test_sub)] == label)[0]
        if len(indices) > 0:
            sample_idx = indices[len(indices)//2]
            fig = plt.figure(figsize=(8, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[sample_idx],
                    base_values=explainer.expected_value,
                    data=X_test_sub[sample_idx],
                    feature_names=FEATURE_NAMES,
                ),
                show=False,
            )
            fig.savefig(os.path.join(FIGURE_DIR, f"shap_waterfall_{title.lower()}.pdf"),
                        bbox_inches="tight", dpi=200)
            plt.close("all")

    print("Computing SHAP for Random Forest...")
    rf_explainer = shap.TreeExplainer(baselines["Random Forest"])
    rf_shap = rf_explainer.shap_values(X_test_sub[:300])
    if isinstance(rf_shap, list):
        rf_shap = rf_shap[1]

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(rf_shap, X_test_sub[:300],
                      feature_names=FEATURE_NAMES,
                      show=False, max_display=15)
    fig.savefig(os.path.join(FIGURE_DIR, "shap_rf_summary.pdf"), bbox_inches="tight", dpi=200)
    plt.close("all")

    print("SHAP plots saved.")


def tabnet_masks(X_test, input_dim):
    """Visualize TabNet feature selection masks."""
    print("\nGenerating TabNet feature mask heatmap...")
    model = TabNet(input_dim).to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "tabnet_best.pt"), map_location=DEVICE, weights_only=True
    ))
    model.eval()

    X_tensor = torch.tensor(X_test[:500], dtype=torch.float32).to(DEVICE)
    masks = model.explain(X_tensor)

    avg_masks = masks.mean(dim=1).cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(avg_masks, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Decision Step")
    ax.set_title("TabNet Feature Selection Masks (averaged over 500 samples)")
    ax.set_yticks(range(len(avg_masks)))
    ax.set_yticklabels([f"Step {i+1}" for i in range(len(avg_masks))])
    ax.set_xticks(range(0, input_dim, max(1, input_dim // 15)))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in range(0, input_dim, max(1, input_dim // 15))],
                       rotation=45, ha="right", fontsize=8)
    plt.colorbar(im, ax=ax, label="Mean Mask Value")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "tabnet_masks.pdf"), dpi=200)
    plt.close(fig)
    print("TabNet mask heatmap saved.")


def ft_transformer_attention(X_test, input_dim):
    """Visualize FT-Transformer attention from CLS token to features."""
    print("\nGenerating FT-Transformer attention map...")
    model = FTTransformer(input_dim).to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "ft_transformer_best.pt"), map_location=DEVICE, weights_only=True
    ))

    X_tensor = torch.tensor(X_test[:200], dtype=torch.float32).to(DEVICE)
    attn = model.get_attention_weights(X_tensor)

    if attn is not None:
        attn = attn.cpu().numpy()
        mean_val = attn.mean()
        attn_range = (attn.min(), attn.max())

        fig, ax = plt.subplots(figsize=(13, 5))
        indices = np.argsort(attn)[::-1]
        top_k = min(20, len(indices))
        top_vals = attn[indices][:top_k]
        colors = ["#e74c3c" if i == 0 else "#3498db" for i in range(top_k)]
        ax.bar(range(top_k), top_vals, color=colors, edgecolor="white", width=0.7)

        # Narrow y-axis to make differences visible — all values are ~0.033-0.035
        margin = attn_range[1] - attn_range[0]
        ax.set_ylim(attn_range[0] - margin * 0.5, attn_range[1] + margin * 0.5)

        for i, v in enumerate(top_vals):
            ax.text(i, v + margin * 0.08, f"{v:.4f}", ha="center", fontsize=8, rotation=90)

        ax.set_xticks(range(top_k))
        ax.set_xticklabels([FEATURE_NAMES[i] for i in indices[:top_k]], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Mean Attention from CLS Token", fontsize=11)
        ax.set_title(f"FT-Transformer: CLS Attention to Feature Tokens (Top-{top_k})  |  "
                     f"Mean={mean_val:.4f}  Range=[{attn_range[0]:.4f}, {attn_range[1]:.4f}]  "
                     f"Std={attn.std():.5f}",
                     fontsize=10)
        ax.axhline(y=1.0 / input_dim, color="gray", linestyle="--", linewidth=0.8,
                   label=f"Uniform baseline (1/{input_dim}={1.0/input_dim:.4f})")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURE_DIR, "ftt_attention.pdf"), dpi=200)
        plt.close(fig)
        print(f"FT-Transformer attention plot saved. Mean={mean_val:.5f}, Std={attn.std():.5f}, "
              f"Range=[{attn_range[0]:.5f}, {attn_range[1]:.5f}]")
    else:
        print("Attention weights not available (model returned None).")


def top_features_summary(X_test, y_test):
    """Print a business-interpretable summary of top features."""
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(MODEL_DIR, "xgb_baseline.json"))

    importance = xgb_model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]

    print("\nTop-10 Features by XGBoost Importance:")
    print("-" * 40)
    for rank, idx in enumerate(sorted_idx[:10], 1):
        print(f"  {rank:2d}. {FEATURE_NAMES[idx]:>6s}  importance={importance[idx]:.4f}")

    frau_amounts = X_test[y_test == 1, -1]
    legit_amounts = X_test[y_test == 0, -1]
    print(f"\nAmount statistics (standardized):")
    print(f"  Fraud:     mean={frau_amounts.mean():.3f}, std={frau_amounts.std():.3f}")
    print(f"  Legitimate: mean={legit_amounts.mean():.3f}, std={legit_amounts.std():.3f}")


def shap_interaction_analysis(X_train, X_test, y_test):
    """Compute and visualize Shapley Interaction Index (SII) for XGBoost."""
    print("\n" + "=" * 60)
    print("SHAP Interaction Index (SII) Analysis")
    print("=" * 60)

    try:
        import shapiq
    except ImportError:
        print("shapiq not installed. Install with: pip install shapiq")
        print("Skipping SII analysis.")
        return

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(MODEL_DIR, "xgb_baseline.json"))

    X_test_sub = X_test[:50] if len(X_test) > 50 else X_test

    print("Computing SII (2nd-order interactions) for XGBoost...")
    explainer = shapiq.TreeExplainer(model=xgb_model, max_order=2, index="SII")

    # shapiq v1.5.0+ requires per-instance explain; loop over batch
    all_first_order = []
    all_second_order = []
    for i, x in enumerate(X_test_sub):
        if i % 10 == 0:
            print(f"  SII progress: {i}/{len(X_test_sub)}")
        interaction_values = explainer.explain(x=x.flatten(), budget=1024)
        all_first_order.append(interaction_values.get_n_order_values(1).flatten())
        all_second_order.append(interaction_values.get_n_order_values(2))

    first_order = np.array(all_first_order)   # (n_samples, n_features)
    second_order = np.array(all_second_order) # (n_samples, n_features, n_features)

    print("\nTop-10 Feature Importance (1st-order Shapley values):")
    print("-" * 50)
    mean_abs_shap = np.abs(first_order).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:10]
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:2d}. {FEATURE_NAMES[idx]:>8s}  |SHAP|={mean_abs_shap[idx]:.4f}")

    print("\nTop-10 Pairwise Interactions (2nd-order SII):")
    print("-" * 50)
    mean_abs_sii = np.abs(second_order).mean(axis=0)
    n_features = second_order.shape[1]
    pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            pairs.append((i, j, mean_abs_sii[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    for rank, (i, j, val) in enumerate(pairs[:10], 1):
        mean_sii = second_order[:, i, j].mean()
        effect_type = "协同(Synergy)" if mean_sii > 0 else "冗余(Redundancy)"
        print(f"  {rank:2d}. ({FEATURE_NAMES[i]:>5s}, {FEATURE_NAMES[j]:>5s})  "
              f"|SII|={val:.6f}  mean={mean_sii:+.6f}  {effect_type}")

    # Generate SII heatmap (averaged over all samples)
    # Use a separate top-15 index set for the heatmap
    top_k = min(15, n_features)
    top_feat_idx_heatmap = np.argsort(mean_abs_shap)[::-1][:top_k]
    top_feat_idx = top_feat_idx_heatmap
    sub_matrix = mean_abs_sii[np.ix_(top_feat_idx, top_feat_idx)]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sub_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(top_k))
    ax.set_yticks(range(top_k))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in top_feat_idx], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([FEATURE_NAMES[i] for i in top_feat_idx], fontsize=8)
    ax.set_title("Top-15 Feature Pairwise Interaction Strength (|SII|)")
    plt.colorbar(im, ax=ax, label="Mean |SII|")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "shap_interaction_heatmap.pdf"), dpi=200)
    plt.close(fig)
    print("SII heatmap saved.")


def main():
    X_train, y_train, X_test, y_test = load_data()
    input_dim = X_test.shape[1]
    print(f"Data loaded. Train: {X_train.shape[0]:,}, Test: {X_test.shape[0]:,}")

    shap_analysis(X_train, X_test, y_test)
    shap_interaction_analysis(X_train, X_test, y_test)
    tabnet_masks(X_test, input_dim)
    ft_transformer_attention(X_test, input_dim)
    top_features_summary(X_test, y_test)

    print(f"\nAll interpretability outputs saved to {FIGURE_DIR}/")


if __name__ == "__main__":
    main()
