"""
探索性数据分析（EDA）。
关键设计：EDA 用下采样后的 0.5% 数据而非原始 50:50 均衡数据。
原因：原始均衡数据会误导——"欺诈占 50%"与现实严重不符。
EDA 需反映模型实际看到的分布，否则会产生"EDA 画饼，训练打脸"的脱节。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use("Agg")  # 非交互式后端（服务器环境无 GUI）

from config import RAW_DATA, FIGURE_DIR, SEED

os.makedirs(FIGURE_DIR, exist_ok=True)

# 下采样参数（必须与 03_preprocess.py 一致）
TARGET_FRAUD_RATIO = 0.005   # 目标欺诈率 0.5%
MAX_LEGIT_SAMPLES = 150_000  # 最多保留 15 万正常样本


def load_data():
    """加载原始数据并打印基本信息。"""
    df = pd.read_csv(RAW_DATA)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nOriginal class distribution:\n{df['Class'].value_counts()}")
    print(f"Original fraud ratio: {df['Class'].mean():.4%}")
    print(f"\nMissing values:\n{df.isnull().sum().sum()}")
    print(f"\ndtypes:\n{df.dtypes.value_counts()}")
    return df


def create_imbalance(df):
    """
    应用和 03_preprocess.py 相同的下采样。
    在 EDA 阶段就要看到模型将要面对的分布。
    """
    legit = df[df["Class"] == 0]
    fraud = df[df["Class"] == 1]

    if MAX_LEGIT_SAMPLES and len(legit) > MAX_LEGIT_SAMPLES:
        legit = legit.sample(MAX_LEGIT_SAMPLES, random_state=SEED)

    target_fraud_count = int(len(legit) * TARGET_FRAUD_RATIO)
    if target_fraud_count < 500:
        target_fraud_count = min(500, len(fraud))

    fraud = fraud.sample(target_fraud_count, random_state=SEED)
    df_imb = pd.concat([legit, fraud], ignore_index=True)
    df_imb = df_imb.sample(frac=1, random_state=SEED).reset_index(drop=True)

    n_legit = (df_imb["Class"] == 0).sum()
    n_fraud = (df_imb["Class"] == 1).sum()
    print(f"\nAfter imbalance: {n_legit:,} legit, {n_fraud:,} fraud ({n_fraud/(n_legit+n_fraud):.4%})")
    return df_imb


def plot_class_balance(df):
    """类别分布柱状图——欺诈的柱子非常短，直观展示不平衡度。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["Class"].value_counts()
    labels = ["Legitimate (0)", "Fraud (1)"]
    values = [counts[0], counts[1]]
    colors = ["#2ecc71", "#e74c3c"]

    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, v in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + max(values) * 0.02,
                f"{v:,}\n({v/len(df):.3%})", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Class Distribution (After Imbalance: 150,000 legit vs 750 fraud)", fontsize=12)
    ax.set_ylim(0, max(values) * 1.15)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "class_balance.pdf"), dpi=150)
    plt.close(fig)
    print(f"  Class balance plot saved (ratio 1:{counts[0]/counts[1]:.0f})")


def plot_amount_distribution(df):
    """
    交易金额分布（KDE 曲线叠加）：
    蓝色 = 正常交易，橙色 = 欺诈交易。
    截断到 99 分位数防止长尾拉伸。
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    legit_amount = df[df["Class"] == 0]["Amount"]
    fraud_amount = df[df["Class"] == 1]["Amount"]

    upper = df["Amount"].quantile(0.99)
    legit_clipped = legit_amount.clip(upper=upper)
    fraud_clipped = fraud_amount.clip(upper=upper)

    from scipy.stats import gaussian_kde
    for data, color, label in [
        (legit_clipped, "#2e77bf", "Legitimate (n=150,000)"),
        (fraud_clipped, "#e67e22", "Fraud (n=750)"),
    ]:
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 500)
        ax.plot(x_range, kde(x_range), color=color, lw=2.2, label=label)
        ax.fill_between(x_range, kde(x_range), color=color, alpha=0.12)

    ax.set_xlabel("Transaction Amount (standardized)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Transaction Amount Distribution: Legitimate vs Fraud", fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "amount_distribution.pdf"), dpi=200)
    plt.close(fig)


def plot_correlation_heatmap(df, top_k=20):
    """
    特征相关性热力图（Top-20 与 Class 最相关的特征）。
    红蓝配色：红色 = 正相关，蓝色 = 负相关。
    注意 PCA 特征 V1-V28 之间相关性应该较低（PCA 的性质）。
    """
    corr_with_target = df.corr(numeric_only=True)["Class"].abs().sort_values(ascending=False)
    selected = corr_with_target.head(top_k).index.tolist()
    if "Class" not in selected:
        selected.append("Class")

    corr_mat = df[selected].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
    sns.heatmap(corr_mat, mask=mask, cmap="RdBu_r", center=0,
                annot=True, fmt=".2f", linewidths=0.3, ax=ax,
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.7})
    ax.set_title("Feature Correlation Heatmap (Top-20 features by |corr| with Class)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "correlation_heatmap.pdf"), dpi=150)
    plt.close(fig)


def plot_feature_distributions(df):
    """Top-8 特征的类别条件分布对比（2×4 小多图）。"""
    v_cols = [c for c in df.columns if c.startswith("V")]
    corrs = df[v_cols + ["Amount"]].apply(lambda c: df["Class"].corr(c)).abs()
    top = corrs.sort_values(ascending=False).head(8).index.tolist()

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, col in zip(axes.flat, top):
        for label, color in [(0, "#2ecc71"), (1, "#e74c3c")]:
            subset = df[df["Class"] == label][col]
            ax.hist(subset, bins=60, alpha=0.5, color=color, label=f"Class={label}", density=True)
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("Top-8 Feature Distributions by Class", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "feature_distributions.pdf"), dpi=150)
    plt.close(fig)


def print_summary_stats(df):
    """关键统计量汇总，可直接复制到论文中。"""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total transactions:   {len(df):,}")
    print(f"Fraudulent:           {df['Class'].sum():,} ({df['Class'].mean():.4%})")
    print(f"Legitimate:           {(df['Class'] == 0).sum():,}")
    print(f"Features:             {df.shape[1]}")
    print(f"Mean Amount (all):    {df['Amount'].mean():.2f}")
    print(f"Mean Amount (fraud):  {df[df['Class'] == 1]['Amount'].mean():.2f}")
    print(f"Mean Amount (legit):  {df[df['Class'] == 0]['Amount'].mean():.2f}")
    print(f"Max Amount:           {df['Amount'].max():.2f}")
    print(f"Min Amount:           {df['Amount'].min():.2f}")


def main():
    df = load_data()
    df = create_imbalance(df)  # 先下采再到真实分布再做 EDA

    print_summary_stats(df)
    print("\nGenerating EDA plots on imbalanced data...")
    plot_class_balance(df)
    plot_amount_distribution(df)
    plot_correlation_heatmap(df)
    plot_feature_distributions(df)
    print(f"Plots saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
