"""
预处理全流程：从原始 50:50 均衡数据 → 0.5% 欺诈率的真实场景数据。

步骤：
  1. 删除 id 列 + 处理缺失值
  2. 下采样到 0.5% 欺诈率（关键——模拟真实场景）
  3. IQR 截断（仅 Amount 列，从训练集计算）
  4. StandardScaler 标准化
  5. 分层抽样 70/15/15 划分
  6. 保存为 .pt + .pkl

关键决策解释：
  - 下采样：放弃 50% 的原始欺诈数据来模拟真实场景（0.5% ≈ 现实欺诈率）
  - IQR 截断仅对 Amount：PCA 特征 V1-V28 是线性组合，无传统"异常值"概念
  - StandardScaler 只 fit 训练集：避免数据泄露
  - 分层抽样：保持各 split 的欺诈率一致
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import RAW_DATA, PROCESSED_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED

# 目标欺诈率 0.5%（≈ 现实世界信用卡欺诈率）
TARGET_FRAUD_RATIO = 0.005
# 最多保留 15 万正常样本（控制数据量，加速训练）
MAX_LEGIT_SAMPLES = 150_000

# 显式设置所有随机种子
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_and_clean():
    """加载 CSV，删除 id 列，填充缺失值。"""
    df = pd.read_csv(RAW_DATA)
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)
    if df.isnull().sum().sum() > 0:
        df.fillna(df.median(), inplace=True)
    return df


def create_imbalance(df):
    """
    下采样到 0.5% 欺诈率——这是最重要的预处理决策。
    原始 Kaggle 数据是 50:50 均衡的（数据发布者人工平衡）。
    真实欺诈率约 0.1-1%，我们用 0.5% 作为折中。
    
    做法：保留所有正常样本（上限 15 万），随机丢弃欺诈样本至目标比例。
    """
    legit = df[df["Class"] == 0]
    fraud = df[df["Class"] == 1]
    print(f"  Raw:   {len(legit):,} legit, {len(fraud):,} fraud (balanced)")

    # 可选：限制正常样本数量（控制数据集大小）
    if MAX_LEGIT_SAMPLES and len(legit) > MAX_LEGIT_SAMPLES:
        legit = legit.sample(MAX_LEGIT_SAMPLES, random_state=SEED)

    # 计算需要保留的欺诈样本数
    target_fraud_count = int(len(legit) * TARGET_FRAUD_RATIO)
    if target_fraud_count < 500:
        target_fraud_count = min(500, len(fraud))

    fraud = fraud.sample(target_fraud_count, random_state=SEED)

    df_imb = pd.concat([legit, fraud], ignore_index=True)
    df_imb = df_imb.sample(frac=1, random_state=SEED).reset_index(drop=True)  # 打乱

    n_legit = (df_imb["Class"] == 0).sum()
    n_fraud = (df_imb["Class"] == 1).sum()
    print(f"  Imbal: {n_legit:,} legit, {n_fraud:,} fraud "
          f"(fraud={n_fraud/(n_legit+n_fraud):.4%})")
    return df_imb


def preprocess(df):
    """
    标准化 + 分层划分 + 保存。
    注意：所有变换（IQR 截断、StandardScaler）都只从训练集学习参数。
    """
    feature_cols = [c for c in df.columns if c != "Class"]
    target_col = "Class"

    X = df[feature_cols].copy()
    y = df[target_col].values.astype(np.float32)

    # 分层抽样：先分 70% 训练，再分 15%/15% 验证/测试
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - TRAIN_RATIO), stratify=y, random_state=SEED
    )
    val_frac = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_frac), stratify=y_temp, random_state=SEED
    )

    # IQR 截断——只对 Amount 做
    # PCA 特征 V1-V28 不应截断，因为它们的分布是线性组合
    iqr_cols = ["Amount"] if "Amount" in feature_cols else []
    for col in iqr_cols:
        q1, q3 = X_train[col].quantile(0.25), X_train[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        X_train[col] = X_train[col].clip(lower, upper)
        X_val[col] = X_val[col].clip(lower, upper)
        X_test[col] = X_test[col].clip(lower, upper)

    # StandardScaler：fit 在训练集，transform 应用到所有 split
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 转为 PyTorch 张量
    train_X = torch.tensor(X_train_scaled, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.float32)
    val_X = torch.tensor(X_val_scaled, dtype=torch.float32)
    val_y = torch.tensor(y_val, dtype=torch.float32)
    test_X = torch.tensor(X_test_scaled, dtype=torch.float32)
    test_y = torch.tensor(y_test, dtype=torch.float32)

    # 保存为 .pt（PyTorch 原生格式，加载快）
    torch.save((train_X, train_y), os.path.join(PROCESSED_DIR, "train.pt"))
    torch.save((val_X, val_y), os.path.join(PROCESSED_DIR, "val.pt"))
    torch.save((test_X, test_y), os.path.join(PROCESSED_DIR, "test.pt"))

    # 保存预处理器（scaler 和特征信息），供后续脚本使用
    preprocessor = {"scaler": scaler, "feature_cols": feature_cols, "input_dim": len(feature_cols)}
    with open(os.path.join(PROCESSED_DIR, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)

    # 打印各 split 的欺诈率
    for name, t in [("Train", (train_X, train_y)),
                     ("Val", (val_X, val_y)),
                     ("Test", (test_X, test_y))]:
        n_pos = t[1].sum().item()
        n_total = len(t[1])
        print(f"{name:>6s}: {n_total:,} samples, "
              f"fraud={n_pos:.0f} ({n_pos/n_total:.4%})")

    print(f"\nInput dimension: {len(feature_cols)}")
    print(f"Processed data saved to {PROCESSED_DIR}")


def main():
    print("Loading raw data...")
    df = load_and_clean()
    print(f"  Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

    print("\nCreating realistic class imbalance...")
    df = create_imbalance(df)

    preprocess(df)


if __name__ == "__main__":
    main()
