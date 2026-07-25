"""
BCE消融实验：检验注意力塌缩的根因是PCA空间还是Focal Loss。

核心逻辑：
  - 训练 FTT+Softmax+BCE（与 FTT+Softmax+FocalLoss 对比）
  - 如果 BCE 版注意力也塌缩 → PCA空间是根因
  - 如果 BCE 版注意力不塌缩 → Focal Loss 至少是促进因素

同时训练 FTT+Sparsemax+BCE 作为对照，验证 Sparsemax 的独立效果。
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    PROCESSED_DIR, MODEL_DIR, RESULT_DIR, DEVICE, SEED,
    BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
)
from training.trainer import Trainer
from training.metrics import compute_all_metrics, find_best_threshold

# ═══════════════════════════════════════════════════════════════
# 种子重置
# ═══════════════════════════════════════════════════════════════
def reset_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════
def load_data():
    train_X, train_y = torch.load(os.path.join(PROCESSED_DIR, "train.pt"), weights_only=True)
    val_X, val_y = torch.load(os.path.join(PROCESSED_DIR, "val.pt"), weights_only=True)
    test_X, test_y = torch.load(os.path.join(PROCESSED_DIR, "test.pt"), weights_only=True)
    return train_X, train_y, val_X, val_y, test_X, test_y

def create_loaders(train_X, train_y, val_X, val_y, test_X, test_y):
    train_ds = TensorDataset(train_X, train_y)
    val_ds = TensorDataset(val_X, val_y)
    test_ds = TensorDataset(test_X, test_y)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False)
    return train_loader, val_loader, test_loader

# ═══════════════════════════════════════════════════════════════
# 构建可切换 Softmax/Sparsemax 的 FTT 模型
# ═══════════════════════════════════════════════════════════════
from models.ft_transformer import (
    FTTransformer, FeatureTokenizer, TransformerBlock,
    sparsemax, FTT_D_MODEL, FTT_N_HEADS, FTT_N_LAYERS, FTT_FFN_DIM
)

class SwitchableAttention(nn.Module):
    """可切换 Softmax/Sparsemax 的多头注意力。"""

    def __init__(self, d_model, n_heads, attn_dropout=0.1, use_sparsemax=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_sparsemax = use_sparsemax
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x, return_attention=False):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = self.head_dim ** -0.5
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale

        if self.use_sparsemax:
            attn = sparsemax(attn_logits, dim=-1)
        else:
            attn = torch.softmax(attn_logits, dim=-1)

        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        result = self.out_proj(out)
        if return_attention:
            return result, attn
        return result

class SwitchableTransformerBlock(nn.Module):
    """可切换注意力的 Transformer 块。"""
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1, use_sparsemax=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SwitchableAttention(d_model, n_heads, dropout, use_sparsemax)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ffn_dim, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + self.dropout1(attn_out)
        else:
            x = x + self.dropout1(self.attn(self.norm1(x)))
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        if return_attention:
            return x, attn_weights
        return x

class SwitchableFTT(nn.Module):
    """可切换 Softmax/Sparsemax + 可提取注意力的 FT-Transformer。"""

    def __init__(self, input_dim, d_model=FTT_D_MODEL, n_heads=FTT_N_HEADS,
                 n_layers=FTT_N_LAYERS, ffn_dim=FTT_FFN_DIM, dropout=0.1, use_sparsemax=True):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_sparsemax = use_sparsemax

        self.tokenizer = FeatureTokenizer(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            SwitchableTransformerBlock(d_model, n_heads, ffn_dim, dropout, use_sparsemax)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)
        return self.head(tokens[:, 0, :])

    def get_attention_weights(self, x):
        """提取最后一层 CLS→特征的注意力权重，返回 (29,) 张量。"""
        self.eval()
        with torch.no_grad():
            tokens = self.tokenizer(x)
            cls = self.cls_token.expand(x.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            for i, block in enumerate(self.blocks):
                if i == self.n_layers - 1:
                    _, attn = block(tokens, return_attention=True)
                    # attn: (B, n_heads, 30, 30), CLS=index 0, features=1..29
                    cls_attn = attn[:, :, 0, 1:]  # (B, n_heads, 29)
                    return cls_attn.mean(dim=(0, 1))  # 在 batch 和 head 维度平均
                tokens = block(tokens)
        return None


# ═══════════════════════════════════════════════════════════════
# 训练 + 注意力提取
# ═══════════════════════════════════════════════════════════════
def train_and_extract(name, use_sparsemax, loss_fn, train_loader, val_loader, test_loader,
                      input_dim, lr=LEARNING_RATE):
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"  use_sparsemax={use_sparsemax}, loss={type(loss_fn).__name__}")
    print(f"{'='*60}")

    model = SwitchableFTT(input_dim, use_sparsemax=use_sparsemax).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8
    )
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer,
                      scheduler=scheduler, model_name=name)
    trainer.fit(train_loader, val_loader, max_epochs=MAX_EPOCHS)
    trainer.load_best()

    # 测试集评估
    test_scores, test_labels = trainer.predict(test_loader)
    val_scores, val_labels = trainer.predict(val_loader)
    best_threshold, _ = find_best_threshold(val_labels, val_scores)
    metrics = compute_all_metrics(test_labels, test_scores, threshold=best_threshold)
    metrics["Model"] = name

    # 保存模型
    save_path = os.path.join(MODEL_DIR, f"{name.replace(' ', '_').replace('(','').replace(')','')}_best.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # 提取注意力权重
    test_X = test_loader.dataset.tensors[0][:200].to(DEVICE)
    attn_weights = model.get_attention_weights(test_X)
    if attn_weights is not None:
        attn_np = attn_weights.cpu().numpy()
        uniform_baseline = 1.0 / input_dim
        print(f"\n  Attention Analysis for {name}:")
        print(f"    Mean:   {attn_np.mean():.6f}")
        print(f"    Std:    {attn_np.std():.6f}")
        print(f"    Min:    {attn_np.min():.6f}")
        print(f"    Max:    {attn_np.max():.6f}")
        print(f"    Uniform baseline (1/d): {uniform_baseline:.6f}")
        print(f"    Std / Uniform baseline: {attn_np.std() / uniform_baseline:.4f}")
        metrics["AttnMean"] = float(attn_np.mean())
        metrics["AttnStd"] = float(attn_np.std())
        metrics["AttnMin"] = float(attn_np.min())
        metrics["AttnMax"] = float(attn_np.max())

    print(f"\n{name} Test Results (threshold={best_threshold:.3f}):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics, attn_weights


# ═══════════════════════════════════════════════════════════════
# 主流程：两个对照实验
# ═══════════════════════════════════════════════════════════════
def main():
    print(f"Device: {DEVICE}")
    train_X, train_y, val_X, val_y, test_X, test_y = load_data()
    input_dim = train_X.shape[1]
    train_loader, val_loader, test_loader = create_loaders(
        train_X, train_y, val_X, val_y, test_X, test_y
    )
    print(f"Train: {len(train_y):,} samples")

    bce_loss = nn.BCEWithLogitsLoss()

    # 实验1: FTT + Softmax + BCE — 核心消融
    reset_seed(SEED)
    metrics_soft_bce, attn_soft_bce = train_and_extract(
        "FTT-Softmax-BCE", use_sparsemax=False, loss_fn=bce_loss,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        input_dim=input_dim, lr=5e-4,
    )

    # 实验2: FTT + Sparsemax + BCE — 对照（分离 Sparsemax 和 Focal Loss 的独立效应）
    reset_seed(SEED)
    metrics_sparse_bce, attn_sparse_bce = train_and_extract(
        "FTT-Sparsemax-BCE", use_sparsemax=True, loss_fn=bce_loss,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        input_dim=input_dim, lr=5e-4,
    )

    # ── 汇总对比 ──
    print(f"\n{'='*70}")
    print("BCE ABLATION SUMMARY: Attention Collapse Root-Cause Analysis")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'AUPRC':>8} {'AttnStd':>10} {'Std/Uniform':>12} {'Conclusion'}")
    print("-" * 70)

    # 引用原始 Focal Loss 结果（从已有训练结果）
    print(f"{'FTT-Softmax-FocalLoss':<25} {'0.8738':>8} {'0.0004':>10} {'0.0116':>12} {'Collapse (baseline)':<20}")
    print(f"{'FTT-Softmax-BCE':<25} {metrics_soft_bce['AUPRC']:>8.4f} "
          f"{metrics_soft_bce.get('AttnStd', 0):>10.4f} "
          f"{metrics_soft_bce.get('AttnStd', 0)/0.0345:>12.4f} "
          f"{'BCE ablation':<20}")
    print(f"{'FTT-Sparsemax-BCE':<25} {metrics_sparse_bce['AUPRC']:>8.4f} "
          f"{metrics_sparse_bce.get('AttnStd', 0):>10.4f} "
          f"{metrics_sparse_bce.get('AttnStd', 0)/0.0345:>12.4f} "
          f"{'Sparsemax+BCE control':<20}")

    print(f"\nInterpretation:")
    soft_bce_std = metrics_soft_bce.get('AttnStd', 0)
    if soft_bce_std < 0.002:  # still very concentrated
        print(f"  → BCE attention std = {soft_bce_std:.5f} ≈ FocalLoss std (0.0004)")
        print(f"  → CONCLUSION: PCA space IS the root cause of attention collapse.")
        print(f"  → Focal Loss is NOT the primary driver of uniform attention.")
    else:
        print(f"  → BCE attention std = {soft_bce_std:.5f} >> FocalLoss std (0.0004)")
        print(f"  → CONCLUSION: Focal Loss IS a contributing factor to attention collapse.")
        print(f"  → The collapse is a PCA+FocalLoss interaction effect.")

    # 保存汇总 CSV
    import pandas as pd
    df = pd.DataFrame([metrics_soft_bce, metrics_sparse_bce])
    csv_path = os.path.join(RESULT_DIR, "bce_ablation_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
