"""
BCE消融：FTT + Softmax + BCE（手写实现，ReGLU）。
与 04 唯一变量：损失函数（BCE vs Focal Loss）。
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
from models.ft_transformer import FTTransformer


def reset_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_processed():
    train_X, train_y = torch.load(os.path.join(PROCESSED_DIR, "train.pt"), weights_only=True)
    val_X, val_y = torch.load(os.path.join(PROCESSED_DIR, "val.pt"), weights_only=True)
    test_X, test_y = torch.load(os.path.join(PROCESSED_DIR, "test.pt"), weights_only=True)
    return train_X, train_y, val_X, val_y, test_X, test_y


def main():
    print(f"Device: {DEVICE}")
    train_X, train_y, val_X, val_y, test_X, test_y = load_processed()
    input_dim = train_X.shape[1]

    train_ds = TensorDataset(train_X, train_y)
    val_ds = TensorDataset(val_X, val_y)
    test_ds = TensorDataset(test_X, test_y)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

    reset_seed(SEED)
    model = FTTransformer(input_dim, use_sparsemax=False).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FTT-Softmax-BCE (ReGLU) params: {n_params:,}")

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)

    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer,
                      scheduler=scheduler, model_name="FTT-Softmax-BCE-ReGLU")
    trainer.fit(train_loader, val_loader, max_epochs=MAX_EPOCHS)
    trainer.load_best()

    test_scores, test_labels = trainer.predict(test_loader)
    val_scores, val_labels = trainer.predict(val_loader)
    best_threshold, _ = find_best_threshold(val_labels, val_scores)
    metrics = compute_all_metrics(test_labels, test_scores, threshold=best_threshold)
    metrics["Model"] = "FTT-Softmax-BCE"
    metrics["Params"] = n_params

    test_X_sub = test_X[:200].to(DEVICE)
    attn = model.get_attention_weights(test_X_sub)
    if attn is not None:
        attn_np = attn.cpu().numpy()
        uniform_baseline = 1.0 / input_dim
        metrics["AttnMean"] = float(attn_np.mean())
        metrics["AttnStd"] = float(attn_np.std())
        print(f"  Attention: mean={attn_np.mean():.5f}, std={attn_np.std():.5f}, "
              f"std/uniform={attn_np.std()/uniform_baseline:.4f}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "ftt_softmax_bce_best.pt"))
    print(f"\nFTT-Softmax-BCE Results (threshold={best_threshold:.3f}):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    import pandas as pd
    pd.DataFrame([metrics]).to_csv(os.path.join(RESULT_DIR, "ftt_bce_ablation_metrics.csv"), index=False)


if __name__ == "__main__":
    main()
