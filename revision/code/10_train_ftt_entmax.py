"""
10_train_ftt_entmax.py — FTT + α-Entmax + FocalLoss
对照：Softmax(α→1) vs Sparsemax(α=2) vs Entmax(α=1.5)
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score

from config import *
from models.ft_transformer import FTTransformer
from training.losses import FocalLoss
from training.metrics import compute_all_metrics, find_best_threshold
from training.trainer import Trainer

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

ENTMAX_ALPHA = 1.5

print("=" * 60)
print(f"FTT + Alpha-Entmax (alpha={ENTMAX_ALPHA}) + FocalLoss")
print("=" * 60)

# Load data
train_X, train_y = torch.load(os.path.join(PROCESSED_DIR, "train.pt"), weights_only=True)
val_X, val_y = torch.load(os.path.join(PROCESSED_DIR, "val.pt"), weights_only=True)
test_X, test_y = torch.load(os.path.join(PROCESSED_DIR, "test.pt"), weights_only=True)

# Model: Entmax
model = FTTransformer(INPUT_DIM, use_sparsemax=False, use_entmax=True,
                      entmax_alpha=ENTMAX_ALPHA).to(DEVICE)
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

# Training
train_ds = TensorDataset(train_X, train_y)
val_ds = TensorDataset(val_X, val_y)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=8)
criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

trainer = Trainer(model=model, loss_fn=criterion, optimizer=optimizer,
                  scheduler=scheduler, model_name="FTT-Entmax15", patience=EARLY_STOP_PATIENCE)
trainer.fit(train_loader, val_loader, max_epochs=MAX_EPOCHS)
trainer.load_best()

# Evaluate
@torch.no_grad()
def get_scores(m, X):
    B = 2048; s = []
    for i in range(0, len(X), B):
        s.append(torch.sigmoid(m(X[i:i+B].to(DEVICE)).squeeze(-1)).cpu())
    return torch.cat(s).numpy()

test_s = get_scores(model, test_X)
val_s = get_scores(model, val_X)
test_y_np = test_y.numpy(); val_y_np = val_y.numpy()

auprc = average_precision_score(test_y_np, test_s)
best_th, _ = find_best_threshold(val_y_np, val_s)
m = compute_all_metrics(test_y_np, test_s, best_th)

# Attention
idx = np.random.RandomState(42).choice(len(test_X), 200, replace=False)
X_sub = test_X[idx].to(DEVICE)
attn = model.get_attention_weights(X_sub).cpu().numpy()
asd = float(np.std(attn)); amn = float(np.mean(attn)); ub = 1.0/29

print(f"\nAUPRC:       {auprc:.4f}")
print(f"F1:          {m['F1']:.4f}")
print(f"AttnStd:     {asd:.6f}")
print(f"Std/Uniform: {asd/ub:.4f}")
print(f"Cost:        {m['Cost']}")
print(f"\nBaselines:")
print(f"  Softmax:   AUPRC=0.9126  AttnStd=0.00125  Std/U=0.036")
print(f"  Sparsemax: AUPRC=0.9384  AttnStd=0.00942  Std/U=0.273")

torch.save(model.state_dict(), os.path.join(MODEL_DIR, "ftt_entmax_best.pt"))
print("\nSaved: ftt_entmax_best.pt")
