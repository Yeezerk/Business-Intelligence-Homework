"""
11_train_lor_mim_entmax.py — LOR-MIM + FTT + Alpha-Entmax + FocalLoss
联合：数据端对称破缺 + 注意力端 α-Entmax 稀疏化
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score

from config import *
from models.ft_transformer import FTTransformer
from models.lor_mim import LORMIM, build_affinity_groups
from training.losses import FocalLoss
from training.metrics import compute_all_metrics, find_best_threshold
from training.trainer import Trainer

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

N_GROUPS = 5; ENTMAX_ALPHA = 1.5

class LORMIM_FTT_Entmax(nn.Module):
    def __init__(self, input_dim, group_assignments, entmax_alpha=1.5):
        super().__init__()
        self.lor_mim = LORMIM(input_dim - 1, group_assignments)
        self.ftt = FTTransformer(input_dim, use_sparsemax=False,
                                 use_entmax=True, entmax_alpha=entmax_alpha)
    def forward(self, x):
        x_t, _ = self.lor_mim(x); return self.ftt(x_t)
    def get_attention_weights(self, x):
        x_t, _ = self.lor_mim(x); return self.ftt.get_attention_weights(x_t)

print("=" * 60)
print(f"LOR-MIM + FTT + Alpha-Entmax (alpha={ENTMAX_ALPHA}) + FocalLoss")
print("=" * 60)

# Load
train_X, train_y = torch.load(os.path.join(PROCESSED_DIR, "train.pt"), weights_only=True)
val_X, val_y = torch.load(os.path.join(PROCESSED_DIR, "val.pt"), weights_only=True)
test_X, test_y = torch.load(os.path.join(PROCESSED_DIR, "test.pt"), weights_only=True)

# Affinity groups
ga = build_affinity_groups(train_X[:, :28].numpy(), train_X[:, 28].numpy(),
                           n_groups=N_GROUPS, random_state=SEED)

# Model
model = LORMIM_FTT_Entmax(INPUT_DIM, ga, entmax_alpha=ENTMAX_ALPHA).to(DEVICE)
n_total = sum(p.numel() for p in model.parameters())
n_lor = sum(p.numel() for p in model.lor_mim.parameters())
print(f"  Total params: {n_total:,}  LOR-MIM: {n_lor:,} ({n_lor/n_total*100:.2f}%)")

# Train
train_ds = TensorDataset(train_X, train_y)
val_ds = TensorDataset(val_X, val_y)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=8)
criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

trainer = Trainer(model=model, loss_fn=criterion, optimizer=optimizer,
                  scheduler=scheduler, model_name="LOR-MIM-Entmax15",
                  patience=EARLY_STOP_PATIENCE)
trainer.fit(train_loader, val_loader, max_epochs=MAX_EPOCHS)
trainer.load_best()

# Eval
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

idx = np.random.RandomState(42).choice(len(test_X), 200, replace=False)
attn = model.get_attention_weights(test_X[idx].to(DEVICE)).cpu().numpy()
asd = float(np.std(attn)); amn = float(np.mean(attn)); ub = 1.0/29

@torch.no_grad()
def gs(m, X):
    B = 2048; o = []
    for i in range(0, len(X), B):
        _, g = m.lor_mim(X[i:i+B].to(DEVICE))
        o.append(g.detach().cpu())
    return torch.cat(o).numpy()
gm = gs(model, test_X).mean(0)
dv = model.lor_mim.get_mixing_divergence()

print(f"\nAUPRC:       {auprc:.4f}")
print(f"F1:          {m['F1']:.4f}")
print(f"AttnStd:     {asd:.6f}")
print(f"Std/Uniform: {asd/ub:.4f}")
print(f"Cost:        {m['Cost']}")
print(f"Gate means:  {gm.round(4)}")
print(f"Mixing div:  { {k:round(v,3) for k,v in dv.items()} }")
print(f"\nAll baselines:")
print(f"  Softmax:        AUPRC=0.9126  AttnStd=0.00125  Std/U=0.036")
print(f"  Sparsemax:      AUPRC=0.9384  AttnStd=0.00942  Std/U=0.273")
print(f"  LOR-MIM+Softmax: AUPRC=0.9242  AttnStd=0.01401  Std/U=0.406")

torch.save(model.state_dict(), os.path.join(MODEL_DIR, "lor_mim_entmax_best.pt"))
print("\nSaved: lor_mim_entmax_best.pt")
