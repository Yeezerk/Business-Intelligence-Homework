"""12_train_lor_mim_sparsemax.py — LOR-MIM + FTT + Sparsemax + FocalLoss"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np; import torch; import torch.nn as nn
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

class LORMIM_FTT(nn.Module):
    def __init__(self, d_in, ga, use_sm=False):
        super().__init__()
        self.lor_mim = LORMIM(d_in-1, ga)
        self.ftt = FTTransformer(d_in, use_sparsemax=use_sm)
    def forward(self, x):
        x_t, _ = self.lor_mim(x); return self.ftt(x_t)
    def get_attention_weights(self, x):
        x_t, _ = self.lor_mim(x); return self.ftt.get_attention_weights(x_t)

print("LOR-MIM + Sparsemax + FocalLoss")
tX, ty = torch.load(PROCESSED_DIR+"/train.pt", weights_only=True)
vX, vy = torch.load(PROCESSED_DIR+"/val.pt", weights_only=True)
eX, ey = torch.load(PROCESSED_DIR+"/test.pt", weights_only=True)

ga = build_affinity_groups(tX[:,:28].numpy(), tX[:,28].numpy(), n_groups=5, random_state=SEED)
model = LORMIM_FTT(INPUT_DIM, ga, use_sm=True).to(DEVICE)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

tds = TensorDataset(tX, ty); vds = TensorDataset(vX, vy)
tl = DataLoader(tds, batch_size=BATCH_SIZE, shuffle=True)
vl = DataLoader(vds, batch_size=BATCH_SIZE*2, shuffle=False)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=8)
crit = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
trainer = Trainer(model=model, loss_fn=crit, optimizer=opt, scheduler=sch,
                  model_name="LOR-MIM-Sparsemax", patience=EARLY_STOP_PATIENCE)
trainer.fit(tl, vl, max_epochs=MAX_EPOCHS)
trainer.load_best()

@torch.no_grad()
def sc(m, X):
    B=2048; s=[]
    for i in range(0,len(X),B): s.append(torch.sigmoid(m(X[i:i+B].to(DEVICE)).squeeze(-1)).cpu())
    return torch.cat(s).numpy()

es=sc(model,eX); vs=sc(model,vX)
eyn=ey.numpy(); vyn=vy.numpy()
auprc = average_precision_score(eyn, es)
bt,_ = find_best_threshold(vyn, vs)
mt = compute_all_metrics(eyn, es, bt)

idx = np.random.RandomState(42).choice(len(eX),200,replace=False)
attn = model.get_attention_weights(eX[idx].to(DEVICE)).cpu().numpy()
asd=float(np.std(attn)); ub=1.0/29

@torch.no_grad()
def gs(m,X):
    B=2048; o=[]
    for i in range(0,len(X),B):
        _,g=m.lor_mim(X[i:i+B].to(DEVICE)); o.append(g.detach().cpu())
    return torch.cat(o).numpy()
gm=gs(model,eX).mean(0)
dv=model.lor_mim.get_mixing_divergence()

print(f"\nAUPRC: {auprc:.4f}  F1: {mt['F1']:.4f}  Cost: {mt['Cost']}")
print(f"AttnStd: {asd:.6f}  Std/U: {asd/ub:.4f}")
print(f"Gate: {gm.round(4)}")
print(f"Mixing: { {k:round(v,3) for k,v in dv.items()} }")
print(f"\nvs LOR-MIM+Softmax:  AUPRC=0.9242  AttnStd=0.01401  Std/U=0.406")
print(f"vs Sparsemax alone:   AUPRC=0.9384  AttnStd=0.00942  Std/U=0.273")

torch.save(model.state_dict(), MODEL_DIR+"/lor_mim_sparsemax_best.pt")
print("\nSaved.")
