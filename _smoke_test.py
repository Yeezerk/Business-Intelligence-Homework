"""Smoke test: build each model with default and use_official=True, run a forward pass."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))

import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

INPUT_DIM = 29
BATCH = 8
X = torch.randn(BATCH, INPUT_DIM)
y = torch.randint(0, 2, (BATCH,)).float()

print("=" * 60)
print("Test 1: Self-implementation (default, use_official=False)")
print("=" * 60)

from models.mlp import MLP
m = MLP(INPUT_DIM)
out = m(X)
print(f"  MLP         out: {out.shape}  loss(BCEwLL)={torch.nn.functional.binary_cross_entropy_with_logits(out.squeeze(-1), y):.4f}")

from models.tabnet import TabNet
t = TabNet(INPUT_DIM)
out, masks = t(X, return_masks=True)
print(f"  TabNet      out: {out.shape}  masks: {masks.shape}  loss={torch.nn.functional.binary_cross_entropy_with_logits(out.squeeze(-1), y):.4f}")

from models.ft_transformer import FTTransformer
f = FTTransformer(INPUT_DIM)
out = f(X)
attn = f.get_attention_weights(X)
print(f"  FTT         out: {out.shape}  attn: {attn.shape}  loss={torch.nn.functional.binary_cross_entropy_with_logits(out.squeeze(-1), y):.4f}")

from models.deepfm import DeepFM
d = DeepFM(INPUT_DIM)
out = d(X)
print(f"  DeepFM      out: {out.shape}  loss={torch.nn.functional.binary_cross_entropy_with_logits(out.squeeze(-1), y):.4f}")

print()
print("=" * 60)
print("Test 2: Official backends (use_official=True)")
print("=" * 60)

for name, cls in [("MLP", MLP), ("TabNet", TabNet), ("FTTransformer", FTTransformer), ("DeepFM", DeepFM)]:
    try:
        m = cls(INPUT_DIM, use_official=True)
        out = m(X)
        print(f"  {name:13s}  use_official=True   out: {out.shape}  loss={torch.nn.functional.binary_cross_entropy_with_logits(out.squeeze(-1), y):.4f}")
    except ImportError as e:
        print(f"  {name:13s}  use_official=True   [SKIP] {e}")
    except Exception as e:
        print(f"  {name:13s}  use_official=True   [ERROR] {type(e).__name__}: {e}")

print()
print("All smoke tests done.")
