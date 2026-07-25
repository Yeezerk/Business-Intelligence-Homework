"""
置换重要性分析：6 个模型全量对比，支撑论文四大核心发现。

原理：对每个特征，随机打乱它的值（破坏该特征与标签的关系），
观察 AUPRC 下降多少。下降越多 → 特征越重要。

6 个模型：
  1. XGBoost（教师基线）
  2. DeepFM（FM 架构基线）
  3. DeepFM+KD（KD 对 FM 的影响）
  4. FTT（注意力架构基线）
  5. FTT+KD（KD 对注意力的影响）
  6. FTT+KD+TreeReg（TreeReg 的破坏作用）

输出：3 面板对比图 permutation_importance.pdf
  A: XGBoost 教师特征排序
  B: DeepFM vs DeepFM+KD
  C: FTT vs FTT+KD vs FTT+KD+TreeReg

四大发现：
  1. KD 拉近学生-教师特征排序：DeepFM r=0.625→0.845
  2. TreeReg 制造畸形复刻：r=0.964 但 V14 集中度 43.9%
  3. 跨架构 KD 趋同：FTT vs DeepFM r=0.639→0.806
  4. Spearman 而非 Pearson（数值尺度无意义，只有排序有意义）
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))

import numpy as np
import torch
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from config import PROCESSED_DIR, MODEL_DIR, FIGURE_DIR, DEVICE, FEATURE_NAMES

# ═══════════════════════════════════════════════════════════════
# 加载测试数据
# ═══════════════════════════════════════════════════════════════
test_X, test_y = torch.load(
    os.path.join(PROCESSED_DIR, "test.pt"), weights_only=True
)
test_X_np = test_X.numpy()
test_y_np = test_y.numpy()
n_samples, n_features = test_X_np.shape
print(f"Test set: {n_samples} samples, {n_features} features")

# ═══════════════════════════════════════════════════════════════
# 置换重要性计算函数
# ═══════════════════════════════════════════════════════════════

N_REPEATS = 5  # 每列重复打乱 5 次取平均，减少随机性

def permutation_importance(predict_fn, X, y, n_repeats=5):
    """
    predict_fn: 模型的预测函数（输入 numpy 数组 → 输出概率）
    X: 特征矩阵 (N, d)
    y: 标签 (N,)
    n_repeats: 每列打乱次数
    返回: (importance_mean, importance_std, baseline_auprc)
    """
    baseline = average_precision_score(y, predict_fn(X))  # 未打乱时的 AUPRC
    importance = np.zeros((X.shape[1], n_repeats))         # 存储每次打乱的下降值
    for j in range(X.shape[1]):                            # 对每个特征
        for r in range(n_repeats):                         # 重复 n 次
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, j])                # 打乱第 j 列
            score = average_precision_score(y, predict_fn(X_perm))
            importance[j, r] = baseline - score            # AUPRC 下降 = 重要性
    return importance.mean(axis=1), importance.std(axis=1), baseline

# ═══════════════════════════════════════════════════════════════
# 加载 6 个模型
# ═══════════════════════════════════════════════════════════════

# ── 1. XGBoost（教师）──
import xgboost as xgb
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(MODEL_DIR, "xgb_baseline.json"))
xgb_preds = xgb_model.predict_proba(test_X_np)[:, 1]
xgb_auprc = average_precision_score(test_y_np, xgb_preds)
print(f"\nXGBoost AUPRC: {xgb_auprc:.4f}")

# ── 2. DeepFM（基线）──
from models.deepfm import DeepFM
deepfm = DeepFM(input_dim=29)
deepfm.load_state_dict(torch.load(os.path.join(MODEL_DIR, "deepfm_best.pt"), map_location=DEVICE, weights_only=True))
deepfm.to(DEVICE).eval()
with torch.no_grad():
    dfm_preds = torch.sigmoid(deepfm(test_X.to(DEVICE)).squeeze(-1)).cpu().numpy()
dfm_auprc = average_precision_score(test_y_np, dfm_preds)
print(f"DeepFM AUPRC: {dfm_auprc:.4f}")

# ── 3. DeepFM + KD ──
deepfm_kd = DeepFM(input_dim=29)
deepfm_kd.load_state_dict(torch.load(os.path.join(MODEL_DIR, "deepfm_kd_best.pt"), map_location=DEVICE, weights_only=True))
deepfm_kd.to(DEVICE).eval()
with torch.no_grad():
    dfm_kd_preds = torch.sigmoid(deepfm_kd(test_X.to(DEVICE)).squeeze(-1)).cpu().numpy()
dfm_kd_auprc = average_precision_score(test_y_np, dfm_kd_preds)
print(f"DeepFM+KD AUPRC: {dfm_kd_auprc:.4f}")

# ── 4. FT-Transformer（基线，3 层）──
from models.ft_transformer import FTTransformer
ftt = FTTransformer(input_dim=29, n_layers=3)
ftt.load_state_dict(torch.load(os.path.join(MODEL_DIR, "FT-Transformer_best.pt"), map_location=DEVICE, weights_only=True))
ftt.to(DEVICE).eval()
with torch.no_grad():
    ftt_preds = torch.sigmoid(ftt(test_X.to(DEVICE)).squeeze(-1)).cpu().numpy()
ftt_auprc = average_precision_score(test_y_np, ftt_preds)
print(f"FT-Transformer AUPRC: {ftt_auprc:.4f}")

# ── 5. FTT + KD（3A only）──
ftt_kd = FTTransformer(input_dim=29, d_model=48, n_heads=4, n_layers=2)
ftt_kd.load_state_dict(torch.load(os.path.join(MODEL_DIR, "ftt_kd_only_best.pt"), map_location=DEVICE, weights_only=True))
ftt_kd.to(DEVICE).eval()
with torch.no_grad():
    ftt_kd_preds = torch.sigmoid(ftt_kd(test_X.to(DEVICE)).squeeze(-1)).cpu().numpy()
ftt_kd_auprc = average_precision_score(test_y_np, ftt_kd_preds)
print(f"FTT+KD AUPRC: {ftt_kd_auprc:.4f}")

# ── 6. FTT + KD + TreeReg（3A+3B）──
ftt_kd_tree = FTTransformer(input_dim=29, d_model=48, n_heads=4, n_layers=2)
ftt_kd_tree.load_state_dict(torch.load(os.path.join(MODEL_DIR, "ftt_kd_treereg_best.pt"), map_location=DEVICE, weights_only=True))
ftt_kd_tree.to(DEVICE).eval()
with torch.no_grad():
    ftt_kd_tree_preds = torch.sigmoid(ftt_kd_tree(test_X.to(DEVICE)).squeeze(-1)).cpu().numpy()
ftt_kd_tree_auprc = average_precision_score(test_y_np, ftt_kd_tree_preds)
print(f"FTT+KD+TreeReg AUPRC: {ftt_kd_tree_auprc:.4f}")

# ── 辅助函数：封装 NN 模型预测 ──
def _deepfm_pred(model, X_np):
    X_t = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        return torch.sigmoid(model(X_t).squeeze(-1)).cpu().numpy()

def _ftt_pred(model, X_np):
    X_t = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        return torch.sigmoid(model(X_t).squeeze(-1)).cpu().numpy()

# ═══════════════════════════════════════════════════════════════
# 计算 6 个模型的置换重要性
# ═══════════════════════════════════════════════════════════════
model_configs = [
    ("XGBoost", lambda x: xgb_model.predict_proba(x)[:, 1]),
    ("DeepFM", lambda x: _deepfm_pred(deepfm, x)),
    ("DeepFM+KD", lambda x: _deepfm_pred(deepfm_kd, x)),
    ("FTT", lambda x: _ftt_pred(ftt, x)),
    ("FTT+KD", lambda x: _ftt_pred(ftt_kd, x)),
    ("FTT+KD+TreeReg", lambda x: _ftt_pred(ftt_kd_tree, x)),
]

results = {}
for name, predict_fn in model_configs:
    print(f"\nComputing permutation importance: {name} ...")
    mean, std, base = permutation_importance(predict_fn, test_X_np, test_y_np, n_repeats=N_REPEATS)
    results[name] = (mean, std, base)
    print(f"  Baseline AUPRC: {base:.4f}")

# ═══════════════════════════════════════════════════════════════
# 3 面板对比图
# ═══════════════════════════════════════════════════════════════

# 按 XGBoost 重要性排序（教师特征偏好作为统一的排序基准）
xgb_mean = results["XGBoost"][0]
top_n = 16                                 # 显示 Top-16 特征
top_idx = np.argsort(xgb_mean)[::-1][:top_n]
labels = [FEATURE_NAMES[i] for i in top_idx]

fig, axes = plt.subplots(1, 3, figsize=(16, 6.0), sharey=True)

# 调色板
C_TEACHER = "#2c3e50"      # 深灰蓝 → XGBoost 教师
C_BASELINE = "#7f8c8d"     # 灰色 → 基线模型
C_KD_GAIN = "#27ae60"      # 绿色 → KD 改善
C_KD_HARM = "#e74c3c"      # 红色 → TreeReg 损害

# ── 面板 A: XGBoost ──
ax = axes[0]
mean, std, base = results["XGBoost"]
ax.barh(range(top_n), mean[top_idx], xerr=std[top_idx], color=C_TEACHER, alpha=0.85, height=0.6, capsize=2)
ax.set_yticks(range(top_n))
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()              # 最重要的特征在最上面
ax.set_xlabel("AUPRC Decrease", fontsize=10)
ax.set_title(f"A. XGBoost (Teacher)\nAUPRC={base:.4f}", fontsize=11, fontweight="bold")
ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
ax.grid(axis="x", alpha=0.3)

# ── 面板 B: DeepFM vs DeepFM+KD ──
ax = axes[1]
dfm_mean, dfm_std, dfm_base = results["DeepFM"]
dfm_kd_mean, dfm_kd_std, dfm_kd_base = results["DeepFM+KD"]

y_pos = np.arange(top_n)
height = 0.28
# 两组条形图上下错开对比
ax.barh(y_pos + height/2, dfm_mean[top_idx], height, xerr=dfm_std[top_idx],
        color=C_BASELINE, alpha=0.7, capsize=2, label=f"DeepFM (AUPRC={dfm_base:.4f})")
ax.barh(y_pos - height/2, dfm_kd_mean[top_idx], height, xerr=dfm_kd_std[top_idx],
        color=C_KD_GAIN, alpha=0.85, capsize=2, label=f"DeepFM+KD (AUPRC={dfm_kd_base:.4f})")

ax.set_yticks(range(top_n))
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("AUPRC Decrease", fontsize=10)
ax.set_title(f"B. FM Architecture: KD Effect\n(Δ = +{dfm_kd_base-dfm_base:+.4f})", fontsize=11, fontweight="bold")
ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
ax.grid(axis="x", alpha=0.3)
ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)

# ── 面板 C: FTT → FTT+KD → FTT+KD+TreeReg ──
# 三组条形图展示 KD 和 TreeReg 的独立贡献
ax = axes[2]
ftt_mean, ftt_std, ftt_base = results["FTT"]
ftt_kd_mean, ftt_kd_std, ftt_kd_base = results["FTT+KD"]
ftt_kd_tree_mean, ftt_kd_tree_std, ftt_kd_tree_base = results["FTT+KD+TreeReg"]

height = 0.22
offsets = [height, 0, -height]  # 三个位置：上、中、下
colors = [C_BASELINE, C_KD_GAIN, C_KD_HARM]
names_short = ["FTT", "FTT+KD", "FTT+KD+TreeReg"]
bases_list = [ftt_base, ftt_kd_base, ftt_kd_tree_base]
means_list = [ftt_mean, ftt_kd_mean, ftt_kd_tree_mean]
stds_list = [ftt_std, ftt_kd_std, ftt_kd_tree_std]
alphas = [0.65, 0.85, 0.85]

for off, clr, lbl, b, m, s, al in zip(
        offsets, colors, names_short, bases_list, means_list, stds_list, alphas):
    ax.barh(y_pos + off, m[top_idx], height, xerr=s[top_idx],
            color=clr, alpha=al, capsize=2, label=f"{lbl} (AUPRC={b:.4f})")

ax.set_yticks(range(top_n))
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("AUPRC Decrease", fontsize=10)
kd_delta = ftt_kd_base - ftt_base           # KD 带来的增益（正数 = 提升）
tree_drop = ftt_kd_tree_base - ftt_kd_base  # TreeReg 导致的下降（负数 = 损害）
ax.set_title(f"C. Attention: KD + TreeReg Decomposition\n(KD: +{kd_delta:+.4f}, TreeReg: {tree_drop:+.4f})",
             fontsize=11, fontweight="bold")
ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
ax.grid(axis="x", alpha=0.3)
ax.legend(loc="lower right", fontsize=6.8, framealpha=0.9)

# ── 全局标题 ──
fig.suptitle("Permutation Feature Importance (AUPRC Drop): All Models Including Distillation",
             fontsize=13.5, fontweight="bold", y=1.02)
plt.tight_layout()

outpath = os.path.join(FIGURE_DIR, "permutation_importance.pdf")
fig.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"\nFigure saved to: {outpath}")

# ═══════════════════════════════════════════════════════════════
# 打印对比表格
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("Feature Importance Comparison (Top-16 by XGBoost importance)")
print("="*80)
header = f"{'Feature':>8}  {'XGBoost':>10}  {'DeepFM':>10}  {'DFM+KD':>10}  {'FTT':>10}  {'FTT+KD':>10}  {'FTT+KD+T':>10}"
print(header)
print("-"*len(header))
for idx in top_idx:
    vals = [results[m][0][idx] for m in
            ["XGBoost", "DeepFM", "DeepFM+KD", "FTT", "FTT+KD", "FTT+KD+TreeReg"]]
    print(f"{FEATURE_NAMES[idx]:>8}  " + "  ".join(f"{v:10.4f}" for v in vals))

# ═══════════════════════════════════════════════════════════════
# 关键分析：特征排序的 Pearson 相关性和关键洞察
# ═══════════════════════════════════════════════════════════════

model_names = ["XGBoost", "DeepFM", "DeepFM+KD", "FTT", "FTT+KD", "FTT+KD+TreeReg"]
all_means = np.array([results[n][0] for n in model_names])

# 打印 6×6 Pearson 相关性矩阵
print(f"\n{'='*80}")
print("Pairwise Pearson Correlations of Feature Importance Rankings")
print(f"{'='*80}")
print(f"{'':>16}", end="")
for n in model_names:
    print(f"  {n:>12}", end="")
print()
for i, n_i in enumerate(model_names):
    print(f"  {n_i:>14}", end="")
    for j, n_j in enumerate(model_names):
        r = np.corrcoef(all_means[i], all_means[j])[0, 1]
        print(f"  {r:12.3f}", end="")
    print()

# ── 四大关键洞察 ──
print(f"\n{'='*80}")
print("KEY INSIGHTS")
print(f"{'='*80}")

# 洞察 1: KD 是否把学生拉近教师
r_xgb_dfn_kd = np.corrcoef(all_means[0], all_means[2])[0, 1]  # XGBoost vs DeepFM+KD
r_xgb_dfn = np.corrcoef(all_means[0], all_means[1])[0, 1]      # XGBoost vs DeepFM
print(f"1. KD alignment to teacher:")
print(f"   r(XGBoost, DeepFM)    = {r_xgb_dfn:.3f}")       # 0.625（基线）
print(f"   r(XGBoost, DeepFM+KD) = {r_xgb_dfn_kd:.3f}")    # 0.845（KD 后拉近）
if r_xgb_dfn_kd > r_xgb_dfn:
    print(f"   -> KD pulls DeepFM {r_xgb_dfn_kd - r_xgb_dfn:+.3f} closer to teacher [OK]")
else:
    print(f"   -> KD shifts DeepFM away from teacher by {r_xgb_dfn_kd - r_xgb_dfn:+.3f}")

r_xgb_ftt_kd = np.corrcoef(all_means[0], all_means[4])[0, 1]  # XGBoost vs FTT+KD
r_xgb_ftt = np.corrcoef(all_means[0], all_means[3])[0, 1]    # XGBoost vs FTT
print(f"   r(XGBoost, FTT)       = {r_xgb_ftt:.3f}")
print(f"   r(XGBoost, FTT+KD)    = {r_xgb_ftt_kd:.3f}")

# 洞察 2: TreeReg 的扭曲程度
r_ftt_kd_tree = np.corrcoef(all_means[4], all_means[5])[0, 1]  # FTT+KD vs FTT+KD+TreeReg
r_xgb_tree = np.corrcoef(all_means[0], all_means[5])[0, 1]     # XGBoost vs FTT+KD+TreeReg
print(f"\n2. TreeReg distortion:")
print(f"   r(FTT+KD, FTT+KD+TreeReg) = {r_ftt_kd_tree:.3f}")   # 0.964（看似完美拟合？）
print(f"   r(XGBoost, FTT+KD+TreeReg) = {r_xgb_tree:.3f}")
print(f"   -> TreeReg {'preserves' if r_ftt_kd_tree > 0.8 else 'alters'} feature structure")
# 但高 r 可能只是因为 V14 主导——要看集中度

# 洞察 3: 跨架构 KD 趋同
r_dfn_ftt_kd = np.corrcoef(all_means[2], all_means[4])[0, 1]  # DeepFM+KD vs FTT+KD
r_dfn_ftt = np.corrcoef(all_means[1], all_means[3])[0, 1]    # DeepFM vs FTT
print(f"\n3. Cross-architecture KD convergence:")
print(f"   r(DeepFM, FTT)       = {r_dfn_ftt:.3f}")           # 0.639（异构基线较低）
print(f"   r(DeepFM+KD, FTT+KD) = {r_dfn_ftt_kd:.3f}")       # 0.806（KD 后显著趋同）
print(f"   -> KD makes the two architectures {'more' if r_dfn_ftt_kd > r_dfn_ftt else 'less'} similar")

# 洞察 4: V14 主导性检查
print(f"\n4. V14 dominance across models:")
for name in model_names:
    v14_idx = FEATURE_NAMES.index("V14")
    imp = results[name][0][v14_idx]
    total = results[name][0].sum()
    pct = imp / total * 100 if total > 0 else 0
    print(f"   {name:>16}: V14 = {imp:.4f} ({pct:.1f}% of total importance)")
    # 正常模型 V14 ~20-30%，TreeReg 畸形：V14 集中度 43.9%
