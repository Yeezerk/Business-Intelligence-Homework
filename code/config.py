"""
全局配置：本项目所有路径、超参、模型维度的集中管理。
设计哲学：单例式——所有脚本统一引用，改一处全局生效。
"""
import os
import torch

# ═══════════════════════════════════════════════════════════════
# 路径配置：所有输出集中到 outputs/ 目录
# ═══════════════════════════════════════════════════════════════
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录（code/ 的父目录）
DATA_DIR = os.path.join(ROOT_DIR, "outputs")                          # 所有产出的统一根目录
RAW_DATA = os.path.join(DATA_DIR, "raw", "credit_card_fraud.csv")    # 原始数据集（Kaggle 2023）
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")                   # 预处理后的 .pt 张量 + scaler
MODEL_DIR = os.path.join(DATA_DIR, "models")                          # 训练好的模型权重
FIGURE_DIR = os.path.join(DATA_DIR, "figures")                        # EDA/评估/可解释性图表
RESULT_DIR = os.path.join(DATA_DIR, "results")                        # CSV 指标表 + LaTeX 表

# 自动创建目录（幂等，重复运行不会报错）
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 可复现性：固定随机种子 + 关闭 cuDNN 不确定性算法
# ═══════════════════════════════════════════════════════════════
SEED = 42                                                            # 所有随机操作的全局种子
torch.manual_seed(SEED)                                              # PyTorch CPU 随机数
torch.cuda.manual_seed_all(SEED)                                     # PyTorch GPU 随机数
torch.backends.cudnn.deterministic = True                           # cuDNN 确定算法（慢但可复现）
torch.backends.cudnn.benchmark = False                              # 不自动搜索最优算法（确定性优先）

# ═══════════════════════════════════════════════════════════════
# 数据划分比例（分层抽样保持欺诈率一致）
# ═══════════════════════════════════════════════════════════════
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ═══════════════════════════════════════════════════════════════
# 训练默认超参
# ═══════════════════════════════════════════════════════════════
BATCH_SIZE = 512                                                     # 批次大小（GPU 显存允许范围内尽量大）
MAX_EPOCHS = 100                                                     # 最大训练轮数（早停通常在前 20-40 轮触发）
EARLY_STOP_PATIENCE = 15                                             # 验证集 AUPRC 连续 15 轮不提高即停
LEARNING_RATE = 1e-3                                                 # AdamW 默认学习率
WEIGHT_DECAY = 1e-5                                                  # L2 正则化强度（防过拟合）

# ═══════════════════════════════════════════════════════════════
# Focal Loss 默认参数
# ═══════════════════════════════════════════════════════════════
FOCAL_ALPHA = 0.25                                                   # FTT 用 α=0.25（Sparsemax 已有特征选择，需均衡梯度）
FOCAL_GAMMA = 2.0                                                    # γ=2.0 是 Focal Loss 原论文推荐值

# ═══════════════════════════════════════════════════════════════
# 各模型架构维度的默认值（在 05_train_*.py 中可按需覆盖）
# ═══════════════════════════════════════════════════════════════
MLP_HIDDEN_DIMS = [256, 128, 64, 32]                                 # MLP 隐层宽度逐层递减
MLP_DROPOUT = 0.3

TABNET_N_D = 32                                                      # TabNet 决策步输出维度
TABNET_N_A = 32                                                      # TabNet 注意力步输出维度
TABNET_N_STEPS = 4                                                   # 决策步数（每个步选不同特征子集）
TABNET_GAMMA = 1.3                                                   # 注意力先验松弛因子

FTT_D_MODEL = 64                                                     # FTT 嵌入维度（每个特征映射到 64 维）
FTT_N_HEADS = 8                                                      # 注意力头数
FTT_N_LAYERS = 3                                                     # Transformer 编码器层数（3 层足够，过深会过拟合）
FTT_FFN_DIM = 256                                                    # FFN 中间层宽度

DEEPFM_EMBED_DIM = 16                                                # DeepFM 嵌入维度（比 FTT 小 → FM 内积计算更轻量）
DEEPFM_HIDDEN_DIMS = [256, 128, 64]                                  # Deep 组件 MLP 宽度逐层递减
DEEPFM_DROPOUT = 0.3

# ═══════════════════════════════════════════════════════════════
# 设备与特征
# ═══════════════════════════════════════════════════════════════
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"             # 自动检测 GPU

FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Amount"]        # 29 维特征：V1-V28（PCA 匿名化）+ Amount
INPUT_DIM = len(FEATURE_NAMES)                                       # 输入维度 = 29

# ═══════════════════════════════════════════════════════════════
# 后端切换：官方库 vs 手搓实现
# ═══════════════════════════════════════════════════════════════
# True = 调用官方库（pytorch-tabnet / rtdl_revisiting_models / deepctr_torch）
# False = 项目自带的手搓实现（默认，因为 rtdl 在本环境不可用）
USE_OFFICIAL_BACKEND = False

# ═══════════════════════════════════════════════════════════════
# 树监督蒸馏配置（方案 3A + 3B）—— 本项目核心创新
# ═══════════════════════════════════════════════════════════════
USE_DISTILLATION = True                # 是否启用知识蒸馏
TEMPERATURE = 4.0                      # 蒸馏温度 T：T 越大软标签越平滑，暴露类别间暗知识
KD_ALPHA = 0.7                         # 硬标签权重：L = α·FL(硬) + (1-α)·T²·KL(软)
                                       # α=0.7 表示 70% 硬标签信号 + 30% 蒸馏信号
TREE_REG_LAMBDA = 0.1                  # TreeReg 正则化强度（方案 3B 的 λ）
                                       # λ=0.1 已造成 -3.73pp 反效果，增大只会更糟
FEATURE_WEIGHTS_PATH = os.path.join(   # XGBoost split gain 缓存路径
    PROCESSED_DIR, "xgb_feature_weights.pt"
)
