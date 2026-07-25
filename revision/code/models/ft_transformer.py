"""
FT-Transformer：自注意力架构的表格数据模型。
项目核心改进：用 Sparsemax 替代 Softmax 注意力。

问题：PCA 空间 29 个特征的 Query-Key 内积高度同质化
→ Softmax 输出趋近均匀 1/29≈0.0345（标准差仅 0.0004）
→ 注意力塌缩，AUPRC 仅 0.8401
→ Sparsemax 闭式解强制稀疏选择 → AUPRC +4.4pp，ECE ×20

参考：Gorishniy et al. (2021), Martins & Astudillo (2016)
"""
import torch
import torch.nn as nn

from config import FTT_D_MODEL, FTT_N_HEADS, FTT_N_LAYERS, FTT_FFN_DIM, USE_OFFICIAL_BACKEND


# ═══════════════════════════════════════════════════════════════
# Sparsemax 实现（项目的核心算法创新）
# ═══════════════════════════════════════════════════════════════

def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Sparsemax 激活函数（Martins & Astudillo 2016）。
    本质：欧氏投影到概率单纯形 → 可输出精确零概率

    与 Softmax 的关键区别：
    - Softmax：信息论投影（最大化熵），永远不能输出零
    - Sparsemax：欧氏投影（最小化 L2 距离），允许解在边界上

    闭式解算法 O(d log d)：
      1. 降序排序：z_(1) ≥ z_(2) ≥ ... ≥ z_(d)
      2. 找支持集大小 k(z)：满足 1 + k·z_(k) > Σ_{j≤k} z_(j) 的最大 k
      3. 阈值 τ = (Σ_{j∈S} z_(j) − 1) / k
      4. sparsemax_i(z) = max(z_i − τ, 0)
    """
    z = logits
    d = z.size(dim)

    # Step 1: 沿目标维度降序排序
    z_sorted, _ = z.sort(dim=dim, descending=True)

    # Step 2: 构建 k 索引 [1, 2, ..., d]
    k_range = torch.arange(1, d + 1, device=z.device, dtype=z.dtype)
    view_shape = [1] * z_sorted.dim()
    view_shape[dim] = d
    k_range = k_range.view(*view_shape)

    # Step 3: 沿排序维度的累积和
    # cssv[k] = Σ_{j≤k} z_sorted[j]
    cssv = z_sorted.cumsum(dim=dim)

    # Step 4: 计算支持条件
    # 条件：1 + k·z_sorted[k] > cssv[k]
    # 等价于：z_sorted[k] > (cssv[k] − 1) / k
    threshold_vals = (cssv - 1.0) / k_range
    support = (z_sorted > threshold_vals).to(z.dtype)

    # Step 5: 每个元素的支持集大小 k(z)
    k_z = support.sum(dim=dim, keepdim=True)

    # Step 6: 计算阈值 τ
    # τ = (Σ_{j∈S} z_sorted[j] − 1) / k(z)
    # 注意：Σ_{j∈S} 只累加被支持的项
    tau = ((support * z_sorted).sum(dim=dim, keepdim=True) - 1.0) / k_z.clamp(min=1.0)

    # Step 7: 截断：低于 τ 的置零，高于 τ 的保留差值
    output = (z - tau).clamp(min=0.0)
    return output


# ═══════════════════════════════════════════════════════════════
# Alpha-Entmax：Softmax(α=1) 与 Sparsemax(α=2) 之间的连续统
# ═══════════════════════════════════════════════════════════════

def entmax(logits: torch.Tensor, alpha: float = 1.5, dim: int = -1,
           n_iter: int = 30) -> torch.Tensor:
    """
    Alpha-Entmax (Peters et al., 2019; Correia et al., 2019).
    泛化 Softmax(α=1) 和 Sparsemax(α=2)：
      α→1: Softmax（熵最大，无稀疏）
      α=1.5: 中等稀疏
      α=2: Sparsemax（欧氏投影，硬稀疏）

    闭式：p_i ∝ max(0, 1 + (α-1)(z_i - τ))^{1/(α-1)}
    τ 通过二分搜索求解使得 Σp_i = 1。
    """
    z = logits
    d = z.size(dim)

    # 排序便于高效搜索
    z_sorted, _ = z.sort(dim=dim, descending=True)

    # 二分搜索 τ
    tau_min = z_sorted.min(dim=dim, keepdim=True).values - 1.0
    tau_max = z_sorted.max(dim=dim, keepdim=True).values + 1.0

    for _ in range(n_iter):
        tau = (tau_min + tau_max) / 2
        # p_i for sorted z
        inner = 1.0 + (alpha - 1.0) * (z_sorted - tau)
        p_sorted = inner.clamp(min=0.0) ** (1.0 / (alpha - 1.0))
        sum_p = p_sorted.sum(dim=dim, keepdim=True)

        # f(τ) = Σp_i - 1, 寻找零点
        where_greater = (sum_p > 1.0).to(z.dtype)
        # sum_p > 1 → 需要增大 τ
        tau_min = tau_min * where_greater + tau * (1.0 - where_greater)
        tau_max = tau * where_greater + tau_max * (1.0 - where_greater)

    # 最终概率
    inner = 1.0 + (alpha - 1.0) * (z - tau)
    p = inner.clamp(min=0.0) ** (1.0 / (alpha - 1.0))
    return p / p.sum(dim=dim, keepdim=True)


# ═══════════════════════════════════════════════════════════════
# Feature Tokenizer：每个标量特征 → 嵌入向量
# ═══════════════════════════════════════════════════════════════

class FeatureTokenizer(nn.Module):
    """
    将每个标量特征映射到 d_model 维嵌入。
    直觉：x_j 是特征 j 的值，嵌入 = x_j * W_j + b_j
    相当于每个特征有自己独立的线性层。
    29 个特征 → 29 个 token，每个 token 是 d_model 维。
    """

    def __init__(self, input_dim: int, d_model: int = FTT_D_MODEL):
        super().__init__()
        # 权重矩阵 W_j: 每个特征一个 d_model 维向量
        self.W = nn.Parameter(torch.empty(input_dim, d_model))
        nn.init.kaiming_uniform_(self.W, a=5**0.5)
        self.bias = nn.Parameter(torch.zeros(input_dim, d_model))

    def forward(self, x):
        # x: (B, 29) → (B, 29, 64)
        # x_j * W_j: 标量 × 向量 = 向量（外积风格）
        return x.unsqueeze(-1) * self.W.unsqueeze(0) + self.bias.unsqueeze(0)


# ═══════════════════════════════════════════════════════════════
# 多头自注意力（用 Sparsemax 替代 Softmax）
# ═══════════════════════════════════════════════════════════════

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.1,
                 use_sparsemax: bool = True, use_entmax: bool = False,
                 entmax_alpha: float = 1.5):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_sparsemax = use_sparsemax
        self.use_entmax = use_entmax
        self.entmax_alpha = entmax_alpha
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x, return_attention=False):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.head_dim ** -0.5
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale

        if self.use_entmax:
            attn = entmax(attn_logits, alpha=self.entmax_alpha, dim=-1)
        elif self.use_sparsemax:
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


# ═══════════════════════════════════════════════════════════════
# Transformer 编码器块（Pre-Norm 架构）
# ═══════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1,
                 use_sparsemax: bool = True, use_entmax: bool = False,
                 entmax_alpha: float = 1.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout,
                                           use_sparsemax, use_entmax, entmax_alpha)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        # ReGLU FFN（匹配官方 rtdl_revisiting_models）
        self.ffn_linear1 = nn.Linear(d_model, ffn_dim * 2)
        self.ffn_linear2 = nn.Linear(ffn_dim, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + self.dropout1(attn_out)
        else:
            x = x + self.dropout1(self.attn(self.norm1(x)))

        # ReGLU: linear1 outputs 2*ffn_dim, split into gate and value
        h = self.ffn_linear1(self.norm2(x))
        gate, value = h.chunk(2, dim=-1)
        x = x + self.dropout2(self.ffn_linear2(torch.nn.functional.relu(gate) * value))

        if return_attention:
            return x, attn_weights
        return x


# ═══════════════════════════════════════════════════════════════
# 完整 FT-Transformer 模型（手搓实现）
# ═══════════════════════════════════════════════════════════════

class _CustomFTTransformer(nn.Module):
    """
    架构流程：
    输入 x: (B, 29)
    → FeatureTokenizer: 每个特征 j 的 64 维嵌入 (B, 29, 64)
    → [CLS] Token: 拼接在前面 (B, 30, 64)
    → TransformerEncoder × 3 层 (8 头, Pre-Norm, ReGLU FFN)
      注意力支持 Softmax/Sparsemax 切换
    → LayerNorm + Linear(64→1) 分类头
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = FTT_D_MODEL,
        n_heads: int = FTT_N_HEADS,
        n_layers: int = FTT_N_LAYERS,
        ffn_dim: int = FTT_FFN_DIM,
        dropout: float = 0.1,
        use_sparsemax: bool = True,
        use_entmax: bool = False,
        entmax_alpha: float = 1.5,
        feature_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_sparsemax = use_sparsemax
        self.use_entmax = use_entmax
        self._feature_weights = feature_weights

        self.tokenizer = FeatureTokenizer(input_dim, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, dropout,
                           use_sparsemax, use_entmax, entmax_alpha)
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

    def tree_regularization_loss(self) -> torch.Tensor:
        """
        3B 正则化损失。
        L_reg = Σ(1-w_i) × ||E_i||₂
        低重要性特征 (w_i→0) 的嵌入范数被惩罚，趋向零。
        """
        if self._feature_weights is None:
            return torch.tensor(0.0, device=self.head.weight.device)
        w = self._feature_weights.to(self.tokenizer.W.device)
        per_feature_norm = self.tokenizer.W.norm(p=2, dim=1)
        return ((1.0 - w) * per_feature_norm).sum()

    def get_attention_weights(self, x):
        """提取最后一层CLS→特征的注意力权重，返回 (29,) 张量。"""
        self.eval()
        with torch.no_grad():
            tokens = self.tokenizer(x)
            cls = self.cls_token.expand(x.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            for i, block in enumerate(self.blocks):
                if i == self.n_layers - 1:
                    _, attn = block(tokens, return_attention=True)
                    cls_attn = attn[:, :, 0, 1:]  # (B, n_heads, 29)
                    return cls_attn.mean(dim=(0, 1))
                tokens = block(tokens)
        return None


# ═══════════════════════════════════════════════════════════════
# 官方库包装（环境受限时不可用，保留接口兼容）
# ═══════════════════════════════════════════════════════════════

class _OfficialFTTransformer(nn.Module):
    """包装 rtdl_revisiting_models.FTTransformer"""
    def __init__(self, input_dim, d_model=FTT_D_MODEL, n_heads=FTT_N_HEADS,
                 n_layers=FTT_N_LAYERS, ffn_dim=FTT_FFN_DIM, dropout=0.1):
        super().__init__()
        try:
            from rtdl_revisiting_models import FTTransformer as RtdlFTT
        except ImportError as e:
            raise ImportError("pip install rtdl_revisiting_models") from e
        self.model = RtdlFTT(
            n_cont_features=input_dim, cat_cardinalities=[], d_out=1,
            n_blocks=n_layers, d_block=d_model,
            attention_n_heads=n_heads, attention_dropout=dropout,
            ffn_d_hidden=ffn_dim, ffn_d_hidden_multiplier=None,
            ffn_dropout=dropout, residual_dropout=0.0,
        )

    def forward(self, x):
        return self.model(x, None).squeeze(-1)

    def get_attention_weights(self, x):
        """近似提取第一层注意力的 CLS→特征权重（官方库不直接暴露）"""
        self.eval()
        with torch.no_grad():
            block = getattr(self.model, "first_block", None) or self.model.transformer_blocks[0]
            x_token = self.model.cont_embedding(x)
            cls = self.model.cls_token.expand(x.size(0), -1, -1)
            tokens = torch.cat([cls, x_token], dim=1)
            h = block.attention.norm(tokens)
            q, k, v = block.attention.q(h), block.attention.k(h), block.attention.v(h)
            B, N, _ = h.shape
            q = q.reshape(B, N, block.attention.n_heads, block.attention.head_dim).transpose(1, 2)
            k = k.reshape(B, N, block.attention.n_heads, block.attention.head_dim).transpose(1, 2)
            a = torch.softmax(q @ k.transpose(-2, -1) / (block.attention.head_dim ** 0.5), dim=-1)
            return a[:, :, 0, 1:].mean(dim=(0, 1))


class FTTransformer(nn.Module):
    """统一的 FT-Transformer 入口。支持 Softmax/Sparsemax/Entmax 切换。"""
    def __init__(self, input_dim, d_model=FTT_D_MODEL, n_heads=FTT_N_HEADS,
                 n_layers=FTT_N_LAYERS, ffn_dim=FTT_FFN_DIM, dropout=0.1,
                 use_sparsemax=True, use_entmax=False, entmax_alpha=1.5,
                 use_official=False, feature_weights=None):
        super().__init__()
        if use_official:
            self.impl = _OfficialFTTransformer(input_dim, d_model, n_heads, n_layers, ffn_dim, dropout)
        else:
            self.impl = _CustomFTTransformer(input_dim, d_model, n_heads, n_layers, ffn_dim,
                                             dropout, use_sparsemax, use_entmax, entmax_alpha, feature_weights)

    def forward(self, x):
        return self.impl(x)

    def get_attention_weights(self, x):
        return self.impl.get_attention_weights(x)

    def tree_regularization_loss(self):
        if hasattr(self.impl, 'tree_regularization_loss'):
            return self.impl.tree_regularization_loss()
        return torch.tensor(0.0)
