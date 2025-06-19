from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fairseq_signals import utils
from fairseq_signals.distributed import utils as dist_utils
from fairseq_signals.modules import (
    ESPNETMultiHeadedAttention,
    LayerNorm,
    MultiheadAttention,
    RelPositionMultiHeadedAttention,
    RotaryPositionMultiHeadedAttention,
)
from fairseq_signals.modules.moe import MOELayer, Top1Gate, Top2Gate
from fairseq_signals.utils1 import get_activation_fn
print(MultiHeadAttention)
class ConformerAttentionBlock(nn.Module):
    """Modified from Conformer's attention module with multi-head attention"""
    
    def __init__(self, embed_dim, heads, dropout, pos_enc_type):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, mask, pos_emb):
        x = self.norm(x)
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        return self.dropout(attn_out)

class ConformerConvBlock(nn.Module):
    """Depthwise separable convolution from Conformer"""
    
    def __init__(self, embed_dim, kernel_size, dropout, activation):
        super().__init__()
        self.depthwise = nn.Conv1d(
            embed_dim, embed_dim, 
            kernel_size, 
            padding=kernel_size//2,
            groups=embed_dim
        )
        self.pointwise = nn.Conv1d(embed_dim, embed_dim, 1)
        self.activation = activation
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = self.norm(x)
        # 维度转换 (T,B,C) -> (B,C,T)
        x = x.permute(1, 2, 0)
        x = self.depthwise(x)
        x = self.activation(x)
        x = self.pointwise(x)
        # 恢复维度 (B,C,T) -> (T,B,C)
        return x.permute(2, 0, 1)

class BranchformerEncoderLayer(nn.Module):
    """Branchformer block with multi-branch architecture"""

    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
        dropout: float = 0.1,
        depthwise_conv_kernel_size: int = 31,
        activation_fn: str = "swish",
        pos_enc_type: str = "abs"
    ):
        super().__init__()
        
        # 分支1：Transformer分支（基于Conformer的注意力模块）
        self.transformer_branch = ConformerAttentionBlock(
            embed_dim,
            attention_heads,
            dropout,
            pos_enc_type
        )
        
        # 分支2：卷积分支（基于Conformer的卷积模块）
        self.conv_branch = ConformerConvBlock(
            embed_dim,
            depthwise_conv_kernel_size,
            dropout,
            activation_fn
        )
        
        # 动态门控融合模块
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # 前馈网络（基于Conformer的FFN模块）
        self.ffn = ConformerFFNBlock(
            embed_dim,
            ffn_embed_dim,
            dropout,
            activation_fn
        )
        
        # 归一化层
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,
        encoder_padding_mask: Optional[Tensor] = None,
        position_emb: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            x: 输入序列 (T, B, C)
            encoder_padding_mask: 填充掩码 (B, T)
            position_emb: 位置编码
        """
        # 残差连接初始值
        residual = x
        
        # 并行分支处理
        attn_out = self.transformer_branch(x, encoder_padding_mask, position_emb)
        conv_out = self.conv_branch(x)
        
        # 动态门控融合
        combined = torch.cat([attn_out, conv_out], dim=-1)
        gate_weights = self.gate(combined)
        fused = gate_weights[..., 0:1] * attn_out + gate_weights[..., 1:2] * conv_out
        
        # 前馈网络
        ffn_out = self.ffn(fused)
        
        # 残差连接和归一化
        x = self.norm(residual + ffn_out)
        
        return x

layer = BranchformerEncoderLayer(
    embed_dim=512,
    ffn_embed_dim=2048,
    attention_heads=8,
    depthwise_conv_kernel_size=31
)

# 输入序列 (T=100, B=16, C=512)
x = torch.randn(100, 16, 512)
output = layer(x)