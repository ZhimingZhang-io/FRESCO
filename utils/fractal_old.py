import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

class ECGFractalBlock(nn.Module):
    """优化的分形处理块"""
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 seq_len: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_flash_attention: bool = False):
        super().__init__()
        
        # 确保输出维度能被num_heads整除
        assert out_channels % num_heads == 0, f"out_channels {out_channels} must be divisible by num_heads {num_heads}"
        
        # 改进的多尺度卷积 - 使用深度可分离卷积减少参数
        self.multi_scale_conv = nn.ModuleList()
        kernels = [3, 7, 15, 31]
        
        for k in kernels:
            conv_block = nn.Sequential(
                # 深度卷积
                nn.Conv1d(in_channels, in_channels, kernel_size=k, padding=k//2, groups=in_channels),
                # 点卷积
                nn.Conv1d(in_channels, out_channels//4, kernel_size=1),
                nn.BatchNorm1d(out_channels//4),
                nn.GELU()
            )
            self.multi_scale_conv.append(conv_block)
        
        # 特征融合卷积
        self.feature_fusion = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.fusion_norm = nn.BatchNorm1d(out_channels)
        
        # 优化的自注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 改进的层归一化和前馈网络
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        # 使用更高效的前馈网络
        hidden_dim = out_channels * 2
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_channels),
            nn.Dropout(dropout)
        )
        
        # 改进的下采样 - 使用可学习的池化
        self.downsample = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )
        
        # 位置编码
        self.pos_encoding = self._build_pos_encoding(seq_len, out_channels)
        
    def _build_pos_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """构建正弦位置编码"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, seq_len, d_model]
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: [B, C, L]
        B, C, L = x.shape
        
        # 1. 多尺度卷积特征提取
        multi_scale_features = []
        for conv_block in self.multi_scale_conv:
            feat = conv_block(x)
            multi_scale_features.append(feat)
        
        # 拼接多尺度特征
        x = torch.cat(multi_scale_features, dim=1)  # [B, out_channels, L]
        
        # 特征融合
        x = self.feature_fusion(x)
        x = self.fusion_norm(x)
        x = F.gelu(x)
        
        # 2. 自注意力计算
        x_seq = x.transpose(1, 2)  # [B, L, out_channels]
        
        # 添加位置编码
        if hasattr(self, 'pos_encoding') and self.pos_encoding.size(1) == L:
            pe = self.pos_encoding[:, :L, :].to(x_seq.device)
            x_seq = x_seq + pe
        
        # 自注意力
        attn_out, attn_weights = self.attention(x_seq, x_seq, x_seq)
        x_seq = self.norm1(x_seq + attn_out)
        
        # 前馈网络
        ffn_out = self.ffn(x_seq)
        x_seq = self.norm2(x_seq + ffn_out)
        
        # 3. 转换回卷积格式
        x = x_seq.transpose(1, 2)  # [B, out_channels, L]
        
        # 4. 下采样
        x_down = self.downsample(x)
        
        return x, x_down, attn_weights


class FractalECGEncoder(nn.Module):
    """优化的分形ECG编码器"""
    def __init__(self,
                 num_leads: int = 12,
                 seq_len: int = 5000,
                 embed_dims: List[int] = [64, 128, 256, 512],
                 num_heads_list: List[int] = [4, 8, 8, 16],
                 fractal_levels: int = 4,
                 dropout: float = 0.1,
                 use_gradient_checkpointing: bool = False):
        super().__init__()
        
        self.num_leads = num_leads
        self.fractal_levels = fractal_levels
        self.embed_dims = embed_dims
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # 改进的输入投影 - 使用更大的kernel捕获更多上下文
        self.input_proj = nn.Sequential(
            nn.Conv1d(num_leads, embed_dims[0], kernel_size=15, padding=7),
            nn.BatchNorm1d(embed_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 分形层级
        self.fractal_blocks = nn.ModuleList()
        current_seq_len = seq_len
        
        for i in range(fractal_levels):
            if i == 0:
                in_dim = embed_dims[0]
            else:
                in_dim = embed_dims[i-1]
            
            out_dim = embed_dims[i]
            
            block = ECGFractalBlock(
                in_channels=in_dim,
                out_channels=out_dim,
                seq_len=current_seq_len,
                num_heads=num_heads_list[i],
                dropout=dropout
            )
            self.fractal_blocks.append(block)
            current_seq_len = current_seq_len // 2
        
        # # 改进的全局池化 - 使用注意力池化
        # self.attention_pool = nn.MultiheadAttention(
        #     embed_dim=embed_dims[-1],
        #     num_heads=8,
        #     dropout=dropout,
        #     batch_first=True
        # )
        # self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dims[-1]))
        
        # 改进的多尺度特征融合 - 使用交叉注意力
        self.output_dim = embed_dims[-1]
        
        # 尺度感知的特征投影
        self.scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dims[i], self.output_dim),
                nn.LayerNorm(self.output_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for i in range(fractal_levels)
        ])
        
        # 多尺度交叉注意力融合
        self.cross_scale_attention = nn.MultiheadAttention(
            embed_dim=self.output_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 可学习的尺度权重和温度参数
        self.scale_weights = nn.Parameter(torch.ones(fractal_levels) / fractal_levels)
        self.temperature = nn.Parameter(torch.ones(1))
        
        # 最终输出层
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, self.output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.output_dim, self.output_dim)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
                
    def _attention_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """使用注意力机制进行全局池化"""
        B, C, L = x.shape
        x_seq = x.transpose(1, 2)  # [B, L, C]
        
        # 使用可学习的query进行attention pooling
        query = self.pool_query.expand(B, -1, -1)  # [B, 1, C]
        
        pooled, _ = self.attention_pool(query, x_seq, x_seq)
        return pooled.squeeze(1)  # [B, C]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, num_leads, seq_len]
        B, C, L = x.shape
        
        # 输入投影
        x = self.input_proj(x)  # [B, embed_dims[0], seq_len]
        
        # 收集多尺度特征
        multi_scale_features = []
        current_x = x
        
        # 逐层处理
        for i, block in enumerate(self.fractal_blocks):
            if self.use_gradient_checkpointing and self.training:
                # 使用梯度检查点节省内存
                current_features, next_x, attn_weights = torch.utils.checkpoint.checkpoint(
                    block, current_x, use_reentrant=False
                )
            else:
                current_features, next_x, attn_weights = block(current_x)
            
            
            pooled_features = F.adaptive_avg_pool1d(current_features, 1).squeeze(-1)
            multi_scale_features.append(pooled_features)
            
            current_x = next_x
        
        # 改进的多尺度特征融合
        projected_features = []
        for i, (features, projection) in enumerate(zip(multi_scale_features, self.scale_projections)):
            projected = projection(features)
            projected_features.append(projected)
        
        # 堆叠特征进行交叉注意力
        feature_stack = torch.stack(projected_features, dim=1)  # [B, num_scales, output_dim]
        
        # 交叉尺度注意力
        attended_features, scale_attention = self.cross_scale_attention(
            feature_stack, feature_stack, feature_stack
        )
        
        # 温度缩放的加权融合
        scale_weights = F.softmax(self.scale_weights / self.temperature, dim=0)
        weighted_features = attended_features * scale_weights.unsqueeze(0).unsqueeze(-1)
        
        # 最终特征融合
        final_features = weighted_features.sum(dim=1)  # [B, output_dim]
        
        # 输出投影
        output = self.output_projection(final_features)
        
        return output
    
    # def get_multi_scale_features(self, x: torch.Tensor) -> List[dict]:
    #     """返回详细的多尺度特征信息"""
    #     B, C, L = x.shape
    #     x = self.input_proj(x)
        
    #     multi_scale_info = []
    #     current_x = x
        
    #     for i, block in enumerate(self.fractal_blocks):
    #         current_features, next_x, attn_weights = block(current_x)
    #         pooled_features = self._attention_pooling(current_features)
            
    #         multi_scale_info.append({
    #             'level': i,
    #             'seq_len': current_features.shape[-1],
    #             'features': pooled_features.detach(),
    #             'attention_weights': attn_weights.detach(),
    #             'feature_dim': current_features.shape[1],
    #             'scale_importance': self.scale_weights[i].item()
    #         })
            
    #         current_x = next_x
            
    #     return multi_scale_info

    # def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
    #     """提取所有层的注意力图"""
    #     attention_maps = []
    #     current_x = self.input_proj(x)
        
    #     for block in self.fractal_blocks:
    #         _, current_x, attn_weights = block(current_x)
    #         attention_maps.append(attn_weights.detach())
            
    #     return attention_maps


