import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


# ==================== 基础模块 ====================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).to(dtype) * self.weight


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = attn_drop

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0
        )
        
        x = x.transpose(1, 2).reshape(B, L, D)
        return self.proj(x)


class AttentionPooling(nn.Module):
    """可学习的注意力池化"""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.scale = dim ** -0.5
        nn.init.trunc_normal_(self.query, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)
        attn = (q @ x.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ x
        return out.squeeze(1)


class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, dim, mult=4, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim * mult, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x):
        return self.net(x)


class MultiScaleFusion(nn.Module):
    def __init__(self, num_levels, output_dim, num_heads=8, drop=0.1):
        super().__init__()
        self.num_levels = num_levels
        self.output_dim = output_dim
        self.level_pools = nn.ModuleList([
            AttentionPooling(output_dim) for _ in range(num_levels)
        ])
        self.level_norms = nn.ModuleList([
            RMSNorm(output_dim) for _ in range(num_levels)
        ])
        
        self.fusion_attn = SelfAttention(output_dim, num_heads, drop)
        self.fusion_norm = RMSNorm(output_dim)
        self.final_pool = AttentionPooling(output_dim)

    def forward(self, all_features):
        level_feats = []
        for feat, pool, norm in zip(all_features, self.level_pools, self.level_norms):
            pooled = pool(feat)
            normed = norm(pooled)
            level_feats.append(normed)
        
        stacked = torch.stack(level_feats, dim=1)
        stacked = stacked + self.fusion_attn(self.fusion_norm(stacked))
        return self.final_pool(stacked)


# ==================== 核心 Attention 模块 ====================

class IntraAttention(nn.Module):
    """组内 Attention"""
    def __init__(self, dim, num_heads=8, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm = RMSNorm(dim)
    
    def forward(self, x):
        B, G, N, D = x.shape
        x_flat = x.reshape(B * G, N, D)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x_flat = self.norm(x_flat + attn_out)
        return x_flat.reshape(B, G, N, D)


class InterAttention(nn.Module):
    """组间 Attention"""
    def __init__(self, dim, num_heads=8, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm = RMSNorm(dim)
    
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class GuidedCrossAttention(nn.Module):
    """指导 Cross-Attention"""
    def __init__(self, dim, num_heads=8, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm = RMSNorm(dim)
    
    def forward(self, x, guidance):
        B, G, N, D = x.shape
        x_flat = x.reshape(B * G, N, D)
        guide_flat = guidance.reshape(B * G, 1, D)
        attn_out, _ = self.attn(query=x_flat, key=guide_flat, value=guide_flat)
        x_flat = self.norm(x_flat + attn_out)
        return x_flat.reshape(B, G, N, D)


# ==================== MAE 相关模块 ====================

class MaskedScaleBlock(nn.Module):
    """
    支持 MAE 掩码的尺度处理块
    新增:  掩码生成、mask token、位置恢复
    """
    def __init__(
        self, 
        input_dim, 
        hidden_dim, 
        num_patches, 
        num_heads=8, 
        depth=2, 
        drop=0.1, 
        use_guidance=False
    ):
        super().__init__()
        self.use_guidance = use_guidance
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            RMSNorm(hidden_dim),
            nn.GELU(),
        )
        
        # 位置编码
        self.group_pos = nn.Parameter(torch.randn(1, 12, 1, hidden_dim) * 0.02)
        self.patch_pos = nn.Parameter(torch.randn(1, 1, num_patches, hidden_dim) * 0.02)
        
        # MAE:  可学习的 mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # 多层处理
        self.layers = nn.ModuleList()
        for _ in range(depth):
            layer = nn.ModuleDict({
                'intra_attn': IntraAttention(hidden_dim, num_heads, drop),
                'intra_ffn': FeedForward(hidden_dim, drop=drop),
                'intra_ffn_norm': RMSNorm(hidden_dim),
                'pool': AttentionPooling(hidden_dim),
                'inter_attn': InterAttention(hidden_dim, num_heads, drop),
                'inter_ffn': FeedForward(hidden_dim, drop=drop),
                'inter_ffn_norm': RMSNorm(hidden_dim),
            })
            if use_guidance: 
                layer['guide_attn'] = GuidedCrossAttention(hidden_dim, num_heads, drop)
            self.layers.append(layer)
        
        self.output_norm = RMSNorm(hidden_dim)
    
    def random_masking(self, x:  torch.Tensor, mask_ratio:  float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        """
        对 patches 进行随机掩码
        
        Args:
            x: [B, G, N, D] 输入特征
            mask_ratio: 掩码比例
            
        Returns:
            x_masked: [B, G, N_visible, D] 只保留可见的 patches
            mask:  [B, G, N] bool mask, True 表示被掩码
            ids_restore: [B, G, N] 用于恢复原始顺序的索引
        """
        B, G, N, D = x.shape
        num_mask = int(N * mask_ratio)
        num_visible = N - num_mask
        
        # 为每个 group 生成随机排列
        noise = torch.rand(B, G, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=-1)
        ids_restore = torch.argsort(ids_shuffle, dim=-1)
        
        # 生成 mask:  前 num_visible 个是可见的
        mask = torch.zeros(B, G, N, device=x.device, dtype=torch.bool)
        mask_indices = ids_shuffle[:, : , num_visible:]  # 被掩码的索引
        mask.scatter_(2, mask_indices, True)
        
        # 提取可见的 patches
        ids_keep = ids_shuffle[:, :, : num_visible]
        x_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(-1).expand(-1, -1, -1, D))
        
        return x_masked, mask, ids_restore
    
    def restore_with_mask_tokens(
        self, 
        x_encoded: torch.Tensor, 
        ids_restore: torch.Tensor, 
        num_patches: int
    ) -> torch.Tensor:
        """
        将编码后的可见 patches 与 mask tokens 合并，恢复原始顺序
        
        Args:
            x_encoded: [B, G, N_visible, D] 编码后的可见 patches
            ids_restore: [B, G, N] 恢复索引
            num_patches: 原始 patch 数量
            
        Returns: 
            x_full: [B, G, N, D] 完整序列（可见 + mask tokens）
        """
        B, G, N_visible, D = x_encoded.shape
        num_mask = num_patches - N_visible
        
        # 扩展 mask token
        mask_tokens = self.mask_token.expand(B, G, num_mask, -1)
        
        # 拼接
        x_full = torch.cat([x_encoded, mask_tokens], dim=2)  # [B, G, N, D]
        
        # 恢复原始顺序
        x_full = torch.gather(x_full, dim=2, index=ids_restore.unsqueeze(-1).expand(-1, -1, -1, D))
        
        return x_full
    
    def forward(
        self, 
        x:  torch.Tensor, 
        guidance:  Optional[torch.Tensor] = None,
        mask_ratio:  float = 0.0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args: 
            x: [B, 12, num_patches, input_dim]
            guidance: [B, 12, hidden_dim] (可选)
            mask_ratio: 掩码比例 (0.0 = 不掩码)
            
        Returns:
            output: [B, 12, hidden_dim]
            mask: [B, 12, num_patches] 或 None
            ids_restore: [B, 12, num_patches] 或 None
        """
        B, G, N, _ = x.shape
        
        # 投影
        x = self.input_proj(x)
        
        # 添加位置编码（在掩码之前，这样 mask token 也能获得正确的位置信息）
        x = x + self.group_pos + self.patch_pos
        
        # MAE 掩码
        mask = None
        ids_restore = None
        if mask_ratio > 0 and self.training:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        for layer in self.layers:
            if self.use_guidance and guidance is not None: 
                x = layer['guide_attn'](x, guidance)
            
            # 组内 Attention
            x = layer['intra_attn'](x)
            x = x + layer['intra_ffn'](layer['intra_ffn_norm'](x))
            
            # 池化
            current_N = x.shape[2]
            x_flat = x.reshape(B * G, current_N, -1)
            x_group = layer['pool'](x_flat)
            x_group = x_group.reshape(B, G, -1)
            
            # 组间 Attention
            x_group = layer['inter_attn'](x_group)
            x_group = x_group + layer['inter_ffn'](layer['inter_ffn_norm'](x_group))
            
            guidance = x_group
        
        return self.output_norm(x_group), mask, ids_restore


class MAEDecoder(nn.Module):
    """
    MAE 解码器：用于重建被掩码的 patches
    轻量级设计，每个尺度共享解码器结构但参数独立
    """
    def __init__(
        self, 
        hidden_dim: int, 
        output_dim: int,  # patch_dim，用于重建原始信号
        num_patches: int,
        num_heads: int = 4,
        depth:  int = 2,
        drop:  float = 0.1
    ):
        super().__init__()
        self.num_patches = num_patches
        self.output_dim = output_dim
        
        # 从 encoder hidden_dim 映射到 decoder dim
        decoder_dim = hidden_dim // 2
        self.decoder_embed = nn.Linear(hidden_dim, decoder_dim)
        
        # Mask token for decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, 1, num_patches, decoder_dim) * 0.02)
        self.group_embed = nn.Parameter(torch.randn(1, 12, 1, decoder_dim) * 0.02)
        
        # Decoder layers (轻量级)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': IntraAttention(decoder_dim, num_heads, drop),
                'ffn':  FeedForward(decoder_dim, mult=2, drop=drop),
                'ffn_norm': RMSNorm(decoder_dim),
            })
            for _ in range(depth)
        ])
        
        self.norm = RMSNorm(decoder_dim)
        
        # 重建头
        self.pred = nn.Linear(decoder_dim, output_dim)
    
    def forward(
        self, 
        x_encoded: torch.Tensor,  # [B, G, N_visible, hidden_dim] 编码器输出
        ids_restore: torch.Tensor,  # [B, G, N]
    ) -> torch.Tensor:
        """
        Args:
            x_encoded: 编码后的可见 patches
            ids_restore: 恢复原始顺序的索引
            
        Returns:
            pred: [B, G, N, output_dim] 重建的所有 patches
        """
        B, G, N_visible, _ = x_encoded.shape
        N = self.num_patches
        num_mask = N - N_visible
        
        # 映射到 decoder 维度
        x = self.decoder_embed(x_encoded)
        D = x.shape[-1]
        
        # 添加 mask tokens
        mask_tokens = self.mask_token.expand(B, G, num_mask, -1)
        x_full = torch.cat([x, mask_tokens], dim=2)
        
        # 恢复原始顺序
        x_full = torch.gather(x_full, dim=2, index=ids_restore.unsqueeze(-1).expand(-1, -1, -1, D))
        
        # 添加位置编码
        x_full = x_full + self.pos_embed + self.group_embed
        
        # Decoder layers
        for layer in self.layers:
            x_full = layer['attn'](x_full)
            x_full = x_full + layer['ffn'](layer['ffn_norm'](x_full))
        
        x_full = self.norm(x_full)
        
        # 预测
        pred = self.pred(x_full)  # [B, G, N, output_dim]
        
        return pred


class MultiScaleMAEDecoder(nn.Module):
    """多尺度 MAE 解码器"""
    def __init__(self, hidden_dim:  int, scale_configs: List[Tuple[int, int]], num_heads=4, depth=2, drop=0.1):
        super().__init__()
        self.decoders = nn.ModuleList([
            MAEDecoder(
                hidden_dim=hidden_dim,
                output_dim=patch_dim,
                num_patches=num_patches,
                num_heads=num_heads,
                depth=depth,
                drop=drop
            )
            for num_patches, patch_dim in scale_configs
        ])
    
    def forward(
        self, 
        encoded_features: List[torch.Tensor],  # 每个尺度的编码特征
        ids_restores: List[torch.Tensor],  # 每个尺度的恢复索引
    ) -> List[torch.Tensor]: 
        """返回每个尺度的重建结果"""
        preds = []
        for decoder, feat, ids_restore in zip(self.decoders, encoded_features, ids_restores):
            if ids_restore is not None:
                pred = decoder(feat, ids_restore)
                preds.append(pred)
            else:
                preds.append(None)
        return preds


# ==================== 主模型 ====================

class FractalEcgMAE(nn.Module):
    """
    带 MAE 预训练的 Fractal ECG Encoder
    
    训练模式: 
    - 'pretrain': MAE 预训练，返回重建损失
    - 'finetune':  微调模式，正常分类
    - 'linear_probe': 线性探测，冻结 backbone
    - 'features': 只返回特征
    """
    def __init__(
        self, 
        hidden_dim: int = 512, 
        num_heads: int = 8, 
        depth: int = 2, 
        drop:  float = 0.1,
        num_classes: Optional[int] = None,
        # MAE 参数
        mask_ratios: Optional[List[float]] = None,  # 每个尺度的掩码比例
        decoder_depth: int = 2,
        decoder_num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 默认掩码比例：细粒度尺度掩码更多
        self.mask_ratios = mask_ratios or [0.0, 0.5, 0.6, 0.75]  # 尺度0不掩码（只有1个patch）
        
        # 尺度配置
        self.scale_configs = [
            (1, 5000),    # 1 patch × 5000 dim
            (10, 500),    # 10 patches × 500 dim
            (100, 50),    # 100 patches × 50 dim
            (500, 10),    # 500 patches × 10 dim
        ]
        self.num_scales = len(self.scale_configs)
        
        # ========== Encoder ==========
        # 尺度0: 无指导，无掩码
        self.scale0 = MaskedScaleBlock(
            input_dim=self.scale_configs[0][1],
            hidden_dim=hidden_dim,
            num_patches=self.scale_configs[0][0],
            num_heads=num_heads,
            depth=depth,
            drop=drop,
            use_guidance=False
        )
        
        # 尺度1, 2, 3: 带指导
        self.guided_scales = nn.ModuleList([
            MaskedScaleBlock(
                input_dim=cfg[1],
                hidden_dim=hidden_dim,
                num_patches=cfg[0],
                num_heads=num_heads,
                depth=depth,
                drop=drop,
                use_guidance=True
            )
            for cfg in self.scale_configs[1:]
        ])
        
        # 尺度融合
        self.scale_fusion = MultiScaleFusion(
            num_levels=self.num_scales, 
            output_dim=hidden_dim
        )
        
        # ========== MAE Decoder ==========
        self.mae_decoder = MultiScaleMAEDecoder(
            hidden_dim=hidden_dim,
            scale_configs=self.scale_configs,
            num_heads=decoder_num_heads,
            depth=decoder_depth,
            drop=drop
        )
        
        # ========== 分类头 ==========
        if num_classes is not None:
            self.linear_probe = nn.Linear(hidden_dim, num_classes)
    
    def freeze_backbone(self):
        """冻结主干网络，用于 Linear Probing"""
        for name, param in self.named_parameters():
            if 'linear_probe' not in name:
                param.requires_grad = False
        print("Backbone frozen. Only classifier is trainable.")
    
    def unfreeze_backbone(self):
        """解冻主干网络"""
        for param in self.parameters():
            param.requires_grad = True
        print("All parameters are trainable.")
    
    def freeze_decoder(self):
        """冻结 MAE 解码器（微调时使用）"""
        for param in self.mae_decoder.parameters():
            param.requires_grad = False
        print("MAE decoder frozen.")
    
    def encode(
        self, 
        x: torch.Tensor, 
        mask_ratios: Optional[List[float]] = None
    ) -> Tuple[torch.Tensor, List, List, List]:
        """
        编码器前向传播
        
        Returns:
            features: [B, hidden_dim] 融合后的特征
            all_scale_outputs: 每个尺度的输出
            all_masks: 每个尺度的 mask
            all_ids_restore:  每个尺度的恢复索引
        """
        B = x.shape[0]
        mask_ratios = mask_ratios or self.mask_ratios
        
        all_scale_outputs = []
        all_masks = []
        all_ids_restore = []
        all_encoded_visible = []  # 用于 decoder
        
        # 尺度0
        x0 = x.reshape(B, 12, 1, 5000)
        out0, mask0, ids0 = self.scale0(x0, mask_ratio=mask_ratios[0])
        all_scale_outputs.append(out0)
        all_masks.append(mask0)
        all_ids_restore.append(ids0)
        # 尺度0 encoded visible (需要从 scale block 获取中间结果)
        all_encoded_visible.append(out0.unsqueeze(2))  # [B, G, 1, D]
        
        prev_guidance = out0
        
        # 尺度1, 2, 3
        for i, guided_scale in enumerate(self.guided_scales):
            num_patches, patch_dim = self.scale_configs[i + 1]
            xi = x.reshape(B, 12, num_patches, patch_dim)
            outi, maski, idsi = guided_scale(xi, prev_guidance, mask_ratio=mask_ratios[i + 1])
            all_scale_outputs.append(outi)
            all_masks.append(maski)
            all_ids_restore.append(idsi)
            prev_guidance = outi
        
        # 融合
        features = self.scale_fusion(all_scale_outputs)
        
        return features, all_scale_outputs, all_masks, all_ids_restore
    
    def compute_mae_loss(
        self, 
        x: torch.Tensor,
        preds: List[torch.Tensor],
        masks: List[torch.Tensor],
        normalize_target: bool = True
    ) -> torch.Tensor:
        """
        计算 MAE 重建损失
        
        只计算被掩码位置的重建损失
        """
        B = x.shape[0]
        total_loss = 0.0
        num_valid_scales = 0
        
        for scale_idx, (pred, mask) in enumerate(zip(preds, masks)):
            if pred is None or mask is None: 
                continue
            
            num_patches, patch_dim = self.scale_configs[scale_idx]
            
            # 获取原始 target
            target = x.reshape(B, 12, num_patches, patch_dim)
            
            # 可选：归一化 target（patch-wise normalization）
            if normalize_target: 
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1e-6).sqrt()
            
            # 计算 MSE loss，只在 masked 位置
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [B, G, N]
            
            # 只计算 masked 位置的损失
            mask_float = mask.float()
            loss = (loss * mask_float).sum() / (mask_float.sum() + 1e-6)
            
            total_loss = total_loss + loss
            num_valid_scales += 1
        
        return total_loss / max(num_valid_scales, 1)
    
    def forward(
        self, 
        x:  torch.Tensor, 
        mode: str = 'features',
        mask_ratios: Optional[List[float]] = None
    ):
        """
        Args:
            x:  [B, 12, 5000]
            mode: 
                - 'pretrain': MAE 预训练，返回 (features, mae_loss)
                - 'finetune':  微调，返回分类 logits
                - 'linear_probe':  线性探测，返回分类 logits
                - 'features': 只返回特征
            mask_ratios:  可选，覆盖默认掩码比例
                
        Returns:
            根据 mode 返回不同内容
        """
        if mode == 'pretrain': 
            # MAE 预训练模式
            features, all_outputs, all_masks, all_ids_restore = self.encode(
                x, mask_ratios=mask_ratios
            )
            
            # 准备 decoder 输入
            encoded_features = [out.unsqueeze(2) for out in all_outputs]
            
            # 解码重建
            preds = self.mae_decoder(encoded_features, all_ids_restore)
            
            # 计算损失
            mae_loss = self.compute_mae_loss(x, preds, all_masks)
            
            return features, mae_loss
        
        else:
            # 非预训练模式：不使用掩码
            features, _, _, _ = self.encode(x, mask_ratios=[0.0] * self.num_scales)
            
            if mode == 'features':
                return features
            
            if self.num_classes is None:
                return features
            
            if mode in ['finetune', 'linear_probe']: 
                return self.linear_probe(features)
            
            return features


# ==================== 训练工具函数 ====================

def create_mae_optimizer(model:  FractalEcgMAE, lr: float = 1e-4, weight_decay: float = 0.05):
    """创建 MAE 预训练优化器"""
    # 分离 decay 和 no_decay 参数
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'pos' in name or 'mask_token' in name: 
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=lr, betas=(0.9, 0.95))
    
    return optimizer


def pretrain_step(model: FractalEcgMAE, x: torch.Tensor, optimizer:  torch.optim.Optimizer):
    """MAE 预训练单步"""
    model.train()
    optimizer.zero_grad()
    
    features, mae_loss = model(x, mode='pretrain')
    mae_loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return mae_loss.item()


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 创建模型
    model = FractalEcgMAE(
        hidden_dim=512,
        num_heads=8,
        depth=2,
        drop=0.1,
        num_classes=5,  # 例如 5 分类
        mask_ratios=[0.0, 0.5, 0.6, 0.75],  # 每个尺度的掩码比例
        decoder_depth=2,
        decoder_num_heads=4,
    )
    
    # 模拟输入
    batch_size = 4
    x = torch.randn(batch_size, 12, 5000)
    
    print("=" * 50)
    print("1.MAE 预训练模式")
    print("=" * 50)
    model.train()
    features, mae_loss = model(x, mode='pretrain')
    print(f"Features shape: {features.shape}")
    print(f"MAE Loss: {mae_loss.item():.4f}")
    
    print("\n" + "=" * 50)
    print("2.特征提取模式")
    print("=" * 50)
    model.eval()
    with torch.no_grad():
        features = model(x, mode='features')
    print(f"Features shape: {features.shape}")
    
    print("\n" + "=" * 50)
    print("3.Linear Probing 模式")
    print("=" * 50)
    model.freeze_backbone()
    logits = model(x, mode='linear_probe')
    print(f"Logits shape: {logits.shape}")
    
    print("\n" + "=" * 50)
    print("4.Fine-tuning 模式")
    print("=" * 50)
    model.unfreeze_backbone()
    model.freeze_decoder()  # 微调时冻结 decoder
    logits = model(x, mode='finetune')
    print(f"Logits shape: {logits.shape}")
    
    print("\n" + "=" * 50)
    print("5.参数统计")
    print("=" * 50)
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for n, p in model.named_parameters() if 'mae_decoder' not in n)
    decoder_params = sum(p.numel() for n, p in model.named_parameters() if 'mae_decoder' in n)
    print(f"Total params:  {total_params / 1e6:.2f}M")
    print(f"Encoder params:  {encoder_params / 1e6:.2f}M")
    print(f"Decoder params:  {decoder_params / 1e6:.2f}M")