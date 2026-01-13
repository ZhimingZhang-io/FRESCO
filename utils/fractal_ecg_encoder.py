import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


# ==================== Basic Modules ====================
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
    """Learnable Attention Pooling"""
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
    """Feed-Forward Network"""
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


# ==================== Core Attention Modules ====================

class IntraAttention(nn.Module):
    """Intra-group Attention"""
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
    """Inter-group Attention"""
    def __init__(self, dim, num_heads=8, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm = RMSNorm(dim)
    
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class GuidedCrossAttention(nn.Module):
    """Guided Cross-Attention"""
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


# ==================== MAE Related Modules ====================

class MaskedScaleBlock(nn.Module):

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
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            RMSNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Position encoding
        self.group_pos = nn.Parameter(torch.randn(1, 12, 1, hidden_dim) * 0.02)
        self.patch_pos = nn.Parameter(torch.randn(1, 1, num_patches, hidden_dim) * 0.02)
        
        # MAE: Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Multi-layer processing
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
        self.patch_norm = RMSNorm(hidden_dim) 
    
    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        """
        Perform random masking on patches.
        
        Args:
            x: [B, G, N, D] Input features
            mask_ratio: Masking ratio
            
        Returns:
            x_masked: [B, G, N_visible, D] Only visible patches are kept
            mask: [B, G, N] bool mask, True indicates masked
            ids_restore: [B, G, N] Indices to restore original order
        """
        B, G, N, D = x.shape
        num_mask = int(N * mask_ratio)
        num_visible = N - num_mask
        
        # Generate random permutation for each group
        noise = torch.rand(B, G, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=-1)
        ids_restore = torch.argsort(ids_shuffle, dim=-1)
        
        # Generate mask: first num_visible are visible
        mask = torch.zeros(B, G, N, device=x.device, dtype=torch.bool)
        mask_indices = ids_shuffle[:, : , num_visible:]  # Masked indices
        mask.scatter_(2, mask_indices, True)
        
        # Extract visible patches
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
        Merge encoded visible patches with mask tokens and restore original order.
        
        Args:
            x_encoded: [B, G, N_visible, D] Encoded visible patches
            ids_restore: [B, G, N] Restoration indices
            num_patches: Original number of patches
            
        Returns: 
            x_full: [B, G, N, D] Full sequence (visible + mask tokens)
        """
        B, G, N_visible, D = x_encoded.shape
        num_mask = num_patches - N_visible
        
        # Expand mask tokens
        mask_tokens = self.mask_token.expand(B, G, num_mask, -1)
        
        # Concatenate
        x_full = torch.cat([x_encoded, mask_tokens], dim=2)  # [B, G, N, D]
        
        # Restore original order
        x_full = torch.gather(x_full, dim=2, index=ids_restore.unsqueeze(-1).expand(-1, -1, -1, D))
        
        return x_full
    
    def forward(
        self, 
        x: torch.Tensor, 
        guidance: Optional[torch.Tensor] = None,
        mask_ratio: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args: 
            x: [B, 12, num_patches, input_dim]
            guidance: [B, 12, hidden_dim] (Optional)
            mask_ratio: Masking ratio (0.0 = no masking)
            
        Returns:
            output_pooled: [B, 12, hidden_dim] Pooled output for fusion
            output_patches: [B, 12, N_visible, hidden_dim] Patch output for decoder
            mask: [B, 12, num_patches] or None
            ids_restore: [B, 12, num_patches] or None
        """
        B, G, N, _ = x.shape
        
        # Projection
        x = self.input_proj(x)
        
        # Add position embeddings (before masking, so mask tokens get correct pos info later)
        x = x + self.group_pos + self.patch_pos
        
        # MAE Masking
        mask = None
        ids_restore = None
        if mask_ratio > 0 and self.training:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        for layer in self.layers:
            if self.use_guidance and guidance is not None: 
                x = layer['guide_attn'](x, guidance)
            
            # Intra-group Attention
            x = layer['intra_attn'](x)
            x = x + layer['intra_ffn'](layer['intra_ffn_norm'](x))
            
            # Pooling
            current_N = x.shape[2]
            x_flat = x.reshape(B * G, current_N, -1)
            x_group = layer['pool'](x_flat)
            x_group = x_group.reshape(B, G, -1)
            
            # Inter-group Attention
            x_group = layer['inter_attn'](x_group)
            x_group = x_group + layer['inter_ffn'](layer['inter_ffn_norm'](x_group))
            
            guidance = x_group
        
        # Return both pooled representation and patch representation
        return self.output_norm(x_group), self.patch_norm(x), mask, ids_restore


class MAEDecoder(nn.Module):
    """
    MAE Decoder: Reconstructs masked patches.
    Lightweight design, shared structure per scale but independent parameters.
    """
    def __init__(
        self, 
        hidden_dim: int, 
        output_dim: int,  # patch_dim, for reconstructing original signal
        num_patches: int,
        num_heads: int = 4,
        depth: int = 2,
        drop: float = 0.1
    ):
        super().__init__()
        self.num_patches = num_patches
        self.output_dim = output_dim
        
        # Map from encoder hidden_dim to decoder dim
        decoder_dim = hidden_dim // 2
        self.decoder_embed = nn.Linear(hidden_dim, decoder_dim)
        
        # Mask token for decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Position encodings
        self.pos_embed = nn.Parameter(torch.randn(1, 1, num_patches, decoder_dim) * 0.02)
        self.group_embed = nn.Parameter(torch.randn(1, 12, 1, decoder_dim) * 0.02)
        
        # Decoder layers (Lightweight)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': IntraAttention(decoder_dim, num_heads, drop),
                'ffn':  FeedForward(decoder_dim, mult=2, drop=drop),
                'ffn_norm': RMSNorm(decoder_dim),
            })
            for _ in range(depth)
        ])
        
        self.norm = RMSNorm(decoder_dim)
        
        # Reconstruction head
        self.pred = nn.Linear(decoder_dim, output_dim)
    
    def forward(
        self, 
        x_encoded: torch.Tensor,  # [B, G, N_visible, hidden_dim] Encoder output
        ids_restore: torch.Tensor,  # [B, G, N]
    ) -> torch.Tensor:
        """
        Args:
            x_encoded: Encoded visible patches
            ids_restore: Indices to restore original order
            
        Returns:
            pred: [B, G, N, output_dim] Reconstructed patches
        """
        B, G, N_visible, _ = x_encoded.shape
        N = self.num_patches
        num_mask = N - N_visible
        
        # Map to decoder dimension
        x = self.decoder_embed(x_encoded)
        D = x.shape[-1]
        
        # Append mask tokens
        mask_tokens = self.mask_token.expand(B, G, num_mask, -1)
        x_full = torch.cat([x, mask_tokens], dim=2)
        
        # Restore original order
        x_full = torch.gather(x_full, dim=2, index=ids_restore.unsqueeze(-1).expand(-1, -1, -1, D))
        
        # Add position embeddings
        x_full = x_full + self.pos_embed + self.group_embed
        
        # Decoder layers
        for layer in self.layers:
            x_full = layer['attn'](x_full)
            x_full = x_full + layer['ffn'](layer['ffn_norm'](x_full))
        
        x_full = self.norm(x_full)
        
        # Prediction
        pred = self.pred(x_full)  # [B, G, N, output_dim]
        
        return pred


class MultiScaleMAEDecoder(nn.Module):
    """Multi-scale MAE Decoder"""
    def __init__(self, hidden_dim: int, scale_configs: List[Tuple[int, int]], num_heads=4, depth=2, drop=0.1):
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
        encoded_features: List[torch.Tensor],  # Encoded features for each scale
        ids_restores: List[torch.Tensor],  # Restore indices for each scale
    ) -> List[torch.Tensor]: 
        """Returns reconstruction results for each scale"""
        preds = []
        for decoder, feat, ids_restore in zip(self.decoders, encoded_features, ids_restores):
            if ids_restore is not None:
                pred = decoder(feat, ids_restore)
                preds.append(pred)
            else:
                preds.append(None)
        return preds


# ==================== Main Model ====================

class FractalEcgMAE(nn.Module):
    """
    Fractal ECG Encoder with MAE Pretraining
    
    Training Modes: 
    - 'pretrain': MAE pretraining, returns reconstruction loss.
    - 'linear_probe': Linear probing, backbone frozen.
    - 'features': Returns features only.
    """
    def __init__(
        self, 
        hidden_dim: int = 512, 
        num_heads: int = 8, 
        depth: int = 2, 
        drop: float = 0.1,
        num_classes: Optional[int] = None,
        # MAE Parameters
        mask_ratios: Optional[List[float]] = None,  # Mask ratio for each scale
        decoder_depth: int = 2,
        decoder_num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Default mask ratios: more masking for fine-grained scales
        self.mask_ratios = mask_ratios or [0.0, 0.5, 0.6, 0.75]  # Scale 0 has no masking (only 1 patch)
        
        # Scale configurations
        self.scale_configs = [
            (1, 5000),    # 1 patch × 5000 dim
            (10, 500),    # 10 patches × 500 dim
            (100, 50),    # 100 patches × 50 dim
            (500, 10),    # 500 patches × 10 dim
        ]
        self.num_scales = len(self.scale_configs)
        
        # ========== Encoder ==========
        # Scale 0: No guidance, no masking
        self.scale0 = MaskedScaleBlock(
            input_dim=self.scale_configs[0][1],
            hidden_dim=hidden_dim,
            num_patches=self.scale_configs[0][0],
            num_heads=num_heads,
            depth=depth,
            drop=drop,
            use_guidance=False
        )
        
        # Scale 1, 2, 3: Guided
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
        
        # Scale Fusion
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
        
        # ========== Classification Head ==========
        if num_classes is not None:
            self.linear_probe = nn.Linear(hidden_dim, num_classes)
    
    def freeze_backbone(self):
        """Freeze backbone network for Linear Probing"""
        for name, param in self.named_parameters():
            if 'linear_probe' not in name:
                param.requires_grad = False
        print("Backbone frozen. Only classifier is trainable.")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone network"""
        for param in self.parameters():
            param.requires_grad = True
        print("All parameters are trainable.")
    
    def freeze_decoder(self):
        """Freeze MAE decoder (used during fine-tuning)"""
        for param in self.mae_decoder.parameters():
            param.requires_grad = False
        print("MAE decoder frozen.")
    
    def encode(
        self, 
        x: torch.Tensor, 
        mask_ratios: Optional[List[float]] = None
    ) -> Tuple[torch.Tensor, List, List, List]:
        """
        Encoder forward pass
        
        Returns:
            features: [B, hidden_dim] Fused features
            all_scale_outputs: Output for each scale (pooled)
            all_encoded_patches: Output patches for each scale (for decoder)
            all_masks: Mask for each scale
            all_ids_restore: Restoration indices for each scale
        """
        B = x.shape[0]
        mask_ratios = mask_ratios or self.mask_ratios
        
        all_scale_outputs = [] # Pooled features for Fusion
        all_encoded_patches = [] # Patch features for Decoder
        all_masks = []
        all_ids_restore = []
        
        # Scale 0
        x0 = x.reshape(B, 12, 1, 5000)
        # Receive additional x_patches
        out0, x_patches0, mask0, ids0 = self.scale0(x0, mask_ratio=mask_ratios[0])
        
        all_scale_outputs.append(out0)
        all_encoded_patches.append(x_patches0) # Store patches
        all_masks.append(mask0)
        all_ids_restore.append(ids0)
        
        prev_guidance = out0
        
        # Scale 1, 2, 3
        for i, guided_scale in enumerate(self.guided_scales):
            num_patches, patch_dim = self.scale_configs[i + 1]
            xi = x.reshape(B, 12, num_patches, patch_dim)
            # Receive additional x_patches
            outi, x_patchesi, maski, idsi = guided_scale(xi, prev_guidance, mask_ratio=mask_ratios[i + 1])
            
            all_scale_outputs.append(outi)
            all_encoded_patches.append(x_patchesi) # Store patches
            all_masks.append(maski)
            all_ids_restore.append(idsi)
            prev_guidance = outi
        
        # Fusion
        features = self.scale_fusion(all_scale_outputs)
        
        # Return includes all_encoded_patches
        return features, all_encoded_patches, all_masks, all_ids_restore
    
    def compute_mae_loss(
        self, 
        x: torch.Tensor,
        preds: List[torch.Tensor],
        masks: List[torch.Tensor],
        normalize_target: bool = True
    ) -> torch.Tensor:
        """
        Compute MAE reconstruction loss.
        Calculates loss only on masked positions.
        """
        B = x.shape[0]
        total_loss = 0.0
        num_valid_scales = 0
        
        for scale_idx, (pred, mask) in enumerate(zip(preds, masks)):
            if pred is None or mask is None: 
                continue
            
            num_patches, patch_dim = self.scale_configs[scale_idx]
            
            # Get original target
            target = x.reshape(B, 12, num_patches, patch_dim)
            
            # Optional: Normalize target (patch-wise normalization)
            if normalize_target: 
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                std = (var + 1e-6).sqrt()
                target = (target - mean) / std
            
            # Compute MSE loss
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [B, G, N]
            
            # Calculate loss only on masked positions
            mask_float = mask.float()
            loss = (loss * mask_float).sum() / (mask_float.sum() + 1e-6)
            
            total_loss = total_loss + loss
            num_valid_scales += 1
        
        return total_loss / max(num_valid_scales, 1)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mode: str = 'features',
        mask_ratios: Optional[List[float]] = None
    ):
        """
        Args:
            x: [B, 12, 5000]
            mode: 
                - 'pretrain': MAE pretraining, returns (features, mae_loss)
                - 'finetune': Fine-tuning, returns classification logits
                - 'linear_probe': Linear probing, returns classification logits
                - 'features': Returns features only
            mask_ratios: Optional, overrides default mask ratios
                
        Returns:
            Depends on mode
        """
        if mode == 'pretrain': 
            # MAE Pretraining Mode
            features, all_encoded_patches, all_masks, all_ids_restore = self.encode(
                x, mask_ratios=mask_ratios
            )
            
            # Pass patch sequence directly to decoder
            preds = self.mae_decoder(all_encoded_patches, all_ids_restore)
            
            # Compute loss
            mae_loss = self.compute_mae_loss(x, preds, all_masks)
            
            return features, mae_loss
        
        else:
            # Non-pretraining modes: No masking
            features, _, _, _ = self.encode(x, mask_ratios=[0.0] * self.num_scales)
            
            if mode == 'features':
                return features
            
            if self.num_classes is None:
                return features
            
            if mode in ['finetune', 'linear_probe']: 
                return self.linear_probe(features)
            
            return features


# ==================== Usage Example ====================

if __name__ == "__main__":
    # Create model
    model = FractalEcgMAE(
        hidden_dim=512,
        num_heads=8,
        depth=2,
        drop=0.1,
        num_classes=5,  # e.g., 5-class classification
        mask_ratios=[0.0, 0.5, 0.6, 0.75],  # Mask ratio for each scale
        decoder_depth=2,
        decoder_num_heads=4,
    )
    
    # Simulate input
    batch_size = 4
    x = torch.randn(batch_size, 12, 5000)
    
    print("=" * 50)
    print("1. MAE Pretraining Mode")
    print("=" * 50)
    model.train()
    features, mae_loss = model(x, mode='pretrain')
    print(f"Features shape: {features.shape}")
    print(f"MAE Loss: {mae_loss.item():.4f}")
    
    print("\n" + "=" * 50)
    print("2. Feature Extraction Mode")
    print("=" * 50)
    model.eval()
    with torch.no_grad():
        features = model(x, mode='features')
    print(f"Features shape: {features.shape}")
    
    print("\n" + "=" * 50)
    print("3. Linear Probing Mode")
    print("=" * 50)
    model.freeze_backbone()
    logits = model(x, mode='linear_probe')
    print(f"Logits shape: {logits.shape}")
    

    print("\n" + "=" * 50)
    print("5. Parameter Statistics")
    print("=" * 50)
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for n, p in model.named_parameters() if 'mae_decoder' not in n)
    decoder_params = sum(p.numel() for n, p in model.named_parameters() if 'mae_decoder' in n)
    print(f"Total params:  {total_params / 1e6:.2f}M")
    print(f"Encoder params:  {encoder_params / 1e6:.2f}M")
    print(f"Decoder params:  {decoder_params / 1e6:.2f}M")