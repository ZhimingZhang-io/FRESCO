
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x*norm).type_as(x)*self.weight

class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        hidden_dim = int(dim*mlp_ratio)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(drop)
    
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)*self.w3(x))))

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, L, D = x. shape
        qkv = self.qkv(x). reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(x)

class CrossAttention(nn. Module):
    def __init__(self, dim, cond_dim, num_heads=8, attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(cond_dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x, cond):
        B, L, _ = x.shape
        S = cond.shape[1]
        q = self.q(x).reshape(B, L, self. num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(cond).reshape(B, S, 2, self.num_heads, self. head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        return self.proj(x)

class FractalBlock(nn.Module):
    def __init__(self, dim, cond_dim, num_heads,
                 mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.has_cond = cond_dim is not None
        self.norm1 = RMSNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads, drop)
        if self.has_cond:
            self.norm2 = RMSNorm(dim)
            self.cross_attn = CrossAttention(dim, cond_dim, num_heads, drop)
        self.norm3 = RMSNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio, drop)

    def forward(self, x, cond=None):
        x = x + self.self_attn(self.norm1(x))
        if self.has_cond and cond is not None:
            x = x + self.cross_attn(self.norm2(x), cond)
        x = x + self.ffn(self.norm3(x))
        return x

class MultiScaleFusion(nn.Module):
    def __init__(self, embed_dim_list, output_dim):
        super().__init__()
        self.num_levels = len(embed_dim_list)
        self.output_dim = output_dim
        self.level_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                RMSNorm(output_dim),
            )
            for dim in embed_dim_list
        ])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, output_dim))
        self.fusion_attn = SelfAttention(output_dim, num_heads=8)
        self.fusion_norm = RMSNorm(output_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, all_features):

        B = all_features[0].shape[0]
        level_feats = []
        for i, (feat, proj) in enumerate(zip(all_features, self.level_projs)):
            # [B, L_i, D_i] -> [B, D_i] -> [B, output_dim]
            pooled = feat.mean(dim=1)
            projected = proj(pooled)
            level_feats.append(projected)
        
        stacked = torch.stack(level_feats, dim=1)  # [B, num_levels, dim]
        cls_tokens = self.cls_token. expand(B, -1, -1)  # [B, 1, dim]
        
        tokens = torch.cat([cls_tokens, stacked], dim=1)  # [B, 1+num_levels, dim]
        tokens = self.fusion_attn(tokens)
        tokens = self.fusion_norm(tokens)
        output = tokens[:, 0]  # [B, dim]
        return output
        
class FractalECGEncoder(nn.Module):
    def __init__(
        self,
        num_leads=12,
        seq_len=5000,
        num_patches_list=(10, 10, 5, 1),
        embed_dim_list=(512, 384, 256, 128),
        num_blocks_list=(6, 4, 3, 2),
        num_heads_list=(8, 6, 4, 2),
        mlp_ratio=4.0,
        drop_rate=0.1,
        output_dim=512,
        _fractal_level=0,
        _input_patch_len=None,
        _shared_fusion=None
    ):
        super().__init__()
        
        self.fractal_level = _fractal_level
        self.num_levels = len(num_patches_list)
        self.num_leads = num_leads
        self.is_first = (_fractal_level == 0)
        self.is_last = (_fractal_level == self.num_levels - 1)

        
        self.num_patches = num_patches_list[_fractal_level]
        self.embed_dim = embed_dim_list[_fractal_level]
        

        if self.is_first:
            self.input_patch_len = seq_len
        else:
            self.input_patch_len = _input_patch_len
        self.output_patch_len = self.input_patch_len // self.num_patches
        
        # --------------------------------------------------------------------------
        # Patch Embedding
        self.patch_embed = nn.Conv1d(
            num_leads, self.embed_dim,
            kernel_size=self.output_patch_len,
            stride=self.output_patch_len
        )
        
        # Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self. embed_dim))
        nn.init.trunc_normal_(self. pos_embed, std=0.02)
        

        if not self.is_first:
            prev_dim = embed_dim_list[_fractal_level - 1]
            self.cond_proj = nn.Linear(prev_dim, self.embed_dim)
        
        # Transformer Blocks
        cond_dim = self.embed_dim if not self.is_first else None
        self.blocks = nn.ModuleList([
            FractalBlock(self.embed_dim, cond_dim, num_heads_list[_fractal_level],
                        mlp_ratio, drop_rate)
            for _ in range(num_blocks_list[_fractal_level])
        ])
        self.norm = RMSNorm(self.embed_dim)

        # Share a single fusion head across all recursive levels; if a sub-level
        # is instantiated directly (without a parent), create the fusion here.
        if self.is_first or _shared_fusion is None:
            self.fusion = MultiScaleFusion(
                embed_dim_list=embed_dim_list,
                output_dim=output_dim
            )
            _shared_fusion = self.fusion
        else:
            self.fusion = _shared_fusion

        if not self.is_last:
            self.next_level = FractalECGEncoder(
                num_leads=num_leads,
                seq_len=seq_len,
                num_patches_list=num_patches_list,
                embed_dim_list=embed_dim_list,
                num_blocks_list=num_blocks_list,
                num_heads_list=num_heads_list,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                output_dim=output_dim,
                _fractal_level=_fractal_level + 1,
                _input_patch_len=self. output_patch_len,
                _shared_fusion=_shared_fusion
            )
    def patchify(self, x):
        if self.is_first:
            B = x.shape[0]
            h = self.patch_embed(x).transpose(1, 2)
            patches = x.reshape(B, self.num_leads, self.num_patches, self.output_patch_len)
            patches = patches.permute(0, 2, 1, 3)
        else:
            B, N_prev, leads, patch_len = x.shape
            x_flat = x.reshape(B * N_prev, leads, patch_len)
            h = self.patch_embed(x_flat).transpose(1, 2)
            h = h.reshape(B, N_prev * self.num_patches, self.embed_dim)
            patches = x_flat.reshape(B * N_prev, leads, self.num_patches, self. output_patch_len)
            patches = patches.permute(0, 2, 1, 3)
            patches = patches.reshape(B, N_prev * self.num_patches, leads, self.output_patch_len)
        return h, patches
    
    def _forward_features(self, x, cond=None):
        # 1.  Patchify
        h, patches = self.patchify(x)
        B, N_curr, _ = h.shape
        
        # 2. Position Embedding
        if self.is_first:
            h = h + self.pos_embed
        else:
            N_prev = N_curr // self.num_patches
            h = h.reshape(B, N_prev, self. num_patches, self.embed_dim)
            h = h + self.pos_embed
            h = h.reshape(B, N_curr, self.embed_dim)
        
        if not self.is_first and cond is not None:
            cond_proj = self.cond_proj(cond)
            N_prev = N_curr // self.num_patches
            cond_expanded = cond_proj.unsqueeze(2).expand(-1, -1, self.num_patches, -1)
            cond_expanded = cond_expanded.reshape(B, N_curr, self.embed_dim)
        else:
            cond_expanded = None
        
        # 4.  Transformer Blocks
        for block in self. blocks:
            h = block(h, cond_expanded)
        h = self.norm(h)
        
        if not self.is_last:
            _, all_features = self.next_level._forward_features(patches, cond=h)
            all_features.insert(0, h)
            return h, all_features
        else:
            return h, [h]

    def forward(self, x):
        _, all_features = self._forward_features(x)
        output = self.fusion(all_features)
        
        return output
if __name__ == "__main__":

    model = FractalECGEncoder(
        num_leads=12,
        seq_len=5000,
        num_patches_list=[10, 10, 5, 1],
        embed_dim_list=[256, 192, 128, 64],
        num_blocks_list=[8, 4, 2, 1],
        num_heads_list=[4, 3, 2, 1],
        output_dim=512,
    )
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.2f}M")

    x = torch.randn(4, 12, 5000)
    output= model(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape},{type(output)}")



