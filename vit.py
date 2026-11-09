"""ViT backbone used by JEM (paper Sec. 4.1): patch 16, no class token, 4 register tokens,
axial RoPE, LayerScale 1e-5, drop path. Returns patch tokens at any requested depth."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_angles(h, w, head_dim, base=100.0, device=None):
    """Axial RoPE angles for an h x w patch grid, coordinates normalised to [-1, 1] per crop (DINOv3)."""
    ys = (torch.arange(h, device=device) + 0.5) / h * 2 - 1
    xs = (torch.arange(w, device=device) + 0.5) / w * 2 - 1
    coords = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), -1).flatten(0, 1)  # (h*w, 2)
    periods = base ** (2 * torch.arange(head_dim // 4, device=device) / (head_dim // 2))
    angles = 2 * math.pi * coords[:, :, None] / periods  # (h*w, 2, head_dim/4)
    angles = angles.flatten(1).tile(2)  # (h*w, head_dim)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-x2, x1], -1) * sin


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, drop_path=0.0, ls_init=1e-5):
        super().__init__()
        self.heads, self.drop_path = heads, drop_path
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.fc1, self.fc2 = nn.Linear(dim, int(dim * mlp_ratio)), nn.Linear(int(dim * mlp_ratio), dim)
        if ls_init is None:  # ablation: no LayerScale (plain residual branches)
            self.ls1 = self.ls2 = 1.0
        else:
            self.ls1 = nn.Parameter(torch.full((dim,), ls_init))
            self.ls2 = nn.Parameter(torch.full((dim,), ls_init))

    def attn(self, x, cos, sin, n_prefix):
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        # RoPE only on patch tokens, not on registers
        q = torch.cat([q[:, :, :n_prefix], apply_rope(q[:, :, n_prefix:], cos, sin)], 2)
        k = torch.cat([k[:, :, :n_prefix], apply_rope(k[:, :, n_prefix:], cos, sin)], 2)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, D))

    def residual(self, x, y):
        if self.training and self.drop_path > 0:  # per-sample stochastic depth
            keep = torch.rand(x.shape[0], 1, 1, device=x.device) >= self.drop_path
            y = y * keep / (1 - self.drop_path)
        return x + y

    def forward(self, x, cos, sin, n_prefix):
        x = self.residual(x, self.ls1 * self.attn(self.norm1(x), cos, sin, n_prefix))
        return self.residual(x, self.ls2 * self.fc2(F.gelu(self.fc1(self.norm2(x)))))


class ViT(nn.Module):
    def __init__(self, dim=768, depth=12, heads=12, patch=16, n_reg=4, drop_path=0.1, ls_init=1e-5):
        super().__init__()
        self.patch, self.n_reg, self.head_dim = patch, n_reg, dim // heads
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        self.registers = nn.Parameter(torch.zeros(1, n_reg, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList(Block(dim, heads, drop_path=drop_path, ls_init=ls_init) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.apply(self._init)
        nn.init.trunc_normal_(self.registers, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, img, mask=None, layers=()):
        """img (B,3,H,W); mask (B,N) bool, True = replaced by mask token.
        Returns {k: patch tokens after block k} for k in `layers` (0 = patch embedding output)
        and the final patch tokens after the final LayerNorm under key "final"."""
        x = self.patch_embed(img)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        if mask is not None:
            x = torch.where(mask[..., None], self.mask_token.to(x.dtype), x)
        cos, sin = rope_angles(h, w, self.head_dim, device=x.device)
        out = {0: x} if 0 in layers else {}
        x = torch.cat([self.registers.expand(len(x), -1, -1).to(x.dtype), x], 1)
        for i, blk in enumerate(self.blocks, 1):
            x = blk(x, cos, sin, self.n_reg)
            if i in layers:
                out[i] = x[:, self.n_reg:]
        x = self.norm(x)
        out["final"], out["registers"] = x[:, self.n_reg:], x[:, : self.n_reg]
        return out
