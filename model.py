import torch
import torch.nn as nn

from config import TransformerConfig


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_out = self.norm1(x)
        x = x + self.attn(norm_out, norm_out, norm_out, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        patch_size: int,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        num_patches = (config.image_size // patch_size) ** 2

        self.proj = nn.Linear(patch_size**2 * config.num_channels, config.hidden_dim)
        self.transformer = nn.Sequential(
            *[
                EncoderLayer(config.hidden_dim, config.heads, config.mlp_dim, dropout)
                for _ in range(config.layers)
            ]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, config.hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, *_ = x.shape

        # patch embeddings
        x = self.proj(x)

        cls_tokens = self.cls_token.expand(N, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding

        x = self.dropout(x)
        x = self.transformer(x)
        cls = x[:, 0]
        out = self.mlp_head(cls)
        return out
