import torch
import torch.nn as nn

from config import TransformerConfig


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + self.dropout1(attn_out)

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

        self.patch_size = patch_size
        num_patches = (config.image_size // patch_size) ** 2
        patch_dim = config.num_channels * patch_size**2

        self.proj = nn.Linear(patch_dim, config.hidden_dim)

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
        N, C, *_ = x.shape

        # Patchify the image: (N, C, H, W) -> (N, num_patches, patch_dim)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(N, -1, C * self.patch_size**2)

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

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
