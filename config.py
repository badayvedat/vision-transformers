from dataclasses import dataclass


@dataclass
class TransformerConfig:
    layers: int
    hidden_dim: int
    mlp_dim: int
    heads: int
    num_channels: int = 3
    image_size: int = 224


@dataclass
class ViTBaseConfig(TransformerConfig):
    layers: int = 12
    hidden_dim: int = 768
    mlp_dim: int = 3072
    heads: int = 12


@dataclass
class ViTLargeConfig(TransformerConfig):
    layers: int = 24
    hidden_dim: int = 1024
    mlp_dim: int = 4096
    heads: int = 16


@dataclass
class ViTHugeConfig(TransformerConfig):
    layers: int = 32
    hidden_dim: int = 1280
    mlp_dim: int = 5120
    heads: int = 16


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()

    table = Table(title="ViT Configs")
    table.add_column("Config Name")
    table.add_column("Layers")
    table.add_column("Hidden Dim")
    table.add_column("MLP Dim")
    table.add_column("Heads")

    for config in [ViTBaseConfig(), ViTLargeConfig(), ViTHugeConfig()]:
        cls_name: str = config.__class__.__name__
        config_name = cls_name.replace("Config", "").replace("ViT", "")
        table.add_row(
            config_name,
            str(config.layers),
            str(config.hidden_dim),
            str(config.mlp_dim),
            str(config.heads),
        )

    console.print(table)
