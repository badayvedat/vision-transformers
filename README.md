# Vision Transformers (ViT)

This repository contains a **minimal, self-contained** implementation of Vision Transformers (ViT) in PyTorch.  

The implementation is based on the original paper:  
[Dosovitskiy et al., 2020. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929).


## Installation

Ensure you have **Python 3.11+** installed, then install the required dependencies:

```bash
pip install -e .
```

Alternatively, if you're using uv:

```bash
uv pip install
```


## Structure

- **config.py**: Defines transformer configurations (ViT-Base, ViT-Large, ViT-Huge).  
- **dataset.py**: Loads the `MNIST` dataset, applies transformations, and prepares it for training. Even though the original paper uses `ImageNet`, I used `MNIST` for simplicity.
- **model.py**: Implements the Vision Transformer model, including patch embedding and transformer layers.  
- **train.py**: Training script with AdamW optimizer, cosine LR decay, and gradient clipping.    



## Usage

### Model Configurations
To see the model architectures, run:

```bash
python config.py
```

Which will output:

```bash
                      ViT Configs                      
┏━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┓
┃ Config Name ┃ Layers ┃ Hidden Dim ┃ MLP Dim ┃ Heads ┃
┡━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━┩
│ Base        │ 12     │ 768        │ 3072    │ 12    │
│ Large       │ 24     │ 1024       │ 4096    │ 16    │
│ Huge        │ 32     │ 1280       │ 5120    │ 16    │
└─────────────┴────────┴────────────┴─────────┴───────┘
```

### Training
To train the model with `ViT-Base` configuration and `MNIST` dataset, run:

```bash
python train.py
```
