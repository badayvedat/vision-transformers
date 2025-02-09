from config import ViTBaseConfig
from dataset import patchify, train_ds, test_ds
from model import ViT
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

# learning rate with cosine lr decay
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm  # Import tqdm for progress bars

# Hyperparameters
base_lr = 3e-3
batch_size = 512
warmup_steps = 10000
total_steps = 300000


# Learning rate scheduling function
def lr_lambda(current_step):
    if current_step < warmup_steps:
        return current_step / warmup_steps  # Linear warmup
    else:
        return 0.5 * (
            1
            + torch.cos(
                (current_step - warmup_steps)
                / (total_steps - warmup_steps)
                * 3.141592653589793
            )
        )


# Apply scheduler
def train(model: ViT, patch_size: int, train_loader: Dataset, test_loader: Dataset):
    step = 0

    # Table 3 in the paper
    # ViT-* ImageNet
    optimizer = AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.999), weight_decay=0.3)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    num_epochs = 10
    total_steps_count = num_epochs * len(
        train_loader
    )  # Total number of steps for all epochs

    # Wrap the whole training loop with tqdm for step-based progress
    progress_bar = tqdm(total=total_steps_count, desc="Training", unit="step")

    for epoch in range(num_epochs):
        for batch in train_loader:
            images, labels = batch["image"].cuda(), batch["label"].cuda()
            patches = patchify(images, patch_size)
            outputs = model(patches)

            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            step += 1
            progress_bar.set_postfix(loss=loss.item())  # Update progress bar with loss
            progress_bar.update(1)  # Update progress bar after each step

            if step % 100 == 0:
                print(f"Step {step}, Loss {loss.item()}")

    progress_bar.close()  # Close the progress bar at the end of training


if __name__ == "__main__":
    config = ViTBaseConfig()

    patch_size = 16
    num_classes = 10
    dropout = 0.1

    model = ViT(config, patch_size, num_classes, dropout).cuda()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    train(model, patch_size, train_loader, test_loader)
