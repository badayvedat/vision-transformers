from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
from datasets import load_dataset, Image

train_ds = load_dataset("ylecun/mnist", split="train")
train_ds = train_ds.cast_column("image", Image(mode="RGB"))

test_ds = load_dataset("ylecun/mnist", split="test")
test_ds = test_ds.cast_column("image", Image(mode="RGB"))


def train_transforms(example: dict):
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                (224, 224), scale=(0.8, 1.0), ratio=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.1307], [0.3081]),
        ]
    )
    images = [transform(image) for image in example["image"]]
    return {"image": images, "label": example["label"]}


def test_transforms(example: dict):
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.1307], [0.3081]),
        ]
    )(example["image"][0])

    return {"image": [image], "label": example["label"]}


train_ds.set_transform(train_transforms)
test_ds.set_transform(test_transforms)


def patchify(tensor: torch.Tensor, patch_size: int):
    N, C, H, W = tensor.shape

    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w
    patch_dim = C * patch_size * patch_size

    patches = tensor.reshape(N, C, num_patches_h, patch_size, num_patches_w, patch_size)
    # [B, C, H', p, W', p] -> [B, H', W', C, p, p]
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    # [B, H', W', C, p, p] -> [B, H' * W', C * p * p]
    patches = patches.view(N, num_patches, patch_dim)
    return patches


if __name__ == "__main__":
    dataloader = DataLoader(train_ds, batch_size=512, shuffle=True)
    for batch in dataloader:
        images, labels = batch["image"], batch["label"]
        print(images.shape, labels.shape)
