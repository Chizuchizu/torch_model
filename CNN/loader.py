import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


type_Dataloader = torch.utils.data.dataloader.DataLoader
type_transform = torchvision.transforms.transforms.Compose


def get_CIFAR10(
        train_transform: type_transform,
        val_transform: type_transform,
        root: str = "../data",  # working dirに入って作業することを想定
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        num_workers: int = 10
) -> tuple[type_Dataloader, type_Dataloader]:
    train_loader = DataLoader(
        CIFAR10(
            root=root,
            train=True,
            transform=train_transform,
            download=True
        ),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        CIFAR10(
            root=root,
            train=False,
            transform=val_transform,
            download=True
        ),
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
