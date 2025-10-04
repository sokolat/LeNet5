import argparse
import random

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from time import time
import os
import wandb
import numpy as np

"""
ResNet18 Implementation for ImageNette Classification

This module implements a ResNet18 model for image classification using the ImageNette dataset.
It includes features such as custom data augmentation, experiment tracking with Weights & Biases,
TensorBoard logging, and checkpoint management.

The implementation follows the original ResNet architecture with modifications for the ImageNette
dataset, which contains 10 classes. The training process includes advanced features like
learning rate scheduling and custom data augmentation using eigenvector-based perturbation.

Author: Rango
Date: June 2025
"""


def get_args():
    """
    Parse and return command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments including:
            - device: Device to use for training
            - train_dir: Path to training data
            - val_dir: Path to validation data
            - batch_size: Training batch size
            - lr: Learning rate
            - weight_decay: Weight decay for optimizer
            - momentum: Momentum for optimizer
            - epochs: Number of training epochs
            - wandb_project_name: Weights & Biases project name
            - wandb_entity: Weights & Biases entity/username
            - track: Whether to enable W&B tracking
            - checkpoint: Path to checkpoint for resuming training
            - seed: Random seed for reproducibility
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_dir", type=str, default="./data/train")
    parser.add_argument("--val_dir", type=str, default="./data/val")
    parser.add_argument("--test_dir", type=str, default="./data/test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--wandb_project_name", type=str, default="resnet18")
    parser.add_argument("--wandb_entity", type=str, default="your_wandb_entity")
    parser.add_argument("--track", action="store_true", help="Enable Weights & Biases tracking")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--patience", type=int, default=5, help="Patience for learning rate scheduler")

    return parser.parse_args()


class Block(nn.Module):
    """
    Basic ResNet block with two 3x3 convolutions and a residual connection.

    The block maintains the spatial dimensions of the input if stride=1,
    or reduces spatial dimensions by half if stride=2.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int, optional): Stride for first convolution. Defaults to 1.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()

        # First convolution layer
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution layer
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU()
        
        # Shortcut connection if dimensions change
        self.shortcut = nn.Sequential()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = out + identity
        
        return out


class ResNet18(nn.Module):
    """
    ResNet18 architecture implementation for ImageNette classification.

    The model consists of:
    - Initial 7x7 convolution layer
    - 4 layers of residual blocks (2 blocks each)
    - Average pooling and final fully connected layer
    - Total of 18 layers with learnable parameters

    The model is designed for the ImageNette dataset with 10 classes.
    """
    def __init__(self):
        super(ResNet18, self).__init__()

        # Initial layers
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 1 (64 channels)
        self.block1 = Block(64, 64)
        self.block2 = Block(64, 64)

        # Layer 2 (128 channels)
        self.block3 = Block(64, 128, 2)
        self.block4 = Block(128, 128)

        # Layer 3 (256 channels)
        self.block5 = Block(128, 256, 2)
        self.block6 = Block(256, 256)

        # Layer 4 (512 channels)
        self.block7 = Block(256, 512, 2)
        self.block8 = Block(512, 512)

        # Output layers
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=2)
        self.fc = nn.Linear(in_features=512, out_features=10)  # 10 classes for ImageNette

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.block1(x)
        x = self.block2(x)

        x = self.block3(x)
        x = self.block4(x)

        x = self.block5(x)
        x = self.block6(x)

        x = self.block7(x)
        x = self.block8(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class DatasetWrapper(Dataset):
    """
    Custom dataset wrapper that applies eigenvector-based data augmentation.

    The augmentation process:
    1. Flattens the input image
    2. Computes covariance matrix
    3. Performs eigendecomposition
    4. Applies random perturbations along eigenvectors

    Args:
        dataset (Dataset): Base dataset to wrap
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        flat_data = torch.flatten(data, start_dim=1)
        cov = torch.cov(flat_data)
        eig_vals, eig_vectors = torch.linalg.eig(cov)
        alphas = torch.normal(mean=0, std=0.1, size=(3,))
        multipliers = alphas * eig_vals
        aug = eig_vectors @ multipliers
        aug = torch.unsqueeze(torch.unsqueeze(aug.real, 1), 1)

        data += aug

        return data, target

    def __len__(self):
        return len(self.dataset)


def get_size():
    """
    Generate a random image size for dynamic resizing during training.

    Returns:
        int: Random size between 256 and 480 pixels
    """
    return random.randint(256, 480)


def init_weight(model):
    """
    Initialize model weights using Kaiming normalization.

    Args:
        model (nn.Module): The neural network model to initialize
    """
    for name, params in model.named_parameters():
        if len(params.shape) > 1:
            torch.nn.init.kaiming_normal_(params.data)

def get_transform(size):
    """
    Create a composition of image transformations for data preprocessing.

    Args:
        size (int): Target size for image resizing

    Returns:
        transforms.Compose: Composition of transforms including:
            - ToTensor
            - Resize
            - RandomHorizontalFlip
            - CenterCrop
            - Normalize
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.CenterCrop(224),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

def set_seed(seed=42):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int, optional): Random seed. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args, model, device):
    """
    Train the model using the specified parameters.

    Features:
    - Weights & Biases integration for experiment tracking
    - TensorBoard logging
    - Checkpoint saving
    - Learning rate scheduling
    - Custom data augmentation

    Args:
        args (argparse.Namespace): Command line arguments
        model (nn.Module): The neural network model to train
        device (torch.device): Device to use for training
    """
    if args.track:
        
        run_name = f"run_{int(time())}_resnet18_imagenette"

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            sync_tensorboard=True,
            name=run_name,
            save_code=True
        )

        logger = SummaryWriter(log_dir=f"runs/{run_name}")

    transform = get_transform(get_size())
    train_dataset = ImageFolder(root=args.train_dir, transform=transform)
    train_dataset = DatasetWrapper(train_dataset)
    val_dataset = ImageFolder(root=args.val_dir, transform=transform)
    val_dataset = DatasetWrapper(val_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    model.to(device)
    init_weight(model)

    optimizer = SGD(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum
    )

    """
    optimizer =torch.optim.Adam(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    """

    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, patience=args.patience, mode='max')

    last_epoch = 0

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        last_epoch = checkpoint["epoch"]

        print(f"Resuming from epoch {last_epoch}")
    
    timestamp = str(int(time()))

    for epoch in range(last_epoch+1, args.epochs + 1):

        running_loss = 0.0
        running_accuracy = 0.0

        for idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            running_accuracy += (output.argmax(dim=1) == target).sum().item()

            if idx % 10 == 9:
                print(
                    f"Epoch [{epoch}/{args.epochs}], Step [{idx + 1}/{len(train_loader)}], "
                    f"Loss: {running_loss / ((idx + 1) * data.size(0)):.4f}, "
                    f"Accuracy: {running_accuracy / ((idx + 1) * data.size(0)):.4f}"
                )

                if args.track:
                    logger.add_scalar("train_loss", running_loss / ((idx + 1) * data.size(0)), epoch * len(train_loader) + idx)
                    logger.add_scalar("train_accuracy", running_accuracy / ((idx + 1) * data.size(0)), epoch * len(train_loader) + idx)
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_accuracy = running_accuracy / len(train_dataset)

        val_loss, val_accuracy = validate(model, val_loader, device, criterion)

        scheduler.step(val_accuracy)

        print(f"Epoch [{epoch}/{args.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if args.track:
            logger.add_scalar("epoch_loss", epoch_loss, epoch)
            logger.add_scalar("epoch_accuracy", epoch_accuracy, epoch)
            logger.add_scalar("val_loss", val_loss, epoch)
            logger.add_scalar("val_accuracy", val_accuracy, epoch)

        save_dir = os.path.join("checkpoints", timestamp)
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = f"checkpoint_epoch_{epoch}.pth"

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(save_dir, checkpoint_path)
        )

        print(f"Checkpoint saved to {checkpoint_path}")

@torch.no_grad()
def validate(model, val_loader, device, criterion):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model to validate
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to use for validation
        criterion: Loss
    """

    running_loss = 0.0
    running_accuracy = 0.0

    model.eval()
    for data, target in val_loader:

        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)

        running_loss += loss.item() * data.size(0)
        running_accuracy += (output.argmax(dim=1) == target).sum().item()
    
    val_loss = running_loss / len(val_loader.dataset)
    val_accuracy = running_accuracy / len(val_loader.dataset)

    return val_loss, val_accuracy

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ResNet18()
    train(args, model, device)


if __name__ == "__main__":
    main()
