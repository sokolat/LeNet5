import argparse
import time
import wandb
import yaml
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer, required
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


def get_args():
    """
    Parses and returns command-line arguments for training the LeNet5 model
    with the SDLM optimizer and RBF loss.
    """

    parser = argparse.ArgumentParser(description="Train LeNet5 with SDLM optimizer")
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="Learning rate for SDLM optimizer"
    )
    parser.add_argument(
        "--damping", type=float, default=0.02, help="Damping term for SDLM optimizer"
    )
    parser.add_argument(
        "--hessian_batch_size",
        type=int,
        default=500,
        help="batch size for hessian approximation",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.99,
        help="hyper param to prevent step size from being too large",
    )
    parser.add_argument(
        "--rub_penalty",
        type=float,
        default=1.0,
        help="penalty for non digit characters",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="flag for logging info during training",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=64, help="Batch size for testing"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--train_path", type=str, default="./data/train", help="path to train data"
    )
    parser.add_argument(
        "--test_path", type=str, default="./data/test", help="path to test data"
    )
    parser.add_argument(
        "--ascii_digits_path",
        type=str,
        default="./data/ascii_digits.yaml",
        help="path to ascii digits characters",
    )
    parser.add_argument(
        "--wandb_project_name", type=str, default="lenet5", help="name of wandb project"
    )
    parser.add_argument("--wandb_entity", type=str, help="name of wandb entity")
    parser.add_argument(
        "--output_dir", type=str, default="./runs", help="path to output directory"
    )
    parser.add_argument("--track", type=bool, default=False, help="flag for tracking")
    return parser.parse_args()


class SubSampler(nn.Module):

    """
    A custom subsampling (pooling) layer that applies average pooling
    followed by a learnable affine transformation per channel.
    """

    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.alpha = nn.Parameter(torch.ones(in_channels))
        self.bias = nn.Parameter(torch.zeros(in_channels))

    def forward(self, x):
        channels, size = x.shape[1], x.shape[2]

        assert channels == self.in_channels, (
            f"input channel: {self.in_channels} "
            f"and expected input channel: {channels} mismatch in subsampling layer."
        )
        assert size % 2 == 0, f"input size: {size} must be divisible by 2"
        assert self.kernel_size == 2

        pooled = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=2)
        out = self.alpha.view(1, -1, 1, 1) * pooled + self.bias.view(1, -1, 1, 1)

        return out


class C3Layer(nn.Module):

    """
    Implements the C3 layer of LeNet-5 with custom connection patterns between
    input and output feature maps using grouped convolutions.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.connections = {
            0: [0, 1, 2],
            1: [1, 2, 3],
            2: [2, 3, 4],
            3: [3, 4, 5],
            4: [0, 2, 3],
            5: [0, 4, 5],
            6: [0, 1, 2, 3],
            7: [1, 2, 3, 4],
            8: [2, 3, 4, 5],
            9: [0, 2, 3, 4],
            10: [1, 3, 4, 5],
            11: [0, 1, 4, 5],
            12: [0, 2, 4, 5],
            13: [0, 1, 3, 5],
            14: [1, 2, 4, 5],
            15: [0, 1, 2, 3, 4, 5],
        }
        self.filters = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=len(self.connections[i]),
                    out_channels=1,
                    kernel_size=self.kernel_size,
                )
                for i in range(self.out_channels)
            ]
        )

    def forward(self, x):
        batch_size, channels, in_size, _ = x.shape
        assert channels == self.in_channels, (
            f"input channel: {self.in_channels} "
            f"and expected input channel: {channels} mismatch in subsampling layer."
        )

        out_size = in_size - self.kernel_size + 1
        output = torch.empty(batch_size, self.out_channels, out_size, out_size)
        for i, kernel in enumerate(self.filters):
            subset = x[:, self.connections[i], :, :]
            out = kernel(subset)
            output[:, i, :, :] = out.squeeze(1)

        return output


class RBFLayer(nn.Module):

    """
    A radial basis function (RBF) layer that computes the squared Euclidean distance
    between the input and class centers.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.centers = nn.Parameter(
            torch.zeros(out_features, in_features), requires_grad=False
        )

    def forward(self, x):
        return ((x.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2).sum(dim=-1)


class RBFLoss(nn.Module):

    """
    RBF loss function that penalizes incorrect classifications and includes
    a rubbish class penalty for non-digit inputs.
    """

    def __init__(self, rubbish_penalty=required):
        super().__init__()
        self.j = rubbish_penalty

    def forward(self, outputs, targets):
        batch_size = outputs.size(0)
        correct_class_scores = outputs[torch.arange(batch_size), targets]
        log_term = torch.log(
            torch.exp(-torch.tensor(self.j)) + torch.sum(torch.exp(-outputs), dim=1)
        )
        loss = log_term + correct_class_scores
        return torch.mean(loss)


class SDLMOptimizer(Optimizer):
    """
    Implements the SDLM optimizer, which maintains a moving average of squared
    gradients to estimate a per-parameter second-order curvature.
    """

    def __init__(self, params=required, lr=required, damping=required, mu=required):
        defaults = {'lr': lr, 'damping': damping, 'steps': 0, 'mu': mu}
        super().__init__(params, defaults)

        for group in self.param_groups:
            for param in group["params"]:
                self.state[param]["hessian"] = torch.zeros_like(param)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            damping = group["damping"]
            mu = group["mu"]

            for p in group["params"]:
                grads = p.grad.data

                state = self.state[p]

                assert "hessian" in state, "hessian is missing in state"

                old_h = state["hessian"]

                new_h = mu * old_h + (1 - mu) * grads**2

                self.state[p]["hessian"] = new_h

                update = lr * grads / (damping + new_h)

                p.data.add_(-update)

            group["steps"] += 1

        return loss


class ScheduledOptimizer(LRScheduler):

    """
    Implements a manually scheduled learning rate scheduler that adjusts the learning rate
    based on the epoch number according to predefined intervals.
    """

    def __init__(self, optimizer):
        super().__init__(optimizer)
        self.optimizer = optimizer

    def step(self, epoch=None):
        if epoch is None or epoch <= 2:
            lr = 5e-4
        elif 2 < epoch <= 5:
            lr = 2e-4
        elif 5 < epoch <= 8:
            lr = 1e-4
        elif 8 < epoch <= 12:
            lr = 5e-5
        else:
            lr = 1e-5

        for params in self.optimizer.param_groups:
            params["lr"] = lr


def load_bitmaps(path):
    """
    Loads ASCII digit character representations from a YAML file and
    converts them into binary bitmap tensors for initializing RBF centers.

    Args:
        path (str): Path to the YAML file.

    Returns:
        torch.Tensor: Tensor of shape (10, 84) representing the bitmaps.
    """

    with open("file.txt", "r", encoding="utf-8") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    ascii_digits = data["digits"]

    bitmaps = torch.zeros(10, 12, 7)

    for i, digit in enumerate(ascii_digits):
        for j, row in enumerate(digit):
            for k, char in enumerate(row):
                if char == "*":
                    bitmaps[i, j, k] = 1

    return bitmaps.flatten(start_dim=1, end_dim=2)


def get_transform():
    """
    Returns the composed torchvision transform for preprocessing MNIST data.
    """

    return transforms.Compose([transforms.ToTensor(), transforms.Resize(32)])


class LeNet5(nn.Module):

    """
    Custom implementation of the LeNet-5 architecture using custom components
    such as SubSampler, C3Layer, and RBFLayer.
    """

    def __init__(self):
        super().__init__()
        self.origin = 1.7159
        self.slope = 2 / 3

        self.features = nn.ModuleList(
            [
                nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),  # C1
                SubSampler(in_channels=6, kernel_size=2),  # S2
                C3Layer(in_channels=6, out_channels=16, kernel_size=5),  # C3
                SubSampler(in_channels=16, kernel_size=2),  # S4
                nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5),  # C5
            ]
        )

        self.classifier = nn.ModuleList(
            [
                nn.Flatten(),  # Flatten before FC
                nn.Linear(120, 84),  # F6
                RBFLayer(84, 10),  # Final RBF classification
            ]
        )

    def _custom_activation(self, x):
        return self.origin * torch.tanh(self.slope * x)

    def forward(self, x):
        x = x * (1.175 + 0.1) - 0.1
        for feature in self.features:
            x = feature(x)
            x = self._custom_activation(x)

        for layer in self.classifier:
            x = layer(x)

            if isinstance(layer, nn.Linear):
                x = self._custom_activation(x)

        return x

    def init_weights(self, path):
        for feature in self.features:
            for name, params in feature.named_parameters():
                if "weight" in name:
                    fan_in = params.size(1) * params.size(2) * params.size(3)
                    params.data.uniform_(-2.4 / fan_in, 2.4 / fan_in)
                if "bias" in name:
                    params.data = torch.zeros(params.size(0))
                if "alpha" in name:
                    fan_in = params.size(-1)
                    params.data.uniform_(-2.4 / fan_in, 2.4 / fan_in)

        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                for name, params in layer.named_parameters():
                    if "weight" in name:
                        fan_in = params.size(-1)
                        params.data.uniform_(-2.4 / fan_in, 2.4 / fan_in)
                    if "bias" in name:
                        params.data = torch.zeros(params.size(0))
            elif isinstance(layer, RBFLayer):
                layer.centers.data = load_bitmaps(path)


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    train_loss,
    train_accuracy,
    val_loss,
    val_accuracy,
    checkpoint_path,
):
    """
    Saves model, optimizer, and scheduler state dictionaries and metrics to disk.

    Args:
        model (nn.Module): Trained model to save.
        optimizer (Optimizer): Optimizer used during training.
        scheduler (LRScheduler): Learning rate scheduler.
        epoch (int): Current epoch number.
        train_loss (float): Training loss at checkpoint.
        train_accuracy (float): Training accuracy at checkpoint.
        val_loss (float): Validation loss at checkpoint.
        val_accuracy (float): Validation accuracy at checkpoint.
        checkpoint_path (str): File path to save the checkpoint.
    """

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}!")


def train_model(
    hess_model, model, criterion, scheduler, device, args, logger=None, checkpoint=None
):
    """
    Trains the LeNet5 model using SDLM optimizer and RBF loss,
    optionally resuming from a checkpoint.
    Also computes Hessian approximations periodically for curvature-aware optimization.

    Args:
        hess_model (nn.Module): A separate model used for Hessian estimation.
        model (nn.Module): The main model to train.
        criterion (nn.Module): Loss function (RBFLoss).
        scheduler (LRScheduler): Custom learning rate scheduler with optimizer.
        device (torch.device): Device to train on (CPU or CUDA).
        args (Namespace): Parsed command-line arguments.
        logger (SummaryWriter, optional): TensorBoard logger.
        checkpoint (dict, optional): Previous training state to resume from.
    """

    last_epoch = 0

    model.to(device)

    transform = get_transform()

    train_dataset = datasets.MNIST(
        root=args.train_path, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=args.test_path, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True
    )
    hessian_dataloader = DataLoader(
        train_dataset, batch_size=args.hessian_batch_size, shuffle=True
    )

    hessian_dataloader_iterator = iter(hessian_dataloader)

    if checkpoint is not None:
        if args.verbose:
            print("Resuming training from previous model checkpoint...")
        model.load_state_dict(checkpoint["model_state_dict"])
        if args.verbose:
            print("Reloading previous optimizer state...")
        scheduler.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        last_epoch = checkpoint["epoch"]
    else:
        if args.verbose:
            print("Initializing weights for new model...")
        model.init_weights(args.ascii_digits_path)
        hess_model.init_weights(args.ascii_digits_path)

    if args.verbose:
        print()
        print(f"Starting training at epoch {last_epoch + 1}")

    global_step = 0

    model.train()
    for epoch in range(last_epoch + 1, args.num_epochs + 1):
        scheduler.step(epoch)

        if args.verbose:
            print(f"Estimating new hessian values for epoch {epoch}...")

        try:
            data, target = next(hessian_dataloader_iterator)
            data, target = data.to(device), target.to(device)

        except StopIteration:
            hessian_dataloader_iterator = iter(hessian_dataloader)
            data, target = next(hessian_dataloader_iterator)
            data, target = data.to(device), target.to(device)

        pred = hess_model(data)
        loss = criterion(pred, target)

        for params in hess_model.parameters():
            if params.requires_grad:
                p_grad = torch.autograd.grad(loss, params, create_graph=True)[0]
                second_order_der = torch.autograd.grad(
                    p_grad.sum(), params, retain_graph=True
                )[0]

                scheduler.optimizer.state[params]["hessian"] = second_order_der

        running_loss = 0.0
        running_correct = 0.0
        running_step = 0

        for step, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            scheduler.optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, target)
            loss.backward()
            scheduler.optimizer.step()

            running_loss += loss.item() * data.size(0)
            running_correct += (pred.argmin(dim=-1) == target).sum().item()
            running_step += data.size(0)
            global_step += 1

            if step % 100 == 99:
                running_avg_loss = running_loss / running_step
                running_accuracy = running_correct / running_step
                if args.verbose:
                    print(
                        f"Step: {step + 1} running average loss: {running_avg_loss:.4f}"
                        f" running error rate: {1 - running_accuracy:.4f} "
                    )
                    print("______________________________________")

        epoch_loss = running_loss / len(train_dataset)
        epoch_error_rate = 1 - running_correct / len(train_dataset)

        val_loss, val_error_rate = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch} loss: {epoch_loss:.4f} error rate: {epoch_error_rate:.4f}"
        )
        print(f"Validation Loss: {val_loss:.4f} error rate: {val_error_rate:.4f}")

        logger.add_scalar("Error Rate/Train", epoch_error_rate, epoch)
        logger.add_scalar("Error Rate/Validation", val_error_rate, epoch)

        logger.add_scalar("Loss/Train", epoch_loss, epoch)
        logger.add_scalar("Loss/Validation", val_loss, epoch)


@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    """
    Evaluates the trained model on a validation set.

    Args:
        model (nn.Module): Trained model to evaluate.
        val_loader (DataLoader): DataLoader for validation set.
        criterion (nn.Module): Loss function (RBFLoss).
        device (torch.device): Device to run evaluation on.

    Returns:
        tuple: Average validation loss and error rate.
    """

    val_loss = 0.0
    correct = 0
    total = 0
    model.eval()
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)

        pred = model(data)
        loss = criterion(pred, target)

        val_loss += loss.item() * data.size(0)
        correct += (pred.argmin(dim=-1) == target).sum().item()
        total += data.size(0)

    avg_loss = val_loss / total
    error_rate = 1 - correct / total
    return avg_loss, error_rate


def main():
    """
    Main function to parse arguments, initialize training components,
    and start the training loop.
    """

    args = get_args()

    run_name = f"run-{int(time.time())}_lenet5_mnist"

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            config=vars(args),
            entity=args.wandb_entity,
            sync_tensorboard=True,
            save_code=True,
            name=run_name,
        )

    logger = SummaryWriter(f"{args.output_dir}/{run_name}")

    device = None
    if args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    model = LeNet5()
    hess_model = LeNet5()

    criterion = RBFLoss(rubbish_penalty=args.rub_penalty)
    params = [param for param in model.parameters() if param.requires_grad]

    optimizer = SDLMOptimizer(
        params=params, lr=args.lr, damping=args.damping, mu=args.mu
    )
    scheduler = ScheduledOptimizer(optimizer)

    train_model(hess_model, model, criterion, scheduler, device, args, logger)


if __name__ == "__main__":
    main()
