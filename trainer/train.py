import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

# Import evaluation function
from test import evaluate

print("=== Starting PyTorch DDP-compatible training job (MNIST) ===")
sys.stdout.flush()

EPOCHS = int(os.getenv("EPOCHS", "2"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
DATASET_SIZE = int(os.getenv("DATASET_SIZE", "0"))

WORLD_SIZE = 1
RANK = 0

print(
    f"[MAIN] Hyperparameters -> epochs={EPOCHS}, "
    f"batch_size={BATCH_SIZE}, world_size={WORLD_SIZE}, "
    f"dataset_size={DATASET_SIZE or 'full'}"
)
sys.stdout.flush()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        with torch.no_grad():
            x = torch.zeros(1, 1, 28, 28)
            x = self.conv1(x)
            x = self.conv2(x)
            self.flattened_size = x.numel()

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def setup_single_process():
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(
        backend="gloo",
        rank=RANK,
        world_size=WORLD_SIZE,
        init_method="tcp://127.0.0.1:12355",
    )


def cleanup():
    dist.destroy_process_group()


def main():
    setup_single_process()

    device = torch.device("cpu")
    print(f"[RANK {RANK}] Using device: {device}")
    sys.stdout.flush()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = datasets.MNIST(
        "./data", download=True, train=True, transform=transform
    )

    if DATASET_SIZE > 0:
        subset_indices = list(range(min(DATASET_SIZE, len(full_dataset))))
        dataset = Subset(full_dataset, subset_indices)
    else:
        dataset = full_dataset

    sampler = DistributedSampler(
        dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True
    )

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, sampler=sampler
    )

    model = Net().to(device)
    ddp_model = nn.parallel.DistributedDataParallel(model)

    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    try:
        for epoch in range(EPOCHS):
            sampler.set_epoch(epoch)
            print(f"[RANK {RANK}] --- Epoch {epoch + 1}/{EPOCHS} ---")
            sys.stdout.flush()

            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = ddp_model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if (batch_idx + 1) % 100 == 0:
                    avg_loss = running_loss / 100
                    print(
                        f"[RANK {RANK}] Batch {batch_idx + 1}, Avg Loss: {avg_loss:.4f}"
                    )
                    sys.stdout.flush()
                    running_loss = 0.0

        print("[RANK 0] === DDP training complete ===")
        sys.stdout.flush()

        os.makedirs("/model", exist_ok=True)
        torch.save(model.state_dict(), "/model/model.pt")
        print("[RANK 0] Model saved to /model/model.pt")
        sys.stdout.flush()

        print("[RANK 0] Starting evaluation...")
        sys.stdout.flush()
        evaluate()

    finally:
        cleanup()


if __name__ == "__main__":
    main()
