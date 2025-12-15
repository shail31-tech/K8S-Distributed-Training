import os
import sys
import zipfile
import random
from urllib.parse import urlparse

import boto3
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

# =====================================================
# Config / Hyperparameters
# =====================================================
EPOCHS = int(os.getenv("EPOCHS", "2"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
DATASET_SIZE = int(os.getenv("DATASET_SIZE", "0"))   # 0 = full
WORLD_SIZE = 1
RANK = 0

DATASET_TYPE = os.getenv("DATASET_TYPE", "mnist")    # mnist | butterfly_csv
DATASET_URI = os.getenv("DATASET_URI", "")
JOB_NAME = os.getenv("JOB_NAME", "unknown-job")
LOCAL_DATA_ROOT = "/data"

print("=== Starting PyTorch DDP-compatible training job (MNIST/custom) ===")
print(
    f"[MAIN] epochs={EPOCHS}, batch_size={BATCH_SIZE}, "
    f"dataset_size={'full' if DATASET_SIZE == 0 else DATASET_SIZE}, "
    f"dataset_type={DATASET_TYPE}, dataset_uri={DATASET_URI or 'None'}"
)
sys.stdout.flush()

# =====================================================
# DDP setup (single process, future-ready)
# =====================================================
def setup_single_process():
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(
        backend="gloo",
        rank=RANK,
        world_size=WORLD_SIZE,
    )

def cleanup():
    dist.destroy_process_group()

# =====================================================
# Model (dynamic num_classes)
# =====================================================
class Net(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        with torch.no_grad():
            x = torch.zeros(1, 3, 28, 28)
            x = self.conv2(self.conv1(x))
            self.flattened = x.numel()

        self.fc1 = nn.Linear(self.flattened, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# =====================================================
# S3 utilities
# =====================================================
def download_and_extract_s3_zip(s3_uri: str, target_dir: str) -> str:
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "dataset.zip")

    print(f"[DATA] Downloading {s3_uri}")
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, zip_path)

    print("[DATA] Extracting dataset")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    return target_dir

def find_dataset_root(root: str) -> str:
    for r, d, f in os.walk(root):
        if "train" in d and "Training_set.csv" in f:
            print(f"[DATA] Found dataset root at {r}")
            return r
    raise RuntimeError("Could not locate dataset root")

# =====================================================
# Butterfly CSV Dataset
# =====================================================
class ButterflyDataset(Dataset):
    def __init__(self, root, csv_name, img_dir, transform, label_map=None):
        csv_path = os.path.join(root, csv_name)
        self.df = pd.read_csv(csv_path, sep=None, engine="python")

        cols = list(self.df.columns)
        print(f"[DATA] CSV {csv_name} columns: {cols}")

        filename_col = cols[0]
        label_col = cols[1]

        self.df["label"] = self.df[label_col].astype(str)
        self.img_dir = os.path.join(root, img_dir)
        self.transform = transform

        if label_map is None:
            labels = sorted(self.df["label"].unique())
            self.label_map = {l: i for i, l in enumerate(labels)}
        else:
            self.label_map = label_map

        self.df["label_idx"] = self.df["label"].map(self.label_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row.iloc[0]))
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), int(row["label_idx"])

# =====================================================
# Dataset builders
# =====================================================
def get_mnist_dataset():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    if DATASET_SIZE > 0:
        ds = Subset(ds, list(range(min(DATASET_SIZE, len(ds)))))
    return ds, 10

def build_butterfly_train_val(val_ratio=0.2):
    extracted = download_and_extract_s3_zip(
        DATASET_URI, os.path.join(LOCAL_DATA_ROOT, "butterfly")
    )
    root = find_dataset_root(extracted)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    full = ButterflyDataset(
        root=root,
        csv_name="Training_set.csv",
        img_dir="train",
        transform=transform
    )

    indices = list(range(len(full)))
    random.shuffle(indices)

    if DATASET_SIZE > 0:
        indices = indices[:DATASET_SIZE]

    split = int(len(indices) * (1 - val_ratio))
    train_ds = Subset(full, indices[:split])
    val_ds = Subset(full, indices[split:])

    num_classes = len(full.label_map)
    print(f"[DATA] Butterfly classes={num_classes}, train={len(train_ds)}, val={len(val_ds)}")
    return train_ds, val_ds, num_classes

# =====================================================
# Evaluation
# =====================================================
def evaluate(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=64)
    correct = total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total
    print(f"=== Validation Accuracy: {acc:.2f}% ===")
    model.train()

# =====================================================
# Main
# =====================================================
def main():
    setup_single_process()
    device = torch.device("cpu")
    print(f"[RANK {RANK}] Using device: {device}")

    val_dataset = None

    if DATASET_TYPE == "mnist":
        train_ds, num_classes = get_mnist_dataset()
    elif DATASET_TYPE == "butterfly_csv":
        train_ds, val_dataset, num_classes = build_butterfly_train_val()
    else:
        raise RuntimeError(f"Unknown DATASET_TYPE={DATASET_TYPE}")

    sampler = DistributedSampler(train_ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)

    model = Net(num_classes=num_classes).to(device)
    ddp_model = nn.parallel.DistributedDataParallel(model)

    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        print(f"[RANK {RANK}] Epoch {epoch + 1}/{EPOCHS}")
        running = 0.0

        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(ddp_model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()

            if (i + 1) % 100 == 0:
                print(f"[RANK {RANK}] Batch {i+1}, Avg Loss {running/100:.4f}")
                running = 0.0

    os.makedirs("/model", exist_ok=True)
    torch.save(model.state_dict(), "/model/model.pt")
    print("[RANK 0] Model saved to /model/model.pt")

    if DATASET_TYPE == "butterfly_csv" and DATASET_URI.startswith("s3://"):
        parsed = urlparse(DATASET_URI)
        bucket = parsed.netloc
        prefix = "/".join(parsed.path.lstrip("/").split("/")[:-1])  # folder of dataset zip
        model_key = f"{prefix}/models/{os.getenv('JOB_NAME','run')}/model.pt"

        print(f"[S3] Uploading model to s3://{bucket}/{model_key}")
        sys.stdout.flush()
        boto3.client("s3").upload_file("/model/model.pt", bucket, model_key)
        print("[S3] Model upload complete.")
        sys.stdout.flush()


    if val_dataset:
        print("[RANK 0] Running validation")
        evaluate(ddp_model.module, val_dataset, device)

    cleanup()

if __name__ == "__main__":
    main()
