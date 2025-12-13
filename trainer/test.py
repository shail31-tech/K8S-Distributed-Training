import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

MODEL_PATH = "/model/model.pt"

def evaluate():
    print("=== Starting MNIST Evaluation ===")
    sys.stdout.flush()

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        sys.stdout.flush()
        return

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

    model = Net()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f"=== Test Accuracy: {accuracy:.2f}% ===")
    sys.stdout.flush()


if __name__ == "__main__":
    evaluate()
